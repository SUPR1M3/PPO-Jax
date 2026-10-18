"""PPO-Clip (Schulman et al. 2017, "Proximal Policy Optimization Algorithms") in pure JAX.

The whole training loop (rollouts + updates) is compiled under jax.jit, in the style of
PureJaxRL. Supports discrete (Gymnax CartPole) and continuous (Gymnax Pendulum, Brax)
action spaces.

Layout (each section names the part of the paper it implements):
    1. Config                 -- hyperparameters (paper Sec. 6, Tables 3-5)
    2. Distributions          -- pi_theta(a|s) for discrete / continuous actions
    3. Network                -- actor-critic (paper Sec. 5)
    4. Environment wrappers   -- vectorized envs, auto-reset, logging, normalization
    5. Transition             -- rollout storage (paper Algorithm 1, "run policy for T timesteps")
    6. GAE                    -- advantage estimation (paper eq. 11-12)
    7. PPO loss               -- clipped surrogate + value + entropy (paper eq. 7, 9)
    8. make_train             -- the full training loop (paper Algorithm 1)
    9. Plotting / CLI
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, NamedTuple

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState


# =============================================================================
# 1. Config -- hyperparameters (paper Sec. 6, Tables 3-5)
# =============================================================================
@dataclass(frozen=True)
class PPOConfig:
    # --- Environment ---
    env_name: str = "CartPole-v1"
    env_backend: str = "gymnax"  # "gymnax" | "brax"
    continuous: bool = False  # Gaussian policy if True, Categorical otherwise
    brax_episode_length: int = 1000  # time limit for Brax envs (sets info["truncation"])

    # --- Rollout (paper Algorithm 1: N actors, T timesteps each) ---
    num_envs: int = 4  # N
    num_steps: int = 128  # T
    total_timesteps: int = 500_000

    # --- Optimization (paper Sec. 3, 5) ---
    lr: float = 2.5e-4
    anneal_lr: bool = True  # linearly decay lr to 0 over training
    update_epochs: int = 4  # K
    num_minibatches: int = 4
    max_grad_norm: float = 0.5

    # --- PPO objective (paper eq. 7, 9) ---
    gamma: float = 0.99  # discount
    gae_lambda: float = 0.95  # GAE lambda
    clip_eps: float = 0.2  # epsilon in clip(r, 1-eps, 1+eps)
    vf_coef: float = 0.5  # c1
    ent_coef: float = 0.01  # c2
    clip_vloss: bool = False  # optional value-function clipping

    # --- Normalization wrappers ---
    normalize_obs: bool = False
    normalize_reward: bool = False

    # --- Network ---
    hidden_size: int = 64
    num_layers: int = 2
    shared_trunk: bool = False

    # --- Run ---
    seed: int = 0
    num_seeds: int = 1
    debug_print: bool = False  # live printing via jax.debug.callback

    def __post_init__(self):
        assert self.env_backend in ("gymnax", "brax"), f"unknown backend {self.env_backend!r}"
        assert self.num_envs > 0 and self.num_steps > 0 and self.update_epochs > 0
        assert self.batch_size % self.num_minibatches == 0, (
            f"batch_size (num_envs * num_steps = {self.batch_size}) must be divisible "
            f"by num_minibatches ({self.num_minibatches})"
        )
        assert self.num_updates > 0, (
            f"total_timesteps ({self.total_timesteps}) is smaller than one batch ({self.batch_size})"
        )
        assert 0.0 < self.gamma <= 1.0 and 0.0 <= self.gae_lambda <= 1.0
        assert self.clip_eps > 0.0

    # --- Derived values ---
    @property
    def batch_size(self) -> int:
        """Samples collected per update: N * T."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        """Outer-loop iterations. Any remainder of total_timesteps is dropped."""
        return self.total_timesteps // self.batch_size


# Defaults for small classic-control tasks (paper Table 5 / PureJaxRL ppo.py).
GYMNAX_DEFAULTS = dict(
    env_backend="gymnax",
    lr=2.5e-4,
    num_envs=4,
    num_steps=128,
    total_timesteps=500_000,
    update_epochs=4,
    num_minibatches=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    anneal_lr=True,
    hidden_size=64,
)

# Defaults for massively parallel Brax locomotion tasks (paper Table 3 adapted for 2048 envs).
BRAX_DEFAULTS = dict(
    env_backend="brax",
    continuous=True,
    lr=3e-4,
    num_envs=2048,
    num_steps=10,
    total_timesteps=50_000_000,
    update_epochs=4,
    num_minibatches=32,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    anneal_lr=True,
    normalize_obs=True,
    normalize_reward=True,
    hidden_size=256,
)

# Per-env preset: (backend defaults, env-specific overrides).
ENV_PRESETS = {
    "CartPole-v1": (GYMNAX_DEFAULTS, {}),
    # TODO(Phase 11): tune Pendulum (likely needs normalization, more envs/steps).
    "Pendulum-v1": (GYMNAX_DEFAULTS, {"continuous": True}),
    **{name: (BRAX_DEFAULTS, {}) for name in ("halfcheetah", "ant", "hopper", "walker2d", "humanoid")},
}


def make_config(env_name: str, **overrides) -> PPOConfig:
    """Build a config from the env's preset, then apply user overrides.

    Unknown override keys raise a TypeError (via dataclasses.replace)."""
    if env_name not in ENV_PRESETS:
        raise ValueError(f"no preset for {env_name!r}; known: {sorted(ENV_PRESETS)}")
    defaults, env_overrides = ENV_PRESETS[env_name]
    base = PPOConfig(env_name=env_name, **{**defaults, **env_overrides})
    return dataclasses.replace(base, **overrides)


# =============================================================================
# 2. Distributions -- the stochastic policy pi_theta(a|s)
# =============================================================================
# TODO(Phase 3): Categorical / diagonal Gaussian (log_prob summed over action dims).


# =============================================================================
# 3. Network -- actor-critic (paper Sec. 5: separate or shared policy/value nets)
# =============================================================================
# TODO(Phase 4): ActorCritic module with orthogonal init and state-independent log_std.


# =============================================================================
# 4. Environment wrappers -- vectorized envs with auto-reset, logging, normalization
# =============================================================================
# TODO(Phase 5): Brax adapter, FlattenObservation, ClipAction, AutoResetWrapper,
#                LogWrapper, VecEnv, NormalizeVecObservation, NormalizeVecReward.


# =============================================================================
# 5. Transition -- storage for one rollout (paper Algorithm 1: "run policy for T timesteps")
# =============================================================================
# TODO(Phase 6): Transition NamedTuple.


# =============================================================================
# 6. GAE -- generalized advantage estimation (paper eq. 11-12)
# =============================================================================
# TODO(Phase 7): calculate_gae via reverse lax.scan.


# =============================================================================
# 7. PPO loss -- clipped surrogate objective L^CLIP + c1 L^VF - c2 S (paper eq. 7, 9)
# =============================================================================
# TODO(Phase 8): clipped_surrogate, ppo_loss.


# =============================================================================
# 8. make_train -- the full training loop (paper Algorithm 1)
# =============================================================================
# TODO(Phase 9): make_train(config) -> train(rng) -> (runner_state, metrics).


# =============================================================================
# 9. Plotting / CLI
# =============================================================================
# TODO(Phase 11): plotting, timing, argparse entry point.

if __name__ == "__main__":
    print(f"JAX {jax.__version__} on {jax.devices()}")
