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
# Hand-written (rather than distrax) so the log-prob / entropy math is visible here.
# Both are pytrees, so they can be returned from jitted / vmapped functions.
LOG_2PI = float(np.log(2.0 * np.pi))


@struct.dataclass
class Categorical:
    """Discrete policy over action_dim actions, parameterized by unnormalized logits."""

    logits: jnp.ndarray  # (..., action_dim)

    @property
    def log_probs(self) -> jnp.ndarray:
        return jax.nn.log_softmax(self.logits, axis=-1)

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        return jax.random.categorical(seed, self.logits, axis=-1)

    def log_prob(self, action: jnp.ndarray) -> jnp.ndarray:
        idx = action.astype(jnp.int32)[..., None]
        return jnp.take_along_axis(self.log_probs, idx, axis=-1)[..., 0]

    def entropy(self) -> jnp.ndarray:
        return -jnp.sum(jnp.exp(self.log_probs) * self.log_probs, axis=-1)


@struct.dataclass
class DiagGaussian:
    """Continuous policy N(mean, diag(exp(log_std))^2).

    log_prob and entropy are SUMMED over action dimensions: the dims are independent,
    so log pi(a|s) = sum_i log N(a_i; mean_i, std_i). Actions are sampled unbounded;
    clipping to env bounds happens in the env wrapper, and log_prob is evaluated on the
    unclipped sample (the action the policy actually chose)."""

    mean: jnp.ndarray  # (..., action_dim)
    log_std: jnp.ndarray  # (action_dim,) state-independent, broadcast against mean

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        return self.mean + jnp.exp(self.log_std) * jax.random.normal(seed, self.mean.shape)

    def log_prob(self, action: jnp.ndarray) -> jnp.ndarray:
        z = (action - self.mean) / jnp.exp(self.log_std)
        return jnp.sum(-0.5 * z**2 - self.log_std - 0.5 * LOG_2PI, axis=-1)

    def entropy(self) -> jnp.ndarray:
        log_std = jnp.broadcast_to(self.log_std, self.mean.shape)
        return jnp.sum(0.5 + 0.5 * LOG_2PI + log_std, axis=-1)


# =============================================================================
# 3. Network -- actor-critic (paper Sec. 5: separate or shared policy/value nets)
# =============================================================================
# Orthogonal init with gains from the PPO implementation-details literature:
# sqrt(2) for hidden layers, 0.01 for the policy head (near-uniform / near-zero-mean initial
# policy), 1.0 for the value head. All biases start at zero.
def _dense(features: int, gain: float, name: str | None = None) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=nn.initializers.orthogonal(gain),
        bias_init=nn.initializers.zeros,
        name=name,
    )


class MLP(nn.Module):
    """num_layers x hidden_size hidden layers with an activation after each."""

    hidden_size: int = 64
    num_layers: int = 2
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = {"tanh": nn.tanh, "relu": nn.relu}[self.activation]
        for _ in range(self.num_layers):
            x = act(_dense(self.hidden_size, np.sqrt(2))(x))
        return x


class ActorCritic(nn.Module):
    """Policy pi_theta(a|s) and value V(s).

    By default the actor and critic are separate MLPs (no parameter sharing), so the
    value loss coefficient c1 only matters for the shared-trunk variant (paper Sec. 5:
    "If using a neural network architecture that shares parameters between the policy
    and value function, we must use a loss function that combines ...").

    Returns (pi, value) where value has the batch shape of obs (trailing dim squeezed)."""

    action_dim: int
    continuous: bool = False
    hidden_size: int = 64
    num_layers: int = 2
    shared_trunk: bool = False
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        mlp = lambda name: MLP(self.hidden_size, self.num_layers, self.activation, name=name)
        if self.shared_trunk:
            actor_h = critic_h = mlp("trunk")(obs)
        else:
            actor_h = mlp("actor_trunk")(obs)
            critic_h = mlp("critic_trunk")(obs)

        # Policy head: logits (discrete) or Gaussian mean (continuous).
        out = _dense(self.action_dim, 0.01, name="actor_head")(actor_h)
        if self.continuous:
            # State-independent learnable log std, initialised to 0 (std = 1).
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = DiagGaussian(out, log_std)
        else:
            pi = Categorical(out)

        value = _dense(1, 1.0, name="critic_head")(critic_h)
        return pi, jnp.squeeze(value, axis=-1)


def make_network(config: "PPOConfig", action_dim: int) -> ActorCritic:
    return ActorCritic(
        action_dim=action_dim,
        continuous=config.continuous,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        shared_trunk=config.shared_trunk,
    )


# =============================================================================
# 4. Environment wrappers -- vectorized envs with auto-reset, logging, normalization
# =============================================================================
# Every layer exposes the same functional interface (env params are bound in the adapter):
#     reset(key)                -> (obs, state)
#     step(key, state, action)  -> (obs, state, reward, done, info)
# All state is an explicit pytree; wrappers hold only static configuration.
#
# Stack (inner -> outer), built by make_env:
#     GymnaxAdapter | BraxAdapter   single env, NO auto-reset; info has terminated/truncated
#     FlattenObservation            (only if obs is not 1-D)
#     ClipAction                    (continuous only)
#     AutoResetWrapper              resets on done; info["terminal_obs"] = pre-reset obs
#     LogWrapper                    episodic return/length (raw rewards)
#     VecEnv                        jax.vmap over num_envs
#     NormalizeVecObservation       stats shared across envs (optional)
#     NormalizeVecReward            stats shared across envs (optional)
#
# Terminated vs truncated: `done` = terminated | truncated marks an episode boundary.
# Only `terminated` (a true terminal state) zeroes the bootstrap V(s_{t+1}); on a time-limit
# truncation, GAE bootstraps from V(info["terminal_obs"]) instead (see calculate_gae).


class Wrapper:
    """Base wrapper: forwards attributes (obs_shape, action_dim, ...) to the inner env."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, key):
        return self._env.reset(key)

    def step(self, key, state, action):
        return self._env.step(key, state, action)


class GymnaxAdapter:
    """Wraps a gymnax env using its non-resetting step_env / reset_env.

    Gymnax merges termination and the time limit into a single `done` (is_terminal checks
    `state.time >= params.max_steps_in_episode`). We recover pure termination by evaluating
    is_terminal again with the time limit disabled; truncated = done & ~terminated. This is
    exact even when the pole falls on the very last step (counted as terminated). It assumes
    the env's time limit is expressed via params.max_steps_in_episode, as in all gymnax
    classic-control envs."""

    def __init__(self, env_name: str, params=None):
        self._env, default_params = gymnax.make(env_name)
        self.params = default_params if params is None else params
        self._no_time_limit = self.params.replace(max_steps_in_episode=jnp.iinfo(jnp.int32).max)

        self.obs_shape = tuple(self._env.observation_space(self.params).shape)
        space = self._env.action_space(self.params)
        self.continuous = isinstance(space, gymnax.environments.spaces.Box)
        if self.continuous:
            self.action_dim = int(np.prod(space.shape))
            self.action_low = jnp.broadcast_to(jnp.asarray(space.low, jnp.float32), space.shape)
            self.action_high = jnp.broadcast_to(jnp.asarray(space.high, jnp.float32), space.shape)
        else:
            self.action_dim = int(space.n)

    def reset(self, key):
        return self._env.reset_env(key, self.params)

    def step(self, key, state, action):
        obs, state, reward, done, _ = self._env.step_env(key, state, action, self.params)
        terminated = self._env.is_terminal(state, self._no_time_limit)
        done = jnp.asarray(done, bool)
        info = {"terminated": terminated & done, "truncated": done & ~terminated}
        return obs, state, jnp.asarray(reward, jnp.float32), done, info


class BraxAdapter:
    """Wraps a Brax env (with Brax's EpisodeWrapper for the time limit, but none of its
    auto-reset / vmap wrappers -- ours are used so both backends share one code path).

    EpisodeWrapper sets info["truncation"] = 1 only when the time limit is hit and the
    env did not also terminate, so terminated = done & ~truncation."""

    def __init__(self, env_name: str, episode_length: int = 1000):
        from brax import envs  # imported lazily: slow import, only needed for Brax runs

        self._env = envs.create(env_name, episode_length=episode_length, auto_reset=False)
        self.obs_shape = (self._env.observation_size,)
        self.action_dim = self._env.action_size
        self.continuous = True
        self.action_low = -jnp.ones(self.action_dim)
        self.action_high = jnp.ones(self.action_dim)

    def reset(self, key):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action):  # Brax dynamics are deterministic; key unused
        state = self._env.step(state, action)
        done = state.done.astype(bool)
        truncated = state.info["truncation"].astype(bool) & done
        info = {"terminated": done & ~truncated, "truncated": truncated}
        return state.obs, state, state.reward.astype(jnp.float32), done, info


class FlattenObservation(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_shape = (int(np.prod(env.obs_shape)),)

    def reset(self, key):
        obs, state = self._env.reset(key)
        return obs.reshape(-1), state

    def step(self, key, state, action):
        obs, state, reward, done, info = self._env.step(key, state, action)
        return obs.reshape(-1), state, reward, done, info


class ClipAction(Wrapper):
    """Clip continuous actions to the env's bounds before stepping. The agent's stored
    action (and its log_prob) remain the unclipped Gaussian sample."""

    def step(self, key, state, action):
        action = jnp.clip(action, self._env.action_low, self._env.action_high)
        return self._env.step(key, state, action)


class AutoResetWrapper(Wrapper):
    """Reset the env when an episode ends. The returned obs is the first obs of the new
    episode; the true final obs s_T is kept in info["terminal_obs"] for bootstrapping on
    truncation. (Resetting is computed every step and selected with `where`, since control
    flow must be static under jit.)"""

    def step(self, key, state, action):
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self._env.step(key_step, state, action)
        obs_re, state_re = self._env.reset(key_reset)
        state = jax.tree.map(lambda re, st: jnp.where(done, re, st), state_re, state_st)
        obs = jnp.where(done, obs_re, obs_st)
        return obs, state, reward, done, {**info, "terminal_obs": obs_st}


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray


class LogWrapper(Wrapper):
    """Track undiscounted episodic return/length of the raw (un-normalized) reward.
    info["returned_episode_*"] holds the stats of the most recently completed episode;
    info["returned_episode"] flags the steps where an episode just finished."""

    def reset(self, key):
        obs, env_state = self._env.reset(key)
        zf, zi = jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32)
        return obs, LogEnvState(env_state, zf, zi, zf, zi, zi)

    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action)
        new_return = state.episode_returns + reward
        new_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.where(done, 0.0, new_return),
            episode_lengths=jnp.where(done, 0, new_length),
            returned_episode_returns=jnp.where(done, new_return, state.returned_episode_returns),
            returned_episode_lengths=jnp.where(done, new_length, state.returned_episode_lengths),
            timestep=state.timestep + 1,
        )
        info = {
            **info,
            "returned_episode_returns": state.returned_episode_returns,
            "returned_episode_lengths": state.returned_episode_lengths,
            "returned_episode": done,
        }
        return obs, state, reward, done, info


class VecEnv(Wrapper):
    """num_envs independent copies via jax.vmap. Takes ONE key and splits it per env."""

    def __init__(self, env, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key):
        return jax.vmap(self._env.reset)(jax.random.split(key, self.num_envs))

    def step(self, key, state, action):
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self._env.step)(keys, state, action)


@struct.dataclass
class RunningMeanStd:
    """Running mean/variance merged batch-by-batch (Chan et al. parallel algorithm)."""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def create(cls, shape=()):
        return cls(jnp.zeros(shape), jnp.ones(shape), jnp.asarray(1e-4))

    def update(self, x: jnp.ndarray) -> "RunningMeanStd":
        """x has a leading batch dimension."""
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        return RunningMeanStd(mean, m2 / total, total)


@struct.dataclass
class NormObsState:
    env_state: Any
    obs_rms: RunningMeanStd


class NormalizeVecObservation(Wrapper):
    """obs <- clip((obs - mean) / sqrt(var + 1e-8), -10, 10), with running stats over all
    envs stored in the env state. info["terminal_obs"] is normalized with the same stats."""

    def __init__(self, env, clip: float = 10.0):
        super().__init__(env)
        self.clip = clip

    def _normalize(self, obs, rms):
        return jnp.clip((obs - rms.mean) / jnp.sqrt(rms.var + 1e-8), -self.clip, self.clip)

    def reset(self, key):
        obs, env_state = self._env.reset(key)
        rms = RunningMeanStd.create(obs.shape[1:]).update(obs)
        return self._normalize(obs, rms), NormObsState(env_state, rms)

    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action)
        rms = state.obs_rms.update(obs)
        info = {**info, "terminal_obs": self._normalize(info["terminal_obs"], rms)}
        return self._normalize(obs, rms), NormObsState(env_state, rms), reward, done, info


@struct.dataclass
class NormRewardState:
    env_state: Any
    return_rms: RunningMeanStd
    discounted_return: jnp.ndarray  # (num_envs,)


class NormalizeVecReward(Wrapper):
    """reward <- clip(reward / sqrt(Var[R] + 1e-8), -10, 10), where R_t = gamma R_{t-1} + r_t
    is each env's running discounted return (reset at episode end). Rewards are scaled,
    not centred, so the sign of the reward is preserved."""

    def __init__(self, env, gamma: float, clip: float = 10.0):
        super().__init__(env)
        self.gamma, self.clip = gamma, clip

    def reset(self, key):
        obs, env_state = self._env.reset(key)
        ret = jnp.zeros(obs.shape[0])
        return obs, NormRewardState(env_state, RunningMeanStd.create(), ret)

    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action)
        ret = state.discounted_return * self.gamma + reward
        rms = state.return_rms.update(ret)
        reward = jnp.clip(reward / jnp.sqrt(rms.var + 1e-8), -self.clip, self.clip)
        ret = jnp.where(done, 0.0, ret)
        return obs, NormRewardState(env_state, rms, ret), reward, done, info


def make_env(config: "PPOConfig"):
    """Build the full wrapper stack described above."""
    if config.env_backend == "gymnax":
        env = GymnaxAdapter(config.env_name)
    else:
        env = BraxAdapter(config.env_name, config.brax_episode_length)
    assert env.continuous == config.continuous, (
        f"{config.env_name} has a {'continuous' if env.continuous else 'discrete'} action "
        f"space but config.continuous={config.continuous}"
    )
    if len(env.obs_shape) > 1:
        env = FlattenObservation(env)
    if env.continuous:
        env = ClipAction(env)
    env = AutoResetWrapper(env)
    env = LogWrapper(env)
    env = VecEnv(env, config.num_envs)
    if config.normalize_obs:
        env = NormalizeVecObservation(env)
    if config.normalize_reward:
        env = NormalizeVecReward(env, config.gamma)
    return env


# =============================================================================
# 5. Transition -- storage for one rollout (paper Algorithm 1: "run policy for T timesteps")
# =============================================================================
class Transition(NamedTuple):
    """One environment step for all envs. A rollout stacks these to shape (num_steps, num_envs, ...)."""

    obs: jnp.ndarray  # s_t, the observation the action was chosen from
    action: jnp.ndarray  # a_t ~ pi_theta_old(.|s_t)  (unclipped sample)
    log_prob: jnp.ndarray  # log pi_theta_old(a_t|s_t), the "old" log-prob in the ratio
    value: jnp.ndarray  # V_old(s_t)
    reward: jnp.ndarray  # r_t (normalized if reward normalization is on)
    done: jnp.ndarray  # episode ended at this step (terminated | truncated)
    terminated: jnp.ndarray  # true terminal state: bootstrap value is 0
    bootstrap_value: jnp.ndarray  # V_old(s_{t+1}) of the pre-reset obs; used where truncated
    info: dict  # LogWrapper episode stats etc.


def collect_rollout(env, apply_fn, params, env_state, last_obs, rng, num_steps: int):
    """Run the current (old) policy for num_steps in every env (paper Algorithm 1:
    "Run policy pi_theta_old in environment for T timesteps").

    Returns (env_state, last_obs, rng, traj) where traj is a stacked Transition."""

    def _env_step(carry, _):
        env_state, last_obs, rng = carry
        rng, act_key, step_key = jax.random.split(rng, 3)

        pi, value = apply_fn(params, last_obs)
        action = pi.sample(seed=act_key)
        log_prob = pi.log_prob(action)

        obs, env_state, reward, done, info = env.step(step_key, env_state, action)

        # V of the true next state s_{t+1}. Differs from V(obs) only where the env was
        # auto-reset; GAE uses it to bootstrap through time-limit truncations.
        _, bootstrap_value = apply_fn(params, info["terminal_obs"])
        info = {k: v for k, v in info.items() if k != "terminal_obs"}  # don't store obs twice

        transition = Transition(
            obs=last_obs,
            action=action,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            terminated=info["terminated"],
            bootstrap_value=bootstrap_value,
            info=info,
        )
        return (env_state, obs, rng), transition

    (env_state, last_obs, rng), traj = jax.lax.scan(
        _env_step, (env_state, last_obs, rng), None, length=num_steps
    )
    return env_state, last_obs, rng, traj


# =============================================================================
# 6. GAE -- generalized advantage estimation (paper eq. 11-12)
# =============================================================================
def calculate_gae(traj: Transition, last_val: jnp.ndarray, gamma: float, gae_lambda: float):
    """Generalized advantage estimation, computed backwards in time with a reverse scan:

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)                       (paper eq. 12)
        A_t     = delta_t + gamma * lambda * A_{t+1}                      (paper eq. 11)

    Episode boundaries (done_t = terminated | truncated):
      - A_{t+1} belongs to the next episode, so the recursion is cut: (1 - done_t).
      - V(s_{t+1}) is 0 if the episode terminated, and V(terminal obs) (bootstrap_value)
        if it was truncated by a time limit -- the episode could have continued.
    The last step bootstraps from last_val = V(s_T) of the obs after the rollout.

    Returns (advantages, targets) with targets = advantages + values (the value-function
    regression targets). Both are treated as constants by the loss (stop_gradient)."""

    def _step(carry, t: Transition):
        gae, next_value = carry  # A_{t+1}, V(s_{t+1}) if the episode continues
        next_value = jnp.where(t.done, jnp.where(t.terminated, 0.0, t.bootstrap_value), next_value)
        delta = t.reward + gamma * next_value - t.value
        gae = delta + gamma * gae_lambda * (1.0 - t.done) * gae
        return (gae, t.value), gae

    _, advantages = jax.lax.scan(_step, (jnp.zeros_like(last_val), last_val), traj, reverse=True)
    targets = advantages + traj.value
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(targets)


# =============================================================================
# 7. PPO loss -- clipped surrogate objective L^CLIP + c1 L^VF - c2 S (paper eq. 7, 9)
# =============================================================================
class Batch(NamedTuple):
    """Flattened training samples (leading dim = batch or minibatch size)."""

    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray  # log pi_old(a|s)
    value: jnp.ndarray  # V_old(s), used for optional value clipping
    advantages: jnp.ndarray
    targets: jnp.ndarray


def clipped_surrogate(ratio, advantages, clip_eps):
    """Negated clipped surrogate objective (paper eq. 7):

        L^CLIP = E[ min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) ]

    Taking the min makes this a pessimistic bound: once the ratio moves past 1+eps (A>0)
    or below 1-eps (A<0), the clipped term is selected and its gradient is zero, so there
    is no incentive to move the policy further in that direction."""
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -jnp.mean(jnp.minimum(unclipped, clipped))


def ppo_loss(params, apply_fn, batch: Batch, config: PPOConfig):
    """Total loss = -L^CLIP + c1 L^VF - c2 S  (paper eq. 9, written as a loss to minimize).

    Returns (total_loss, aux) where aux holds the components and diagnostics."""
    pi, value = apply_fn(params, batch.obs)
    log_prob = pi.log_prob(batch.action)

    # Old log-probs and advantages are constants: no gradient flows into them.
    old_log_prob = jax.lax.stop_gradient(batch.log_prob)
    advantages = jax.lax.stop_gradient(batch.advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # per minibatch

    # Probability ratio r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)  (paper eq. 6)
    log_ratio = log_prob - old_log_prob
    ratio = jnp.exp(log_ratio)
    pg_loss = clipped_surrogate(ratio, advantages, config.clip_eps)

    # Value loss L^VF = (V(s) - V_targ)^2 (paper eq. 9), with the conventional 0.5 factor.
    if config.clip_vloss:
        # Optional: limit how far V moves from V_old per update, mirroring the policy clip.
        value_clipped = batch.value + jnp.clip(value - batch.value, -config.clip_eps, config.clip_eps)
        v_loss = 0.5 * jnp.mean(
            jnp.maximum((value - batch.targets) ** 2, (value_clipped - batch.targets) ** 2)
        )
    else:
        v_loss = 0.5 * jnp.mean((value - batch.targets) ** 2)

    # Entropy bonus S[pi_theta] encourages exploration (paper eq. 9).
    entropy = jnp.mean(pi.entropy())

    total_loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * entropy

    # Diagnostics: KL(old || new) estimator (r - 1) - log r (unbiased, always >= 0),
    # and the fraction of samples whose ratio is outside the clip range.
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)
    aux = dict(pg_loss=pg_loss, v_loss=v_loss, entropy=entropy, approx_kl=approx_kl, clip_frac=clip_frac)
    return total_loss, aux


def make_optimizer(config: PPOConfig):
    """Adam with global-norm gradient clipping and optional linear lr annealing to 0.

    Returns (optax transform, lr_fn) where lr_fn(step) gives the lr after `step`
    optimizer updates (for logging)."""
    if config.anneal_lr:
        total_steps = config.num_updates * config.update_epochs * config.num_minibatches
        lr_fn = optax.linear_schedule(config.lr, 0.0, transition_steps=total_steps)
    else:
        lr_fn = optax.constant_schedule(config.lr)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=lr_fn, eps=1e-5),
    )
    return tx, lr_fn


def make_minibatches(batch: Batch, key: jax.Array, num_minibatches: int) -> Batch:
    """Shuffle a flat batch and split it into num_minibatches equal minibatches
    (leading dims: (num_minibatches, minibatch_size, ...))."""
    batch_size = batch.obs.shape[0]
    perm = jax.random.permutation(key, batch_size)
    return jax.tree.map(
        lambda x: x[perm].reshape((num_minibatches, -1) + x.shape[1:]), batch
    )


def ppo_update(train_state: TrainState, traj: Transition, advantages, targets, rng, config: PPOConfig):
    """Optimize the surrogate for K epochs with minibatch SGD (paper Algorithm 1:
    "Optimize surrogate L wrt theta, with K epochs and minibatch size M <= NT").

    Returns (train_state, rng, metrics) with each metric of shape (update_epochs, num_minibatches)."""
    batch = Batch(traj.obs, traj.action, traj.log_prob, traj.value, advantages, targets)
    # (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
    batch = jax.tree.map(lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch)

    def _update_minibatch(train_state, minibatch):
        grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        (loss, aux), grads = grad_fn(train_state.params, train_state.apply_fn, minibatch, config)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, {"total_loss": loss, **aux}

    def _update_epoch(carry, _):
        train_state, rng = carry
        rng, perm_key = jax.random.split(rng)
        minibatches = make_minibatches(batch, perm_key, config.num_minibatches)
        train_state, metrics = jax.lax.scan(_update_minibatch, train_state, minibatches)
        return (train_state, rng), metrics

    (train_state, rng), metrics = jax.lax.scan(
        _update_epoch, (train_state, rng), None, length=config.update_epochs
    )
    return train_state, rng, metrics


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
