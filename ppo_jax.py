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
# TODO(Phase 2): PPOConfig dataclass, GYMNAX_DEFAULTS / BRAX_DEFAULTS presets.


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
