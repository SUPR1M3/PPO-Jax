"""Unit tests and sanity checks for ppo_jax.py (Spec Sec. 8)."""

import pytest

import ppo_jax


# ---------------------------------------------------------------------------
# Config (Spec Sec. 6)
# ---------------------------------------------------------------------------
def test_gymnax_derived_values():
    c = ppo_jax.make_config("CartPole-v1")
    assert c.batch_size == 4 * 128
    assert c.minibatch_size == 128
    assert c.num_updates == 500_000 // 512


def test_brax_derived_values():
    c = ppo_jax.make_config("halfcheetah")
    assert c.continuous and c.normalize_obs and c.normalize_reward
    assert c.batch_size == 2048 * 10
    assert c.minibatch_size == 640
    assert c.num_updates == 50_000_000 // 20480


def test_overrides_and_presets():
    assert ppo_jax.make_config("Pendulum-v1").continuous
    c = ppo_jax.make_config("CartPole-v1", lr=1e-3, shared_trunk=True)
    assert c.lr == 1e-3 and c.shared_trunk
    with pytest.raises(TypeError):
        ppo_jax.make_config("CartPole-v1", not_a_field=1)
    with pytest.raises(ValueError):
        ppo_jax.make_config("NotAnEnv-v0")


def test_invalid_configs_rejected():
    with pytest.raises(AssertionError):  # 4 * 128 = 512 not divisible by 3
        ppo_jax.make_config("CartPole-v1", num_minibatches=3)
    with pytest.raises(AssertionError):  # less than one batch
        ppo_jax.make_config("CartPole-v1", total_timesteps=100)
    with pytest.raises(AssertionError):
        ppo_jax.make_config("CartPole-v1", env_backend="gym")
