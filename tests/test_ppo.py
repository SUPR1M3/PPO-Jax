"""Unit tests and sanity checks for ppo_jax.py (Spec Sec. 8).

Run all:          pytest
Skip slow ones:   pytest -m "not slow"

Spec Sec. 8 checklist -> test:
  ratio == 1 and approx_kl ~ 0 on first minibatch (no jit)  test_ratio_is_one_on_first_minibatch
  GAE vs hand-computed example with a mid-rollout done      test_gae_hand_computed_with_done_mid_rollout
  clipped-loss gradient zero / non-zero cases               test_clipped_surrogate_{zero,nonzero}_grad_*
  continuous log_probs summed over action dims              test_gaussian_log_prob_summed_over_action_dims
  vmapped seeds produce different results                   test_vmapped_seeds_differ
  CartPole reaches ~500 within 5e5 steps                    test_cartpole_learns (slow)
"""

import dataclasses

import distrax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
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


# ---------------------------------------------------------------------------
# Distributions (Spec Sec. 4, 8) -- checked against distrax as a reference
# ---------------------------------------------------------------------------
def test_categorical_matches_distrax():
    logits = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    actions = jnp.array([0, 1, 2, 1, 0])
    ours, ref = ppo_jax.Categorical(logits), distrax.Categorical(logits=logits)
    np.testing.assert_allclose(ours.log_prob(actions), ref.log_prob(actions), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ours.entropy(), ref.entropy(), rtol=1e-5, atol=1e-6)
    assert ours.log_prob(actions).shape == (5,)


def test_categorical_sample_frequencies():
    logits = jnp.log(jnp.array([0.2, 0.5, 0.3]))
    samples = ppo_jax.Categorical(jnp.broadcast_to(logits, (20_000, 3))).sample(jax.random.PRNGKey(1))
    freqs = jnp.bincount(samples, length=3) / samples.shape[0]
    np.testing.assert_allclose(freqs, [0.2, 0.5, 0.3], atol=0.02)


def test_gaussian_log_prob_summed_over_action_dims():
    """Spec Sec. 8: continuous log_probs must be summed over action dims."""
    key_m, key_a = jax.random.split(jax.random.PRNGKey(2))
    mean = jax.random.normal(key_m, (5, 3))
    log_std = jnp.array([-0.5, 0.0, 0.3])
    action = jax.random.normal(key_a, (5, 3))
    pi = ppo_jax.DiagGaussian(mean, log_std)

    logp = pi.log_prob(action)
    assert logp.shape == (5,)  # summed: one log-prob per sample, not per dim
    per_dim = distrax.Normal(mean, jnp.exp(log_std)).log_prob(action)  # (5, 3)
    np.testing.assert_allclose(logp, per_dim.sum(-1), rtol=1e-5, atol=1e-5)

    ref = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    np.testing.assert_allclose(logp, ref.log_prob(action), rtol=1e-5, atol=1e-5)
    assert pi.entropy().shape == (5,)
    np.testing.assert_allclose(pi.entropy(), ref.entropy(), rtol=1e-5, atol=1e-5)


def test_gaussian_sample_statistics():
    mean = jnp.broadcast_to(jnp.array([1.0, -2.0]), (50_000, 2))
    log_std = jnp.log(jnp.array([0.5, 2.0]))
    samples = ppo_jax.DiagGaussian(mean, log_std).sample(jax.random.PRNGKey(3))
    np.testing.assert_allclose(samples.mean(0), [1.0, -2.0], atol=0.05)
    np.testing.assert_allclose(samples.std(0), [0.5, 2.0], rtol=0.03)


def test_gaussian_gradient_reaches_log_std():
    # d/d(log_std) of log N(a; 0, std) = z^2 - 1 per dim; with a=2, std=1: 3 per sample, 4 samples -> 12.
    action = 2.0 * jnp.ones((4, 2))
    grad = jax.grad(lambda ls: ppo_jax.DiagGaussian(jnp.zeros((4, 2)), ls).log_prob(action).sum())(jnp.zeros(2))
    np.testing.assert_allclose(grad, [12.0, 12.0], rtol=1e-6)


def test_distributions_are_jittable_pytrees():
    pi = jax.jit(lambda m: ppo_jax.DiagGaussian(m, jnp.zeros(2)))(jnp.zeros((3, 2)))
    assert isinstance(pi, ppo_jax.DiagGaussian)
    logp = jax.jit(lambda l, a: ppo_jax.Categorical(l).log_prob(a))(jnp.zeros((3, 4)), jnp.array([0, 1, 3]))
    np.testing.assert_allclose(logp, np.log(0.25) * np.ones(3), rtol=1e-6)


# ---------------------------------------------------------------------------
# Network (Spec Sec. 4)
# ---------------------------------------------------------------------------
def _init(net, obs_dim=4, batch=8):
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch, obs_dim))
    return net.init(jax.random.PRNGKey(1), obs)["params"], obs


def _singular_values(kernel):
    return np.linalg.svd(np.asarray(kernel), compute_uv=False)


@pytest.mark.parametrize("continuous", [False, True])
def test_network_output_shapes(continuous):
    net = ppo_jax.ActorCritic(action_dim=2, continuous=continuous)
    params, obs = _init(net)
    pi, value = net.apply({"params": params}, obs)
    assert value.shape == (8,)
    if continuous:
        assert isinstance(pi, ppo_jax.DiagGaussian) and pi.mean.shape == (8, 2)
    else:
        assert isinstance(pi, ppo_jax.Categorical) and pi.logits.shape == (8, 2)
    assert pi.log_prob(pi.sample(jax.random.PRNGKey(2))).shape == (8,)
    # Unbatched obs also works (used inside vmap): scalar value.
    _, v = net.apply({"params": params}, obs[0])
    assert v.shape == ()


def test_log_std_only_for_continuous():
    params_c, _ = _init(ppo_jax.ActorCritic(action_dim=3, continuous=True))
    params_d, _ = _init(ppo_jax.ActorCritic(action_dim=3, continuous=False))
    np.testing.assert_array_equal(params_c["log_std"], np.zeros(3))
    assert "log_std" not in params_d


def test_orthogonal_init_gains_and_zero_biases():
    """Orthogonal init => every singular value of a kernel equals its gain."""
    params, _ = _init(ppo_jax.ActorCritic(action_dim=2, hidden_size=64), obs_dim=4)
    for trunk in ("actor_trunk", "critic_trunk"):
        for layer in params[trunk].values():
            np.testing.assert_allclose(_singular_values(layer["kernel"]), np.sqrt(2), rtol=1e-4)
            np.testing.assert_array_equal(layer["bias"], 0.0)
    np.testing.assert_allclose(_singular_values(params["actor_head"]["kernel"]), 0.01, rtol=1e-4)
    np.testing.assert_allclose(_singular_values(params["critic_head"]["kernel"]), 1.0, rtol=1e-4)
    np.testing.assert_array_equal(params["actor_head"]["bias"], 0.0)
    np.testing.assert_array_equal(params["critic_head"]["bias"], 0.0)


def test_initial_policy_is_near_uniform():
    net = ppo_jax.ActorCritic(action_dim=4)
    params, obs = _init(net)
    pi, _ = net.apply({"params": params}, obs)
    np.testing.assert_allclose(pi.entropy(), np.log(4), atol=1e-3)


def test_separate_vs_shared_trunk():
    obs = jnp.ones((8, 4))
    sep = ppo_jax.ActorCritic(action_dim=2)
    p_sep, _ = _init(sep)
    assert {"actor_trunk", "critic_trunk"} <= set(p_sep) and "trunk" not in p_sep

    # Separate nets: the value has no gradient w.r.t. actor parameters.
    g = jax.grad(lambda p: sep.apply({"params": p}, obs)[1].sum())(p_sep)
    assert all(jnp.all(x == 0) for x in jax.tree.leaves(g["actor_trunk"]))
    assert any(jnp.any(x != 0) for x in jax.tree.leaves(g["critic_trunk"]))

    shared = ppo_jax.ActorCritic(action_dim=2, shared_trunk=True)
    p_sh, _ = _init(shared)
    assert "trunk" in p_sh and "actor_trunk" not in p_sh
    g = jax.grad(lambda p: shared.apply({"params": p}, obs)[1].sum())(p_sh)
    assert any(jnp.any(x != 0) for x in jax.tree.leaves(g["trunk"]))


def test_make_network_uses_config():
    c = ppo_jax.make_config("halfcheetah")
    net = ppo_jax.make_network(c, action_dim=6)
    params, _ = _init(net, obs_dim=17)
    assert net.continuous and params["actor_trunk"]["Dense_0"]["kernel"].shape == (17, 256)


# ---------------------------------------------------------------------------
# Environment wrappers (Spec Sec. 1)
# ---------------------------------------------------------------------------
class DummyEnv:
    """Deterministic env: state = step counter t; episodes last 5 steps (truncated);
    reward = 1; obs = [t, action] so tests can see what action reached the env."""

    obs_shape, action_dim, continuous = (2,), 1, True
    action_low, action_high = -jnp.ones(1), jnp.ones(1)

    def reset(self, key):
        return jnp.zeros(2), jnp.zeros((), jnp.int32)

    def step(self, key, t, action):
        t = t + 1
        done = t % 5 == 0
        obs = jnp.array([t, action[0]], jnp.float32)
        return obs, t, jnp.float32(1.0), done, {"terminated": jnp.zeros((), bool), "truncated": done}


def _random_rollout(env, num_steps, seed=0):
    """Uniformly random actions through a vectorized env; returns stacked (obs, reward, done, info)."""

    def run(key):
        key, k0 = jax.random.split(key)
        obs, state = env.reset(k0)

        def step(carry, k):
            ka, ks = jax.random.split(k)
            if env.continuous:
                a = jax.random.uniform(ka, (env.num_envs, env.action_dim), minval=-3, maxval=3)
            else:
                a = jax.random.randint(ka, (env.num_envs,), 0, env.action_dim)
            obs, carry, r, d, info = env.step(ks, carry, a)
            return carry, (obs, r, d, info)

        return jax.lax.scan(step, state, jax.random.split(key, num_steps))[1]

    return jax.jit(run)(jax.random.PRNGKey(seed))


def test_cartpole_random_policy():
    env = ppo_jax.make_env(ppo_jax.make_config("CartPole-v1"))
    obs, r, done, info = _random_rollout(env, 1000)
    assert obs.shape == (1000, 4, 4)
    mask = info["returned_episode"]
    mean_return = (info["returned_episode_returns"] * mask).sum() / mask.sum()
    assert 15 < mean_return < 30  # random CartPole ~ 22
    # Random play never survives 500 steps: every episode end is a true termination.
    assert jnp.all(info["terminated"] == done) and not info["truncated"].any()
    # CartPole reward is 1 per step, so return == length.
    np.testing.assert_array_equal(info["returned_episode_returns"], info["returned_episode_lengths"])


def test_cartpole_terminal_obs_is_pre_reset_obs():
    env = ppo_jax.make_env(ppo_jax.make_config("CartPole-v1"))
    obs, _, done, info = _random_rollout(env, 500)
    term_obs = info["terminal_obs"]
    np.testing.assert_array_equal(term_obs[~done], obs[~done])  # no reset -> identical
    # On done steps, obs is a fresh reset (|s| <= 0.05) and terminal_obs is the failed state
    # (cart out of bounds or pole past 12 degrees).
    assert jnp.all(jnp.abs(obs[done]) <= 0.05)
    failed = (jnp.abs(term_obs[done][:, 0]) > 2.4) | (jnp.abs(term_obs[done][:, 2]) > 0.2094)
    assert jnp.all(failed)


def test_cartpole_time_limit_is_truncation():
    params = gymnax.make("CartPole-v1")[1].replace(max_steps_in_episode=10)
    env = ppo_jax.VecEnv(ppo_jax.LogWrapper(ppo_jax.AutoResetWrapper(ppo_jax.GymnaxAdapter("CartPole-v1", params))), 8)
    _, _, done, info = _random_rollout(env, 300)
    lengths = info["returned_episode_lengths"]
    assert info["truncated"].any() and info["terminated"].any()
    assert jnp.all(lengths[info["truncated"]] == 10)
    assert not jnp.any(info["terminated"] & info["truncated"])
    np.testing.assert_array_equal(info["terminated"] | info["truncated"], done)


def test_cartpole_fall_on_last_step_counts_as_terminated():
    """Edge case: pole falls on the same step the time limit is reached."""
    gym_env, params = gymnax.make("CartPole-v1")
    adapter = ppo_jax.GymnaxAdapter("CartPole-v1", params.replace(max_steps_in_episode=10))
    _, state = adapter.reset(jax.random.PRNGKey(0))
    state = state.replace(theta=jnp.float32(0.2), theta_dot=jnp.float32(2.0), time=9)
    _, _, _, done, info = adapter.step(jax.random.PRNGKey(1), state, 1)
    assert done and info["terminated"] and not info["truncated"]


def test_pendulum_episodes_are_all_truncations():
    env = ppo_jax.make_env(ppo_jax.make_config("Pendulum-v1"))
    _, _, done, info = _random_rollout(env, 1000)
    assert done.sum() == 4 * 1000 // 200
    assert not info["terminated"].any() and jnp.all(info["truncated"] == done)
    assert jnp.all(info["returned_episode_lengths"][done] == 200)


def test_clip_action_and_auto_reset_and_log_with_dummy_env():
    env = ppo_jax.VecEnv(ppo_jax.LogWrapper(ppo_jax.AutoResetWrapper(ppo_jax.ClipAction(DummyEnv()))), 2)
    obs, state = env.reset(jax.random.PRNGKey(0))
    infos = []
    for i in range(5):
        obs, state, r, done, info = env.step(jax.random.PRNGKey(i), state, jnp.array([[5.0], [-0.5]]))
        infos.append(info)
    np.testing.assert_array_equal(infos[0]["terminal_obs"][:, 1], [1.0, -0.5])  # clipped to [-1, 1]
    assert done.all() and jnp.all(obs[:, 0] == 0)  # step 5: auto-reset back to t=0
    np.testing.assert_array_equal(info["terminal_obs"][:, 0], [5, 5])  # pre-reset obs kept
    np.testing.assert_array_equal(info["returned_episode_returns"], [5.0, 5.0])
    np.testing.assert_array_equal(info["returned_episode_lengths"], [5, 5])
    assert jnp.all(state.episode_returns == 0) and jnp.all(state.episode_lengths == 0)


def test_running_mean_std_matches_numpy():
    batches = [jax.random.normal(jax.random.PRNGKey(i), (16, 3)) * (i + 1) + i for i in range(5)]
    rms = ppo_jax.RunningMeanStd.create((3,))
    for b in batches:
        rms = rms.update(b)
    all_x = np.concatenate(batches)
    np.testing.assert_allclose(rms.mean, all_x.mean(0), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rms.var, all_x.var(0), rtol=1e-3)


def test_obs_normalization():
    env = ppo_jax.make_env(ppo_jax.make_config("Pendulum-v1", normalize_obs=True, num_envs=16))
    obs, _, done, info = _random_rollout(env, 2000)
    late = obs[1000:].reshape(-1, 3)
    np.testing.assert_allclose(late.mean(0), 0.0, atol=0.15)
    np.testing.assert_allclose(late.std(0), 1.0, atol=0.15)
    assert jnp.all(jnp.abs(obs) <= 10.0)
    # terminal_obs goes through the same normalization (equal to obs wherever no reset happened).
    np.testing.assert_allclose(info["terminal_obs"][~done], obs[~done], atol=1e-6)


def test_reward_normalization_with_dummy_env():
    gamma = 0.9
    env = ppo_jax.NormalizeVecReward(ppo_jax.VecEnv(ppo_jax.AutoResetWrapper(DummyEnv()), 1), gamma)
    _, state = env.reset(jax.random.PRNGKey(0))
    returns, rewards = [], []
    R = 0.0
    for i in range(12):
        _, state, r, done, _ = env.step(jax.random.PRNGKey(i), state, jnp.zeros((1, 1)))
        R = R * gamma + 1.0
        returns.append(R)
        rewards.append(float(r[0]))
        if done[0]:
            R = 0.0
        # Scaled by the running std of the discounted returns seen so far (incl. this one),
        # then clipped to +-10 (step 1: Var = 0, so the scaled reward hits the clip).
        expected = min(1.0 / np.sqrt(np.var(returns) + 1e-8), 10.0)
        np.testing.assert_allclose(r[0], expected, rtol=2e-3)
    assert rewards[0] == 10.0
    np.testing.assert_allclose(state.discounted_return, [R], rtol=1e-6)


def test_make_env_rejects_action_space_mismatch():
    with pytest.raises(AssertionError, match="continuous"):
        ppo_jax.make_env(ppo_jax.make_config("CartPole-v1", continuous=True))


def test_brax_adapter_truncation():
    c = ppo_jax.make_config("halfcheetah", num_envs=2, num_minibatches=2, brax_episode_length=20,
                            normalize_obs=True, normalize_reward=True)
    env = ppo_jax.make_env(c)
    obs, r, done, info = _random_rollout(env, 45)
    assert obs.shape == (45, 2, 17)
    np.testing.assert_array_equal(jnp.where(done[:, 0])[0], [19, 39])
    assert not info["terminated"].any() and jnp.all(info["truncated"] == done)
    assert jnp.all(jnp.isfinite(obs)) and jnp.all(jnp.abs(r) <= 10.0)


# ---------------------------------------------------------------------------
# Rollout collection (Spec Sec. 2)
# ---------------------------------------------------------------------------
def _setup_rollout(env_name, **overrides):
    """Returns a jitted rollout(params, env_state, obs, rng, num_steps) and its initial inputs."""
    config = ppo_jax.make_config(env_name, **overrides)
    env = ppo_jax.make_env(config)
    net = ppo_jax.make_network(config, env.action_dim)
    k_init, k_reset, rng = jax.random.split(jax.random.PRNGKey(0), 3)
    obs, env_state = env.reset(k_reset)
    params = net.init(k_init, obs)
    rollout = jax.jit(
        lambda p, s, o, r, n: ppo_jax.collect_rollout(env, net.apply, p, s, o, r, n),
        static_argnums=4,
    )
    return rollout, net, params, env_state, obs, rng


@pytest.mark.parametrize("env_name,obs_dim,action_shape", [("CartPole-v1", 4, ()), ("Pendulum-v1", 3, (1,))])
def test_rollout_shapes(env_name, obs_dim, action_shape):
    rollout, _, params, env_state, obs, rng = _setup_rollout(env_name)
    _, last_obs, _, traj = rollout(params, env_state, obs, rng, 16)
    T, N = 16, 4
    assert traj.obs.shape == (T, N, obs_dim) and last_obs.shape == (N, obs_dim)
    assert traj.action.shape == (T, N, *action_shape)
    for field in ("log_prob", "value", "reward", "done", "terminated", "bootstrap_value"):
        assert getattr(traj, field).shape == (T, N), field
    assert traj.info["returned_episode"].shape == (T, N)
    assert "terminal_obs" not in traj.info


def test_rollout_log_prob_and_value_match_recomputation():
    rollout, net, params, env_state, obs, rng = _setup_rollout("Pendulum-v1")
    _, _, _, traj = rollout(params, env_state, obs, rng, 32)
    pi, value = net.apply(params, traj.obs)
    np.testing.assert_allclose(pi.log_prob(traj.action), traj.log_prob, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(value, traj.value, rtol=1e-5, atol=1e-5)


def test_rollout_bootstrap_value_is_next_value_when_no_reset():
    rollout, _, params, env_state, obs, rng = _setup_rollout("CartPole-v1")
    _, _, _, traj = rollout(params, env_state, obs, rng, 200)
    not_done = ~traj.done[:-1]
    assert traj.done.any() and not_done.any()
    np.testing.assert_allclose(traj.bootstrap_value[:-1][not_done], traj.value[1:][not_done], rtol=1e-5, atol=1e-6)
    # Where an episode was reset, V(terminal obs) != V(first obs of the new episode).
    reset = traj.done[:-1]
    assert jnp.all(traj.bootstrap_value[:-1][reset] != traj.value[1:][reset])


def test_rollout_carry_threads_state():
    """Two consecutive 32-step rollouts == one 64-step rollout (state, obs and rng carried)."""
    rollout, _, params, env_state, obs, rng = _setup_rollout("CartPole-v1")
    _, _, _, full = rollout(params, env_state, obs, rng, 64)
    s, o, r, first = rollout(params, env_state, obs, rng, 32)
    _, _, _, second = rollout(params, s, o, r, 32)
    joined = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), first, second)
    for a, b in zip(jax.tree.leaves(full), jax.tree.leaves(joined)):
        np.testing.assert_array_equal(a, b)


def test_rollout_depends_on_rng():
    rollout, _, params, env_state, obs, rng = _setup_rollout("Pendulum-v1")
    _, _, _, a = rollout(params, env_state, obs, rng, 8)
    _, _, _, b = rollout(params, env_state, obs, jax.random.PRNGKey(123), 8)
    assert not jnp.allclose(a.action, b.action)


# ---------------------------------------------------------------------------
# GAE (Spec Sec. 3, 8)
# ---------------------------------------------------------------------------
def _gae_traj(reward, value, done, terminated, bootstrap_value):
    """Transition with only the fields GAE reads filled in; inputs are (T, N)."""
    arr = lambda x: jnp.asarray(x, jnp.float32)
    zeros = jnp.zeros_like(arr(reward))
    return ppo_jax.Transition(
        obs=zeros, action=zeros, log_prob=zeros, value=arr(value), reward=arr(reward),
        done=jnp.asarray(done, bool), terminated=jnp.asarray(terminated, bool),
        bootstrap_value=arr(bootstrap_value), info={},
    )


def test_gae_hand_computed_with_done_mid_rollout():
    """gamma = lambda = 0.5 (gamma*lambda = 0.25), r = 1 everywhere, V = [0.5, 1, 1.5, 2, 2.5],
    V(s_T) = 3, episode ends at t = 2. Env 0: terminated (bootstrap 0). Env 1: truncated
    (bootstrap from V(terminal obs) = 10). Bootstrap values elsewhere are garbage (99) and
    must be ignored.

    Backwards, both envs:
      t=4: delta = 1 + 0.5*3   - 2.5 = 0     A = 0
      t=3: delta = 1 + 0.5*2.5 - 2.0 = 0.25  A = 0.25 + 0.25*0 = 0.25
    Env 0 (terminated at t=2, recursion cut):
      t=2: delta = 1 + 0       - 1.5 = -0.5  A = -0.5
      t=1: delta = 1 + 0.5*1.5 - 1.0 = 0.75  A = 0.75 + 0.25*(-0.5) = 0.625
      t=0: delta = 1 + 0.5*1.0 - 0.5 = 1.0   A = 1.0 + 0.25*0.625   = 1.15625
    Env 1 (truncated at t=2):
      t=2: delta = 1 + 0.5*10  - 1.5 = 4.5   A = 4.5
      t=1: A = 0.75 + 0.25*4.5   = 1.875
      t=0: A = 1.0  + 0.25*1.875 = 1.46875
    """
    col = lambda x: np.stack([x, x], axis=1)  # same data for both envs -> (5, 2)
    done = col(np.array([0, 0, 1, 0, 0], bool))
    terminated = np.stack([done[:, 0], np.zeros(5, bool)], axis=1)
    bootstrap = np.full((5, 2), 99.0)
    bootstrap[2] = 10.0
    values = col(np.array([0.5, 1.0, 1.5, 2.0, 2.5]))
    traj = _gae_traj(np.ones((5, 2)), values, done, terminated, bootstrap)

    adv, targets = ppo_jax.calculate_gae(traj, jnp.array([3.0, 3.0]), gamma=0.5, gae_lambda=0.5)
    expected = np.array([[1.15625, 1.46875], [0.625, 1.875], [-0.5, 4.5], [0.25, 0.25], [0.0, 0.0]])
    np.testing.assert_allclose(adv, expected, rtol=1e-6)
    np.testing.assert_allclose(targets, expected + values, rtol=1e-6)


def test_gae_lambda_limits():
    """lambda=0 -> one-step TD error; lambda=1 -> discounted return minus V (no dones)."""
    key_r, key_v = jax.random.split(jax.random.PRNGKey(0))
    r = jax.random.normal(key_r, (6, 3))
    v = jax.random.normal(key_v, (6, 3))
    last_val, gamma = jnp.array([0.3, -1.0, 2.0]), 0.9
    traj = _gae_traj(r, v, np.zeros((6, 3)), np.zeros((6, 3)), np.zeros((6, 3)))

    adv0, _ = ppo_jax.calculate_gae(traj, last_val, gamma, 0.0)
    next_v = jnp.concatenate([v[1:], last_val[None]])
    np.testing.assert_allclose(adv0, r + gamma * next_v - v, rtol=1e-5, atol=1e-6)

    _, targets1 = ppo_jax.calculate_gae(traj, last_val, gamma, 1.0)
    ret, returns = last_val, []
    for t in reversed(range(6)):
        ret = r[t] + gamma * ret
        returns.append(ret)
    np.testing.assert_allclose(targets1, jnp.stack(returns[::-1]), rtol=1e-5, atol=1e-5)


def test_gae_outputs_carry_no_gradient():
    traj = _gae_traj(np.ones((4, 1)), np.ones((4, 1)), np.zeros((4, 1)), np.zeros((4, 1)), np.zeros((4, 1)))

    def f(values):
        adv, targets = ppo_jax.calculate_gae(traj._replace(value=values), jnp.ones(1), 0.99, 0.95)
        return adv.sum() + targets.sum()

    np.testing.assert_array_equal(jax.grad(f)(traj.value), 0.0)


# ---------------------------------------------------------------------------
# PPO loss and update (Spec Sec. 5, 8)
# ---------------------------------------------------------------------------
EPS = 0.2


def _surrogate_grad(r, a):
    """d L^CLIP_loss / d logp for one sample with ratio r and advantage a."""
    old_logp = jnp.float32(-1.3)
    f = lambda logp: ppo_jax.clipped_surrogate(jnp.exp(logp - old_logp), jnp.float32(a), EPS)
    return float(jax.grad(f)(old_logp + jnp.log(jnp.float32(r))))


@pytest.mark.parametrize("r,a", [(1.5, 1.0), (0.5, -1.0)])
def test_clipped_surrogate_zero_grad_when_clipped(r, a):
    """(A>0, r>1+eps) and (A<0, r<1-eps): clipped term selected -> no gradient."""
    assert _surrogate_grad(r, a) == 0.0


@pytest.mark.parametrize("r,a", [(0.5, 1.0), (1.5, -1.0), (1.0, 1.0), (1.1, -1.0), (0.9, 1.0)])
def test_clipped_surrogate_nonzero_grad_otherwise(r, a):
    """(A>0, r<1-eps), (A<0, r>1+eps) and inside the range: unclipped term selected,
    loss = -r*A so d/dlogp = -r*A (since dr/dlogp = r)."""
    np.testing.assert_allclose(_surrogate_grad(r, a), -r * a, rtol=1e-5)


def _rollout_and_gae(env_name, **overrides):
    config = ppo_jax.make_config(env_name, **overrides)
    env = ppo_jax.make_env(config)
    net = ppo_jax.make_network(config, env.action_dim)
    k_init, k_reset, k_roll, k_upd = jax.random.split(jax.random.PRNGKey(0), 4)
    obs, env_state = env.reset(k_reset)
    tx, lr_fn = ppo_jax.make_optimizer(config)
    train_state = ppo_jax.TrainState.create(apply_fn=net.apply, params=net.init(k_init, obs), tx=tx)
    _, last_obs, _, traj = ppo_jax.collect_rollout(env, net.apply, train_state.params, env_state, obs, k_roll, config.num_steps)
    _, last_val = net.apply(train_state.params, last_obs)
    adv, targets = ppo_jax.calculate_gae(traj, last_val, config.gamma, config.gae_lambda)
    return config, train_state, lr_fn, traj, adv, targets, k_upd


@pytest.mark.parametrize("env_name", ["CartPole-v1", "Pendulum-v1"])
def test_ratio_is_one_on_first_minibatch(env_name):
    """Spec Sec. 8: without jit, before any update the new policy == old policy, so on the
    first minibatch of the first epoch ratio == 1 exactly and approx_kl == 0."""
    with jax.disable_jit():
        config, ts, _, traj, adv, targets, key = _rollout_and_gae(env_name, num_steps=16)
        batch = ppo_jax.Batch(traj.obs, traj.action, traj.log_prob, traj.value, adv, targets)
        batch = jax.tree.map(lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch)
        _, perm_key = jax.random.split(key)  # same key ppo_update uses for epoch 0
        first = jax.tree.map(lambda x: x[0], ppo_jax.make_minibatches(batch, perm_key, config.num_minibatches))

        pi, _ = ts.apply_fn(ts.params, first.obs)
        ratio = jnp.exp(pi.log_prob(first.action) - first.log_prob)
        np.testing.assert_array_equal(ratio, 1.0)
        _, aux = ppo_jax.ppo_loss(ts.params, ts.apply_fn, first, config)
        assert aux["approx_kl"] == 0.0 and aux["clip_frac"] == 0.0

        _, _, metrics = ppo_jax.ppo_update(ts, traj, adv, targets, key, config)
        assert metrics["approx_kl"][0, 0] == 0.0  # same minibatch inside the real update


def test_make_minibatches_is_a_partition():
    batch = ppo_jax.Batch(*(jnp.arange(12.0) for _ in range(6)))
    mbs = ppo_jax.make_minibatches(batch, jax.random.PRNGKey(0), 3)
    assert mbs.obs.shape == (3, 4)
    np.testing.assert_array_equal(jnp.sort(mbs.obs.reshape(-1)), jnp.arange(12.0))
    np.testing.assert_array_equal(mbs.obs, mbs.targets)  # all fields shuffled identically


def test_value_clipping_matches_unclipped_at_old_value():
    config, ts, _, traj, adv, targets, _ = _rollout_and_gae("CartPole-v1", num_steps=16)
    batch = ppo_jax.Batch(traj.obs, traj.action, traj.log_prob, traj.value, adv, targets)
    batch = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
    _, a = ppo_jax.ppo_loss(ts.params, ts.apply_fn, batch, config)
    _, b = ppo_jax.ppo_loss(ts.params, ts.apply_fn, batch, dataclasses.replace(config, clip_vloss=True))
    np.testing.assert_allclose(a["v_loss"], b["v_loss"], rtol=1e-6)  # V == V_old -> no clipping


@pytest.mark.parametrize("env_name", ["CartPole-v1", "Pendulum-v1"])
def test_ppo_update_runs_and_changes_params(env_name):
    config, ts, lr_fn, traj, adv, targets, key = _rollout_and_gae(env_name)
    update = jax.jit(lambda ts, traj, adv, tgt, key: ppo_jax.ppo_update(ts, traj, adv, tgt, key, config))
    new_ts, new_key, metrics = update(ts, traj, adv, targets, key)

    n_steps = config.update_epochs * config.num_minibatches
    assert int(new_ts.step) == n_steps
    assert not jnp.array_equal(new_key, key)
    for name in ("total_loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac"):
        assert metrics[name].shape == (config.update_epochs, config.num_minibatches), name
        assert jnp.all(jnp.isfinite(metrics[name])), name
    changed = jax.tree.map(lambda a, b: not jnp.array_equal(a, b), ts.params, new_ts.params)
    assert all(jax.tree.leaves(changed))
    # Policy moved but stayed close to the old one (KL small, value fit improved).
    assert 0.0 < float(metrics["approx_kl"][-1].mean()) < 0.1
    assert metrics["v_loss"][-1].mean() < metrics["v_loss"][0].mean()
    # LR annealing: linear decay from lr towards 0.
    total = config.num_updates * n_steps
    np.testing.assert_allclose(lr_fn(new_ts.step), config.lr * (1 - n_steps / total), rtol=1e-5)


# ---------------------------------------------------------------------------
# make_train: full jitted loop (Spec "Overall structure", Sec. 7, 8)
# ---------------------------------------------------------------------------
TINY = dict(total_timesteps=4 * 128 * 3)  # 3 updates


@pytest.mark.parametrize("env_name,extra", [("CartPole-v1", {}), ("Pendulum-v1", dict(normalize_obs=True, normalize_reward=True))])
def test_make_train_metrics_shapes(env_name, extra):
    config = ppo_jax.make_config(env_name, **TINY, **extra)
    runner_state, metrics = jax.jit(ppo_jax.make_train(config))(jax.random.PRNGKey(0))
    expected = {"episodic_return", "episodic_length", "episodes_completed", "total_loss", "pg_loss",
                "v_loss", "entropy", "approx_kl", "clip_frac", "explained_variance", "learning_rate", "env_steps"}
    assert set(metrics) == expected
    for name, value in metrics.items():
        assert value.shape == (3,), name
    np.testing.assert_array_equal(metrics["env_steps"], [512, 1024, 1536])
    assert int(runner_state.train_state.step) == 3 * config.update_epochs * config.num_minibatches
    assert jnp.all(jnp.diff(metrics["learning_rate"]) < 0)  # annealing


def test_vmapped_seeds_differ():
    """Spec Sec. 8: PRNG keys are split correctly -> different seeds, different results;
    the same seed twice -> identical results."""
    config = ppo_jax.make_config("CartPole-v1", **TINY)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    keys = jnp.stack([keys[0], keys[1], keys[0]])  # seeds 0 and 2 are identical
    runner_state, metrics = jax.jit(jax.vmap(ppo_jax.make_train(config)))(keys)
    assert metrics["pg_loss"].shape == (3, 3)
    kernel = runner_state.train_state.params["params"]["actor_head"]["kernel"]
    assert not jnp.allclose(kernel[0], kernel[1])
    assert not jnp.allclose(metrics["v_loss"][0], metrics["v_loss"][1])
    np.testing.assert_array_equal(kernel[0], kernel[2])
    np.testing.assert_array_equal(metrics["v_loss"][0], metrics["v_loss"][2])


def test_debug_print(capsys):
    config = ppo_jax.make_config("CartPole-v1", **TINY, debug_print=True)
    jax.block_until_ready(jax.jit(ppo_jax.make_train(config))(jax.random.PRNGKey(0)))
    jax.effects_barrier()
    out = capsys.readouterr().out
    assert "update     3/3" in out and "return" in out


# ---------------------------------------------------------------------------
# Diagnostics helpers and end-to-end learning (Spec Sec. 7, 8)
# ---------------------------------------------------------------------------
def test_ffill_nan():
    x = np.array([[np.nan, 1.0, np.nan, np.nan, 3.0, np.nan], [2.0, np.nan, 4.0, np.nan, np.nan, np.nan]])
    out = ppo_jax.ffill_nan(x)
    np.testing.assert_array_equal(out[0], [np.nan, 1, 1, 1, 3, 3])
    np.testing.assert_array_equal(out[1], [2, 2, 4, 4, 4, 4])


def test_diagnostics_table():
    config = ppo_jax.make_config("CartPole-v1", **TINY)
    _, metrics = jax.jit(jax.vmap(ppo_jax.make_train(config)))(jax.random.split(jax.random.PRNGKey(0), 2))
    table = ppo_jax.diagnostics_table(metrics, num_rows=10)
    lines = table.splitlines()
    assert len(lines) == 1 + 3  # header + one row per update (only 3 updates)
    assert "approx_kl" in lines[0] and lines[-1].split()[0] == "1536"


@pytest.mark.slow
def test_cartpole_learns():
    """Spec Sec. 8: CartPole should reach ~500 return within 5e5 steps (3 seeds).
    Judged on the mean return over the last 10 updates of each seed."""
    config = ppo_jax.make_config("CartPole-v1")
    keys = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    _, metrics = jax.jit(jax.vmap(ppo_jax.make_train(config)))(keys)
    final = ppo_jax.ffill_nan(metrics["episodic_return"])[:, -10:].mean(axis=1)
    assert final.mean() >= 475 and final.min() >= 400, (
        f"CartPole did not reach ~500: final return per seed = {np.round(final, 1)}\n"
        f"Diagnostics (mean over seeds):\n{ppo_jax.diagnostics_table(metrics)}"
    )
