# PPO implementation spec (pure JAX)

Implement PPO-Clip (Schulman et al. 2017) from scratch in pure JAX, with the entire training loop (rollouts + updates) compiled under jax.jit, in the style of PureJaxRL. Single readable file. Prioritize correctness and clarity. Comment each section with which part of the paper it implements.

## Stack
- JAX, Flax (linen), Optax, Distrax (or hand-written Categorical / diagonal Gaussian log_prob + entropy)
- Environments: Gymnax (CartPole-v1 discrete, Pendulum-v1 continuous), then Brax (e.g. halfcheetah or ant) for MuJoCo-style tasks
- Support discrete and continuous action spaces via a config flag

## Overall structure
- make_train(config) returns a pure function train(rng) -> (final_runner_state, metrics)
- train is jax.jit-compiled; run multiple seeds with jax.vmap(train)(jax.random.split(rng, num_seeds))
- Runner state is an explicit pytree: (train_state, env_state, last_obs, normalization stats, rng)
- Outer loop over updates via jax.lax.scan; no Python loops inside the jitted function
- No hidden mutable state anywhere; thread PRNG keys explicitly and split them for init, action sampling, env steps, and minibatch shuffling

## 1. Environment wrappers
- Vectorize with jax.vmap over env reset/step (num_envs parallel envs)
- Auto-reset on episode end inside step
- LogWrapper that tracks episodic return/length in env state and exposes them in info
- Terminated vs truncated: only zero the bootstrap on true termination. Gymnax merges these into `done`, so document how truncation is handled (or bootstrap from the pre-reset obs on time-limit ends). For Brax, use info["truncation"].
- Observation normalization (running mean/var, clip ±10) and reward scaling by running std of discounted returns (clip ±10), implemented as wrappers whose stats live in env state
- Flatten observations if needed; clip continuous actions to env bounds before stepping

## 2. Rollout collection
- jax.lax.scan over num_steps; each step samples actions for all envs and steps them
- Store per step in a Transition pytree: obs, action, log_prob (under the old policy), reward, value, done, info

## 3. Advantage estimation (GAE)
- Implement with jax.lax.scan in reverse over the rollout, bootstrapping from V(last_obs)
- targets = advantages + values
- Use jax.lax.stop_gradient where appropriate

## 4. Network
- Separate actor and critic MLPs by default (2×64, tanh; larger, e.g. 256, for Brax); optional shared-trunk flag
- Orthogonal init: hidden gain √2, policy output 0.01, value output 1.0; zero biases
- Continuous: state-independent learnable log_std

## 5. PPO update
- jax.lax.scan over K epochs; within each epoch, permute the batch with a PRNG key, reshape into minibatches, and jax.lax.scan over minibatches
- Normalize advantages per minibatch
- ratio = exp(new_logp − old_logp); old_logp and advantages must not carry gradients
- Policy loss: −mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))
- Value loss: 0.5·MSE(V, targets); optional value clipping behind a flag
- Total loss = policy_loss + c1·value_loss − c2·entropy, with jax.value_and_grad(has_aux=True) returning the loss components
- Optimizer: optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5)), with an optional optax.linear_schedule for annealing
- Use flax.training.train_state.TrainState

## 6. Hyperparameters (config dict / dataclass)
- Gymnax defaults: lr 2.5e-4, num_envs 4, num_steps 128, total_timesteps 5e5, K=4 epochs, 4 minibatches, γ 0.99, λ 0.95, ε 0.2, c1 0.5, c2 0.01, anneal_lr True
- Brax defaults: lr 3e-4, num_envs 2048, num_steps 10, total_timesteps 5e7, K=4, 32 minibatches, γ 0.99, λ 0.95, ε 0.2, c1 0.5, c2 0.0, obs + reward normalization on
- Derived: num_updates = total_timesteps // (num_steps · num_envs), minibatch_size = num_envs · num_steps // num_minibatches; assert they divide evenly

## 7. Logging & diagnostics
- Return metrics from the scanned loop as stacked arrays (no logging side effects inside jit by default); optionally a jax.debug.callback flag for live printing
- Per update: episodic return/length, policy loss, value loss, entropy, approx_kl ((ratio−1) − log ratio estimator), clip fraction (share with |ratio−1| > ε), explained variance, learning rate
- After training: plot mean ± std return vs env steps across seeds (matplotlib), and report wall-clock time and steps/sec

## 8. Tests / sanity checks
- Without the jit: on the first minibatch of the first epoch, ratio must be exactly 1 and approx_kl ≈ 0
- Unit test GAE against a hand-computed example, including a done mid-rollout
- Unit test the clipped loss with jax.grad w.r.t. the new log_prob: zero gradient for (A>0, r>1+ε) and (A<0, r<1−ε); non-zero for (A>0, r<1−ε), (A<0, r>1+ε) and inside the range
- Check continuous log_probs are summed over action dims
- Check that vmapped seeds produce different results (PRNG keys correctly split)
- CartPole should reach ~500 return within 5e5 steps; if not, report diagnostics

## 9. Deliverables
- ppo_jax.py, a README explaining how to run each env and change configs, and return curves over ≥3 seeds per environment
- Use PureJaxRL's ppo.py and ppo_continuous_action.py as reference implementations, but write the code independently with the structure above