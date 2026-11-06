"""w-weighted diffusion behavior cloning — the policy-extraction loss of `--extraction diffusion`.

This is NOT a standalone learner with its own DICE dual. The ν/μ dual and the importance
ratio w = d^π/d^D stay in `AET_learner.train_step` (one copy, SSOT). Only the extraction
loss changes: WBC's −w·log π(a|s,racc,t) becomes w·‖ε − ε_θ(a_k, k | s,racc,t)‖².

Why the target is unchanged: the denoising loss is (up to a k-weighting) a bound on
−log π, so minimizing E_{d^D}[w·denoise] fits the same DICE-optimal policy
π*(a|s) ∝ π_D(a|s)·w(s,a) that plain WBC (w¹) projects onto. Weighting the loss directly
(instead of resampling the batch ∝ w) keeps the estimator identical to WBC's, so
`--wbc_weight` shaping (raw / norm / binary / ...) applies unchanged.

Per-timestep only: each transition is scored at its own t_b, as in `timecond`.
"""
import jax
import jax.numpy as jnp

from src.nets.diffusion_policy import DiffusionPolicy


def diffusion_wbc_loss(policy: DiffusionPolicy, states, raccs, timesteps, actions, w_b, key):
    """Returns (loss, per-sample denoising MSE [B])."""
    B, A = actions.shape
    assert states.shape[0] == B and raccs.shape[0] == B, (states.shape, raccs.shape)
    assert timesteps.shape == (B,) and w_b.shape == (B,), (timesteps.shape, w_b.shape)

    k_key, eps_key, drop_key = jax.random.split(key, 3)
    k_idx = jax.random.randint(k_key, (B,), 0, policy.n_steps)
    eps = jax.random.normal(eps_key, (B, A))
    ab = policy.alpha_bars.value[k_idx][:, None]                           # [B, 1]
    noisy = jnp.sqrt(ab) * actions + jnp.sqrt(1.0 - ab) * eps

    # CFG training: drop racc to the null token with prob uncond_prob. At 0 this is a
    # deterministic all-ones mask, i.e. plain conditional training (Exp 1).
    racc_present = 1.0 - jax.random.bernoulli(drop_key, policy.uncond_prob, (B,)).astype(
        jnp.float32)

    eps_hat = policy.predict_eps(noisy, k_idx, states, raccs, timesteps, racc_present)
    mse = jnp.mean((eps_hat - eps) ** 2, axis=-1)                          # [B]
    assert mse.shape == (B,), mse.shape
    return jnp.mean(w_b * mse), mse
