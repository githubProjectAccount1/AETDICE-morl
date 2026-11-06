"""SOFT_CHI lower tail + CLIP_CHI upper tail.

f(w) = w·log(w) − w + 1   for w < 1
     = (w − 1)² / 2        for 1 ≤ w ≤ w_max
f'⁻¹(y) = exp(y)           for y < 0
        = clip(y + 1, 0, w_max)   for y ≥ 0

Bounds BOTH tails of w — smoothed near 0 (no dead zone) and capped above (no outlier
domination). Most aggressive of the χ²-family variants for small-β stability."""
import jax.numpy as jnp


def f(x, eps: float = 1e-10):
    # Primal matches SOFT_CHI for w ∈ [0, w_max]; +∞ above but f'⁻¹ never goes there.
    return jnp.where(x < 1.0, x * jnp.log(x + eps) - x + 1, (x - 1) ** 2 / 2)


def f_derivative_inverse(y, t: float = 1.0, w_max: float = float("inf")):
    return jnp.where(
        y < 0.0,
        jnp.exp(jnp.where(y < 0.0, y, 0.0)),
        jnp.clip(y + 1.0, 0.0, w_max),
    )
