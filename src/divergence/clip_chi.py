"""χ² with hard upper-bound on importance ratio w ≤ w_max.

f(w) = (w − 1)²/2  for w ∈ [0, w_max]; formally +∞ above (but f'⁻¹ never returns w > w_max).
f'⁻¹(y) = clip(y + 1, 0, w_max). The 0-floor is redundant with the caller's jax.nn.relu
but keeps the function contract self-contained.

Mitigates DICE's small-β failure mode where w ≈ 1 + e/β explodes on rare high-advantage
transitions, dominates the policy gradient, and the β·f(w) regularizer loses meaning."""
import jax.numpy as jnp


def f(x, eps: float = 1e-10):
    # Same primal as plain χ². Clipping enforced upstream in f_derivative_inverse.
    return (x - 1) ** 2 / 2


def f_derivative_inverse(y, t: float = 1.0, w_max: float = float("inf")):
    return jnp.clip(y + 1.0, 0.0, w_max)
