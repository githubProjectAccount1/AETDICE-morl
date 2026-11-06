"""SOFT_CHI: KL-shaped lower tail + χ²-shaped upper tail.

f(w) = w·log(w) − w + 1   for w ∈ [0, 1)   (KL piece)
     = (w − 1)² / 2        for w ≥ 1        (χ² piece)
f'⁻¹(y) = exp(y)           for y < 0        (smooth decay to 0+, no hard floor)
        = y + 1            for y ≥ 0        (same as plain χ²)

Removes the χ² lower-tail "dead zone" (w=0 floor) so very-negative residuals contribute
small-but-nonzero importance weights instead of being dropped from the gradient."""
import jax.numpy as jnp


def f(x, eps: float = 1e-10):
    return jnp.where(x < 1.0, x * jnp.log(x + eps) - x + 1, (x - 1) ** 2 / 2)


def f_derivative_inverse(y, t: float = 1.0, w_max: float = float("inf")):
    # Inner where shields jnp.exp from large positive y (we'd compute exp(0) and the
    # outer where masks it). Avoids unnecessary overflow in the unused branch.
    return jnp.where(y < 0.0, jnp.exp(jnp.where(y < 0.0, y, 0.0)), y + 1)


def log_f_derivative_inverse(y):
    """log w = log f'⁻¹(y) in closed form — I-projection 은 데이터 밖 행동의 log w 를 쓴다 (exp 후 log 는 y≪0 에서 −inf)."""
    return jnp.where(y < 0.0, y, jnp.log1p(jnp.maximum(y, 0.0)))
