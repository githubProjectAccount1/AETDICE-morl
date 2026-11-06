"""Plain χ² divergence — f(w) = (w − 1)²/2, f'⁻¹(y) = y + 1."""
import jax.numpy as jnp


def f(x, eps: float = 1e-10):
    return (x - 1) ** 2 / 2


def f_derivative_inverse(y, t: float = 1.0, w_max: float = float("inf")):
    return y + 1.0
