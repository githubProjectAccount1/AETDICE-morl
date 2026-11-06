"""KL divergence — f(w) = w·log(w), f'⁻¹(y) = exp(y − 1)."""
import jax.numpy as jnp


def f(x, eps: float = 1e-10):
    return x * jnp.log(x + eps)


def f_derivative_inverse(y, t: float = 1.0, w_max: float = float("inf")):
    return jnp.exp(y - 1.0)


def log_f_derivative_inverse(y):
    """log w = y − 1 (f'⁻¹(y) = exp(y − 1))."""
    return y - 1.0
