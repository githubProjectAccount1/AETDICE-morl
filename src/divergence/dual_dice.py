"""DualDICE divergence — f(w) = (2/3) · |w|^(3/2). No closed-form f'⁻¹ in this codebase."""
import jax.numpy as jnp


def f(x, eps: float = 1e-10):
    return 2 / 3 * jnp.abs(x) ** (3 / 2)


def f_derivative_inverse(y, t: float = 1.0, w_max: float = float("inf")):
    raise ValueError("f_derivative_inverse is not implemented for DualDICE.")
