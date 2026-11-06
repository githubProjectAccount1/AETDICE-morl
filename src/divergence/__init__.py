"""f-divergence package — per-divergence math in its own module, dispatched here.

Public API is unchanged from the old single-file `divergence.py`:
    from divergence import FDivergence, f, f_derivative_inverse, state_action_ratio

To add a new divergence:
  1. Create `divergence/<name>.py` with `f(x, eps)` and `f_derivative_inverse(y, t, w_max)`.
  2. Add the enum entry and dict entries below.
No if-tree edits anywhere else.
"""
from enum import Enum

import jax
import jax.numpy as jnp
from typing_extensions import assert_never

from src.divergence import soft_clip_chi
from src.divergence import chi, clip_chi, dual_dice, kl, soft_chi


class FDivergence(str, Enum):
    KL = "KL"
    CHI = "Chi"
    SOFT_CHI = "SoftChi"
    DUAL_DICE = "DualDICE"
    CLIP_CHI = "ClipChi"
    SOFT_CLIP_CHI = "SoftClipChi"


# Dispatch tables. Adding a new divergence = one new file + one row in each table.
# Dispatch happens at trace time (f_divergence is a Python-level static arg
# captured in the train_step closure), so there's no per-step lookup overhead.
_F_FUNCS = {
    FDivergence.KL:            kl.f,
    FDivergence.CHI:           chi.f,
    FDivergence.CLIP_CHI:      clip_chi.f,
    FDivergence.SOFT_CHI:      soft_chi.f,
    FDivergence.SOFT_CLIP_CHI: soft_clip_chi.f,
    FDivergence.DUAL_DICE:     dual_dice.f,
}
_F_INV_FUNCS = {
    FDivergence.KL:            kl.f_derivative_inverse,
    FDivergence.CHI:           chi.f_derivative_inverse,
    FDivergence.CLIP_CHI:      clip_chi.f_derivative_inverse,
    FDivergence.SOFT_CHI:      soft_chi.f_derivative_inverse,
    FDivergence.SOFT_CLIP_CHI: soft_clip_chi.f_derivative_inverse,
    FDivergence.DUAL_DICE:     dual_dice.f_derivative_inverse,
}

# log w 는 w > 0 이 모든 y 에서 성립하는 f 에서만 정의된다 (I-projection, --extraction iproj).
# CHI 계열은 w = relu(·) 라 데이터 밖 행동에서 log w = −∞ 가 된다 — OptiDICE 도 ELU(soft-χ²) 를 쓴다.
_LOG_F_INV_FUNCS = {
    FDivergence.KL:       kl.log_f_derivative_inverse,
    FDivergence.SOFT_CHI: soft_chi.log_f_derivative_inverse,
}
LOG_W_DIVERGENCES = tuple(_LOG_F_INV_FUNCS)


def f(x, f_divergence: FDivergence, eps: float = 1e-10):
    x = jnp.array(x)
    fn = _F_FUNCS.get(f_divergence)
    if fn is None:
        assert_never(f_divergence)
    return fn(x, eps=eps)


def f_derivative_inverse(y, f_divergence: FDivergence, t: float = 1.0, w_max: float = float("inf")):
    """Inverse of f' — gives the optimal importance ratio w for a given dual y = e/β.

    `w_max` clips the ratio (used when f_divergence == CLIP_CHI / SOFT_CLIP_CHI).
    Default ∞ → no clipping. Other f-divergences ignore w_max."""
    y = jnp.array(y)
    fn = _F_INV_FUNCS.get(f_divergence)
    if fn is None:
        assert_never(f_divergence)
    return fn(y, t=t, w_max=w_max)


def state_action_ratio(nu, next_nu, rewards, beta: float, discount: float, f_divergence: FDivergence):
    """Importance ratio w = (f')⁻¹(e/β) with e = r + γ·ν' − ν, ReLU-floored at 0."""
    nu = jnp.array(nu)
    next_nu = jnp.array(next_nu)
    rewards = jnp.array(rewards)
    e = rewards + discount * next_nu - nu
    return jax.nn.relu(f_derivative_inverse(e / beta, f_divergence))


def log_f_derivative_inverse(y, f_divergence: FDivergence):
    """log (f')⁻¹(y) — 닫힌 형태가 있는 f (LOG_W_DIVERGENCES) 만."""
    fn = _LOG_F_INV_FUNCS.get(f_divergence)
    if fn is None:
        raise ValueError(f"log w 는 {[d.name for d in LOG_W_DIVERGENCES]} 에서만 정의된다 (받은 값: {f_divergence})")
    return fn(jnp.asarray(y))
