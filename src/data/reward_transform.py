"""
Warning: This file is deprecated and will be removed or modified in the future. It is not intended for use at the moment.



Environment-dependent reward transformations for D4MORL environments.

Rationale (see D4MORL, Zhu et al. 2023, Appendix A):

MO-Ant:       objectives are x/y axis distance. Apply exp(r) to amplify
              differences in the tails of the distance distribution.

MO-Walker2d:  objectives are speed (r_v = dx/dt + r^s) and energy saving
              (r_e = 4.0 - r^a + r^s). Transform r_e -> 5 - r_e to recover
              the underlying energy cost, making both objectives "higher is
              better" in terms of raw physical quantity.

MO-HalfCheetah: same energy-saving structure as Walker2d
              (r_e = 4.0 - r^a + r^s). Apply 5 - r_e.

MO-Swimmer:   objectives are speed and energy saving
              (r_e = 0.3 - 0.15 * r^a). Apply 0.5 - r_e to recover cost.

MO-Hopper:    objectives are running and jumping. No transform needed;
              both are already interpretable physical quantities.
"""
import os
import numpy as np
import jax.numpy as jnp

from src.utils import min_max_normalization
from src.utility import piecewise_log, EPS   # EPS 단일 소스 (utility.py)


# ── MO-Ant 2-obj → (reward, cost) reshape — Cobb-Douglas 용 ──────────────────
# ★ legacy_src/Transformed_AETDICE/reward_transform.py 의 `antcd` 를 그대로 되살린 것.
#   리팩터(src/ 분리) 때 등록이 빠져 CLI 선택지에서 사라져 있었다 — 주석과 viz 기본 경로
#   (`results_Transformed/.../rsf1.0to1.0_musf1e4_antcd/...`)에만 흔적이 남아 있었다
#   (2026-09-11 발견·복원).
#
# ★ 이게 CD 의 핵심이다: **CD 의 두 채널에 env 목적을 그대로 넣지 않는다.** MO-Ant 의 두
#   목적은 둘 다 속도(vx, vy)라 (reward, cost) 레이아웃이 아니다. 그래서 분해해 다시 조립한다:
#
#     ★ 채널 이름은 `reward`(= CD 분자) / `cost`(= CD 분모) 다. **vx / vy 가 아니다.**
#       (인덱스로 r0/r1 이라 부르면 MO-Ant 원목적 vx/vy 로 오독된다 — 금지, 2026-09-12 확정)
#
#     reward = vx_avg + vy_avg = (vx_reward + vy_reward) + ‖a‖² − 2
#              → **순수 전방 속도** (survive·ctrl_cost 를 걷어낸 값). vx·vy 가 여기서 합쳐진다.
#     cost   = ‖a‖² = Σ_i clip(a,−1,1)_i²
#              → **행동 에너지 비용** (0.5 배수 없음 — 그건 ctrl_cost 다)
#
#   복원 근거 (environment/ant.py):
#     ctrl_cost    = 0.5·‖clip(a,−1,1)‖²,   survive = 1.0
#     other_reward = −ctrl_cost + survive = 1 − 0.5·‖a‖²
#     vx_reward    = vx_avg + other_reward,  vy_reward = vy_avg + other_reward
#     ⇒ vx_avg + vy_avg = (vx_reward + vy_reward) − 2·other_reward
#                        = (vx_reward + vy_reward) + ‖a‖² − 2
#
#   ⇒ MO-Ant 의 CD 는 **vx vs vy 트레이드오프가 아니라 속도 vs 에너지**다.
#     두 목적이 `reward` 하나로 합쳐지므로 ρ sweep 은 "속도를 얼마나 에너지로 살 것인가"를 쓴다.
ANTCD_REWARD_DIM = 2
# CD 레이아웃의 채널 인덱스 (SSOT). 숫자 0/1 을 코드에 흩뿌리지 않기 위한 이름.
# CD(=(reward,cost) 레이아웃)를 만드는 reshape 들 — **SSOT**. 예전엔 cli.py 와 evaluate.py 가
# 각자 목록을 들고 있었고 evaluate 쪽에만 "antcd" 가 빠져 있어, MO-Ant CD 런은 CD 점수가
# **아예 계산되지 않았다** (2026-09-14 smoke 에서 발견). 목록을 한 곳에 두어 재발을 막는다.
# safe-CD 가족 (S0 논문 자세 / S1 속도 제한 / S2a 경계 여유 / S3 토크 포화) — WALKER2D_CD_SAFE_KINDS 와 같은 순서·이름
CD_SAFETY_MODES = ("plog", "linear", "product", "inlog", "violog")   # CobbDouglasUtility.safety — 결합 방식 (2026-11-03; inlog 2026-11-06 D6.28; violog D6.30)
CD_SAFETY_THRESHOLDED = ("linear", "inlog")       # 채널마다 문턱 (ΣS_k − th_k) 을 쓰는 결합
CD_SAFETY_PER_CHANNEL = ("linear", "inlog", "violog")   # F 출력 = [CD, safety 채널마다 1 개] (μ 길이 = reward_dim − 1)
WALKER2D_CD_SAFE_RESHAPES = ("walker2d_cd_safe", "walker2d_cd_safe_vel", "walker2d_cd_safe_margin", "walker2d_cd_safe_sat",
                             "walker2d_cd_safe_pv", "walker2d_cd_safe_pok", "walker2d_cd_safe_pokv",
                             "walker2d_cd_safe_ha", "walker2d_cd_safe_h", "walker2d_cd_safe_hm")
CD_RESHAPES = ("antcd", "walker2dcd", "half_cheetahcd") + WALKER2D_CD_SAFE_RESHAPES
# ★ CD 분모의 단위 오프셋 (사용자 확정 2026-09-16):  cost_t := 1 + ‖a_t‖²
#   per-step 에 더한다 — CD 단에 별도 오프셋 파라미터를 두지 않기 위해서다. 누적하면
#   C = T + Σ‖a‖² 가 되어 (i) 분모가 0 이 될 수 없고 (ii) CD 를 비동차로 만드는 항이
#   궤적 스케일과 같은 차수라 내부 최적(“느리지만 efficient 한 보행”)이 생긴다.
#   대안이던 "누적에 한 번만 +1"(C = 1 + Σ‖a‖²)은 폭발하진 않지만 오프셋이 Σ‖a‖²≈490
#   대비 0.2% 라 CD 가 사실상 (2ρ−1)-동차가 되고, 오라클 속도대 2.3~12.8 m/s 를 덮는 ρ
#   구간이 [0.4986, 0.4998] (폭 0.0012) 로 붕괴한다 — ρ 가 손잡이 역할을 못 한다.
#   per-step 판은 같은 구간이 [0.203, 0.408] (폭 0.205) 로 170 배 넓다. 실측 2026-09-16.
CD_COST_UNIT = 1.0
CD_REWARD_CH = 0   # 분자 — 많을수록 좋다 (antcd: 전방속도)
CD_COST_CH   = 1   # 분모 — 적을수록 좋다 (antcd: 행동 에너지 ‖a‖²)


def reshape_antcd_trajectory(actions: np.ndarray, raw_rewards: np.ndarray,
                             observations=None) -> np.ndarray:
    """[T, 8], [T, 2](vx_reward, vy_reward) → [T, 2] (reward=vx+vy, cost=1+‖a‖²)."""
    energy = np.square(np.clip(actions, -1.0, 1.0)).sum(axis=-1)            # [T] ‖a‖² ≥ 0
    reward = (raw_rewards[..., 0] + raw_rewards[..., 1] + energy - 2.0).astype(np.float32)
    cost   = (CD_COST_UNIT + energy).astype(np.float32)
    return np.stack([reward, cost], axis=-1)


def reshape_antcd_step(action: np.ndarray, info_obj: np.ndarray,
                       observation=None) -> np.ndarray:
    """per-step 판 — eval 쪽에서 env.step() 뒤에 쓴다. 같은 대수식, 배치만 없다."""
    energy = float(np.square(np.clip(action, -1.0, 1.0)).sum())
    reward = float(info_obj[0]) + float(info_obj[1]) + energy - 2.0
    return np.array([reward, CD_COST_UNIT + energy], dtype=np.float32)


# ── MO-Walker2d / MO-HalfCheetah 2-obj → (reward, cost) reshape ───────────────
# Both Walker2d and HalfCheetah encode obj[1] as a "saving" reward:
#       reward_energy = 4.0 - ‖a‖² + alive_bonus  =  5 − ‖a‖²    (alive_bonus = 1)
# so the `cost` channel (action energy) is recovered by  cost := 5 − obj[1].  obj[0]
# (speed + alive) is already a clean positive reward, so  reward := obj[0]. This is the
# layout CobbDouglasUtility expects: dim 0 monotone-increasing reward, dim 1
# monotone-increasing cost.
#
# Walker2d: obj[0] = vx/dt + 1                          (speed + alive)
#           obj[1] = 4 − ‖a‖² + 1                       (energy + alive)
# HalfCheetah: obj[0] = clip(vx/dt, ≤4) + 1             (capped speed + alive)
#              obj[1] = 4 − ‖a‖² + 1                    (energy + alive)
#
# Algebra is identical for both; we keep them as named entries so the env
# validator in main.py can match each reshape to its env.
WALKER_CD_REWARD_DIM      = 2
HALFCHEETAH_CD_REWARD_DIM = 2
# ── 조건부 racc 인코딩 (ν/μ/π 가 받는 racc 입력의 형태) ─────────────────────────
# 왜 (2026-09-18 실측): 기본 'ftilde' 는 F̃=exp(R/50)−1 을 저장하고 RaccEncoder 가 그 위에서
# **전역 z-score** 를 한다. F̃ 는 에피소드 후반에 수백~수천까지 커지므로 전역 통계
# (mean [81.0, 99.2], std [166.9, 193.8]) 를 후반이 지배하고, **초반 t 의 모드 차이가
# z-score 후 5e-4~9e-3 으로 눌린다**. 실측: 같은 상태에서 racc 를 mode0↔mode1 로 바꿔도
# 정책 행동 변화가 t≤250 에서 0.0000 (데이터의 모드 행동차는 0.32~1.55). 즉 커밋이
# 필요한 구간에서 racc 는 정보적으로 존재하지 않는다 — 정책이 무시하는 게 아니라
# 입력에 신호가 없다.
# 'rate' 는 R_t/(t+1) = **지금까지의 스텝당 평균 보상**을 조건으로 준다. 지수도 없고
# 시간에 따라 스케일이 커지지도 않아 t 전 구간에서 O(1) 이고 모드 차이가 보인다.
# ★ 이 인코딩은 **조건부 입력에만** 적용된다. per-step 스칼라 보상(텔레스코핑 F̃ 차)은
#   목적함수 자체이므로 그대로 둔다 (buffer 가 두 배열을 따로 들고 있다).
RACC_ENCODINGS = ("ftilde", "rate")


def racc_rate(raw_racc, timesteps, xp=np):
    """R_t / max(t, 1). raw_racc [N, D] (F̃ 이전), timesteps [N] (궤적 내 0-based t)."""
    t = xp.asarray(timesteps).reshape(-1, 1).astype(raw_racc.dtype)
    return raw_racc / xp.maximum(t, 1.0)


def cd_reward_bounds(reward_dim: int, horizon: int):
    """CD 경로의 per-step 정규화 상수 — **shift 없는 scale-only** (2026-09-16 확정).

        reward_min := 0,   reward_max := horizon(T)
        ⇒ racc_d = Σ_t raw_d / T  = per-step 평균

    왜 데이터셋 min/max 를 쓰지 않는가: min-max 는 shift(−rmin)를 포함하는데, CD 에서
    shift 는 **정책 순위를 바꾼다**. antcd 실측 rmin[reward] = −2.667 < 0 이라 정지 정책이
    500 스텝 동안 +99.35 를 공짜로 받아 ρ=0.5 에서 정지 21.7 vs 고속 1.36 으로 **정지가
    1등**이 됐다. 반대로 scale(나눗셈)은 CD 순위를 전혀 바꾸지 않으므로(cd_score docstring)
    네트워크 입력 스케일만 잡아주는 T 로 나눈다. 실측 검증: raw 대비 v* 상대차 0.0,
    Spearman 1.000000000000 (50,000 궤적, 2026-09-16 A/B).

    Returns: (reward_min [reward_dim], reward_max [reward_dim]) — float32.
    """
    return (np.zeros(reward_dim, dtype=np.float32),
            np.full(reward_dim, float(horizon), dtype=np.float32))


def cd_numerator_power(base, rho: float, *, xp=jnp):
    """CD 분자 `Reward**ρ` — base==0 에서 **정확히 0**이 되게 계산한다.

    `0**ρ` 를 직접 쓰지 않는 이유는 base 가 음수로 새는 경로에서 NaN 이 나기 때문이고,
    마스크 안쪽 인자를 1 로 치환해 그걸 막는다.
    ★ ρ==0 은 예외다: CD = Reward^0 / Cost = 순수 비용최소화라 분자는 **상수 1** 이어야
      한다. 마스킹을 그대로 적용하면 정지(base=0)에서 0 이 되어 ρ=0 의 의미가 뒤집힌다.
    """
    if rho == 0.0:
        return xp.ones_like(base)                    # 0**0 := 1 (ρ=0 은 분자가 상수)
    pos = base > 0.0
    return xp.where(pos, xp.power(xp.where(pos, base, 1.0), rho), 0.0)


_WALKER_HC_ENERGY_CONST   = 5.0   # alive_bonus + 4 from env reward_energy formula
# walker2d and halfcheetah happen to have the same energy const

def _reshape_cost_recovery_trajectory(actions: np.ndarray, raw_rewards: np.ndarray,
                                       const: float) -> np.ndarray:
    """Per-trajectory cost recovery: reward unchanged; cost = 1 + (const − obj[1]).
    Shared by Walker2d (const=5) and HalfCheetah (const=5). `const − obj[1]` 이
    ‖a‖² 이므로 antcd 와 같은 `cost := 1 + ‖a‖²` 규약을 그대로 쓴다 (CD_COST_UNIT)."""
    forward = raw_rewards[..., 0].astype(np.float32)
    cost    = (CD_COST_UNIT + const - raw_rewards[..., 1]).astype(np.float32)
    return np.stack([forward, cost], axis=-1)


def _reshape_cost_recovery_step(action: np.ndarray, info_obj: np.ndarray,
                                 const: float) -> np.ndarray:
    """Per-step cost recovery — eval-side after each env.step()."""
    return np.array([float(info_obj[0]), CD_COST_UNIT + const - float(info_obj[1])],
                    dtype=np.float32)


def reshape_walker2dcd_trajectory(actions, raw_rewards, observations=None):
    return _reshape_cost_recovery_trajectory(actions, raw_rewards, _WALKER_HC_ENERGY_CONST)


def reshape_walker2dcd_step(action, info_obj, observation=None):
    return _reshape_cost_recovery_step(action, info_obj, _WALKER_HC_ENERGY_CONST)


def reshape_half_cheetahcd_trajectory(actions, raw_rewards, observations=None):
    return _reshape_cost_recovery_trajectory(actions, raw_rewards, _WALKER_HC_ENERGY_CONST)


def reshape_half_cheetahcd_step(action, info_obj, observation=None):
    return _reshape_cost_recovery_step(action, info_obj, _WALKER_HC_ENERGY_CONST)


# ── walker2d 3-D (speed, cost, safety) reshape — for CD + safety satisficing ──
# Walker2d has a strict `done` (height/angle bounds) that the CD policy can game
# at low ρ ("drop and die" → tiny R₁ → huge CD score). The safety channel adds
# a quadratic-bowl reward that's MAX at the center of the safe zone and ZERO at
# the env bounds — pulling the policy toward "balanced upright". Speed and
# safety stay on SEPARATE channels so CobbDouglasUtility can keep them cleanly
# decoupled (CD over speed/cost, threshold log over safety alone):
#
#   reward = speed   = vx/dt + alive_bonus        (unchanged from env obj[0])
#   cost   = ‖a‖²    = 5 − obj[1]                 (cost recovery, same as walker2dcd)
#   r2 = r_safety  = (−2·h_norm² + 1) + (−2·a_norm² + 1)   ← pure balance reward
#
# Per-step r_safety ∈ [−2, +2] in-bounds: +2 at perfect (h=1.4, ang=0), 0 when
# one dim is centered and the other at its env bound, −2 at both env bounds.
# Each dim contributes +1 at center, 0 at |x_norm|=1/√2, −1 at the env bound.
# Beyond bounds (briefly, before `done`) r_safety can dip below −2.
#
# Used with CobbDouglasUtility(threshold=T) on this 3-D layout:
#   F̃[..., 0] = amp · Reward^rho / Cost^(1-rho)           — pure CD over (speed, cost)
#   F̃[..., 1] = piecewise_log(R2 − T)               — pure safety floor
# F̃ stays 2-D (matches what G expects); the third raw-reward dim is internal to
# the buffer/F-utility pipeline and never reaches G or pref weights.
WALKER2D_CD_SAFE_REWARD_DIM = 3  # 3-D layout: (speed, cost, safety) — fully decoupled
# Speed feeds CD's numerator, cost feeds CD's denominator, safety feeds the
# threshold branch ONLY. F-utility output stays 2-D (CD efficiency, threshold log)
# so G/pref logic is unchanged — the third raw-reward dim is internal to F.
WALKER2D_HEIGHT_BOUNDS = (0.8, 2.0)   # env: done if height ∉ (0.8, 2.0)
WALKER2D_ANGLE_BOUND   = 1.0           # env: done if |ang| ≥ 1.0
_WALKER2D_HEIGHT_CENTER = 0.5 * (WALKER2D_HEIGHT_BOUNDS[0] + WALKER2D_HEIGHT_BOUNDS[1])  # 1.4
_WALKER2D_HEIGHT_HALF   = 0.5 * (WALKER2D_HEIGHT_BOUNDS[1] - WALKER2D_HEIGHT_BOUNDS[0])  # 0.6


def _walker2d_quadratic_safety(height, angle, xp=np):
    """Per-step safety reward: −2·x_norm² + 1 per dim, additive.

    Both height and angle are first normalized so their safe zones map to
    [-1, 1], then the SAME quadratic shape applies to each:

        h_norm = (h − 1.4) / 0.6          # safe zone ↦ [-1, 1]
        a_norm = a / 1.0                   # safe zone ↦ [-1, 1]
        r_h = −2·h_norm²  +  1
        r_a = −2·a_norm²  +  1
        r_safety = r_h + r_a

    Per-step values (each dim independently: +1 at center, 0 at |x_norm|=1/√2,
    −1 at env bound):
        center (h=1.4, ang=0)              → +1 + +1 =  +2 (max)
        one dim centered, other at bound   → +1 + −1 =   0
        at both env bounds                 → −1 + −1 =  −2 (min in-bounds)
        beyond bounds (briefly, before done)→ < −2
    """
    # xp: 배열 모듈 (np 기본 = 기존 호출 그대로). MJX 오라클 env 가 jit 안에서 같은 식을 쓰도록 jnp 도 받는다 (D6.30 (2026-11-06), SSOT).
    h_norm = (xp.asarray(height, dtype=xp.float32) - _WALKER2D_HEIGHT_CENTER) / _WALKER2D_HEIGHT_HALF
    a_norm =  xp.asarray(angle,  dtype=xp.float32) / WALKER2D_ANGLE_BOUND
    r_h = -2.0 * h_norm ** 2 + 1.0
    r_a = -2.0 * a_norm ** 2 + 1.0
    return (r_h + r_a).astype(xp.float32)


# ── safe-CD safety 채널 가족 (2026-11-03, docs/safe_cd_definitions.md §5) ──────────────────────
# 사용자 (2026-11-03): "조합은 지금은 딱히 하지 말고, 일단 각각 safety utility로 돌려봐".
# 세 번째 채널 s_t 만 다르고 (speed, cost) 는 walker2dcd 와 같다. 모든 s_t 는 **스텝 합으로 쓸 수 있는** 양이어야 한다
# (AET 의 F 는 prefix-sum racc 의 함수) — 궤적 최솟값·이전 행동이 필요한 jerk 는 정의할 수 없다.
# 입력은 행동 뒤 상태 next_obs (buffer 는 next_observations, eval 은 env.step 의 obs 를 넘긴다).
_WALKER_ALIVE_BONUS = 1.0            # environment/walker2d.py: reward_speed = Δx/dt + 1
SAFETY_VMAX_WALKER2D = 2.3415        # Safety-Gymnasium SafetyWalker2dVelocity-v1 속도 문턱
SAFETY_SAT_LEVEL = 0.95              # |a_j| > 이 값이면 토크 포화로 센다


# S0·S1 은 xp (np 기본 / jnp) 를 받는다 — MJX 오라클 (experiments/oracle/mjx_mowalker_env.py) 이 jit 안에서
# 같은 함수를 부른다 (D6.30 (2026-11-06), 복사 금지 SSOT). np 경로의 출력은 그대로다.
def _safety_posture(actions, raw_rewards, next_obs, xp=np):
    """S0 (논문): 자세 그릇 (1−2h̄²)+(1−2θ̄²) ∈ [−2, 2]."""
    return _walker2d_quadratic_safety(next_obs[..., 0], next_obs[..., 1], xp=xp)


def _safety_velocity(actions, raw_rewards, next_obs, xp=np):
    """S1: 속도 제한을 지킨 스텝 = 1[v ≤ v_max], v = Δx/dt = reward_speed − alive."""
    v = xp.asarray(raw_rewards, xp.float32)[..., 0] - _WALKER_ALIVE_BONUS
    return (v <= SAFETY_VMAX_WALKER2D).astype(xp.float32)


# 스텝별 자세 문턱 (D6.31, 2026-11-06): 사용자 "자세도 스텝별 문턱으로 세봐" — 누적합 문턱은 앞 스텝의 여유가
#   지금의 위반을 가린다. s_th = 0.5: 데이터 스텝 S0 분포 p10 0.16 / p25 0.84 / p50 1.25, 궤적의 41% 가 위반 0.
SAFETY_POSTURE_STEP_TH = 0.5


def _safety_posture_ok(actions, raw_rewards, next_obs, xp=np):
    """S0ok: 자세 문턱을 지킨 스텝 = 1[S0 ≥ SAFETY_POSTURE_STEP_TH] (violog 에서 위반 스텝 수로 쓰인다)."""
    return (_safety_posture(actions, raw_rewards, next_obs, xp=xp) >= SAFETY_POSTURE_STEP_TH).astype(xp.float32)


def _walker2d_margin_parts(height, angle, xp=np):
    """(높이 여유, 각도 여유) — 각 종료 경계까지 반폭으로 나눈 거리. D6.37 에서 두 채널로 따로 쓰려고 분리."""
    lo, hi = WALKER2D_HEIGHT_BOUNDS
    h = xp.asarray(height, xp.float32)
    m_h = xp.minimum(h - lo, hi - h) / _WALKER2D_HEIGHT_HALF
    m_a = 1.0 - xp.abs(xp.asarray(angle, xp.float32)) / WALKER2D_ANGLE_BOUND
    return m_h, m_a


def _walker2d_boundary_margin(height, angle):
    """종료 경계까지의 여유 m ∈ (0,1] (허용 구간 안): 높이·각도를 각 반폭으로 나눈 가장 가까운 경계까지 거리."""
    m_h, m_a = _walker2d_margin_parts(height, angle)
    return np.minimum(m_h, m_a)


def _safety_margin(actions, raw_rewards, next_obs):
    """S2a: 평균 경계 여유 clip(m, 0, 1)."""
    return np.clip(_walker2d_boundary_margin(next_obs[..., 0], next_obs[..., 1]), 0.0, 1.0).astype(np.float32)


# D6.37 (2026-11-06): 높이·각도 경계 여유를 각각 채널로. 사용자 "둘 다 쓰면되잖아. 각각 log(높이) + log(각도)로
#   충분히 더 추가할 수 있으니까". 스텝 값 clip(·, 0, 1), inlog 문턱은 데이터 분위수 (p75/p90/p95).
def _safety_margin_height(actions, raw_rewards, next_obs, xp=np):
    """S_h: 높이 종료 경계까지 여유 clip(m_h, 0, 1)."""
    return xp.clip(_walker2d_margin_parts(next_obs[..., 0], next_obs[..., 1], xp=xp)[0], 0.0, 1.0).astype(xp.float32)


def _safety_margin_angle(actions, raw_rewards, next_obs, xp=np):
    """S_θ: 각도 종료 경계까지 여유 clip(m_θ, 0, 1)."""
    return xp.clip(_walker2d_margin_parts(next_obs[..., 0], next_obs[..., 1], xp=xp)[1], 0.0, 1.0).astype(xp.float32)


# D6.39 (2026-11-06): 높이 위반 지표의 스텝 문턱. 사용자 선택 0.25 (z ≈ 0.95, 전문가 데이터 스텝 m^h 하위 0.1%,
#   데이터 궤적 99% 이상이 위반 0 → 위반은 사실상 낙상 직전에만).
SAFETY_HEIGHT_STEP_TH = 0.25


def _safety_height_ok(actions, raw_rewards, next_obs, xp=np):
    """S_hok: 높이 여유가 문턱 이상인 스텝 = 1[m^h ≥ SAFETY_HEIGHT_STEP_TH] (위반 = 1 − S_hok, 못 산 스텝도 위반)."""
    m_h = _walker2d_margin_parts(next_obs[..., 0], next_obs[..., 1], xp=xp)[0]
    return (m_h >= SAFETY_HEIGHT_STEP_TH).astype(xp.float32)


def _safety_saturation(actions, raw_rewards, next_obs):
    """S3: 1 − (포화 관절 비율), 포화 = |a_j| > SAFETY_SAT_LEVEL."""
    a = np.asarray(actions, np.float32)
    return (1.0 - np.mean(np.abs(a) > SAFETY_SAT_LEVEL, axis=-1)).astype(np.float32)


# reshape 이름 → safety 함수들 (채널 2, 3, … 순서). walker2d_cd_safe (S0) 는 논문 이름 그대로 둔다.
# walker2d_cd_safe_pv (2026-11-05): S0 자세 + S1 속도를 **각자의 문턱을 가진 두 제약**으로 동시에 건다.
#   사용자: "velocity랑 posture에 대해서 동시에 safety로 … 이 둘의 combination은 가능할지도. 그게 이 walker env에서
#   safety utility definition에 제일 부합하기도하고." → 조합 방식 = 제약 2개 따로 (채널 4개, 문턱 2개) 선택.
WALKER2D_CD_SAFE_KINDS = {
    "walker2d_cd_safe":        (_safety_posture,),                     # S0
    "walker2d_cd_safe_vel":    (_safety_velocity,),                    # S1
    "walker2d_cd_safe_margin": (_safety_margin,),                      # S2a
    "walker2d_cd_safe_sat":    (_safety_saturation,),                  # S3
    "walker2d_cd_safe_pv":     (_safety_posture, _safety_velocity),    # S0 + S1 (두 제약)
    "walker2d_cd_safe_pok":    (_safety_posture_ok,),                  # S0ok 스텝별 자세 문턱 (D6.31)
    "walker2d_cd_safe_pokv":   (_safety_posture_ok, _safety_velocity), # S0ok + S1 (D6.31)
    "walker2d_cd_safe_ha":     (_safety_margin_height, _safety_margin_angle),  # 높이 여유 + 각도 여유 (D6.37)
    "walker2d_cd_safe_h":      (_safety_margin_height,),               # 높이 여유만 (D6.38: 낙상 전부 높이 경계)
    "walker2d_cd_safe_hm":     (_safety_margin_height, _safety_height_ok),  # 채점용: 높이 여유 + 높이 준수 (D6.39)
}


def _make_cd_safe_reshapes(safety_fns):
    """(trajectory, step) reshape 한 쌍 — (speed, cost) 는 walker2dcd 와 같고 채널 2.. 가 safety_fns 순서."""
    def traj(actions, raw_rewards, observations=None):
        if observations is None:
            raise ValueError("safe-CD reshape 는 next observation (height, angle) 이 필요하다")
        sc = _reshape_cost_recovery_trajectory(actions, raw_rewards, _WALKER_HC_ENERGY_CONST)
        safety = [fn(actions, raw_rewards, observations) for fn in safety_fns]
        for x in safety:
            assert x.shape == sc.shape[:-1], (x.shape, sc.shape)
        return np.concatenate([sc] + [x[..., None] for x in safety], axis=-1).astype(np.float32)

    def step(action, info_obj, observation=None):
        if observation is None:
            raise ValueError("safe-CD reshape 는 next observation (height, angle) 이 필요하다")
        sc = _reshape_cost_recovery_step(action, info_obj, _WALKER_HC_ENERGY_CONST)
        # 배치 1 로 같은 함수를 부른다 — 데이터 경로와 평가 경로가 같은 식을 쓰게 (SSOT)
        args = (np.asarray(action, np.float32)[None], np.asarray(info_obj, np.float32)[None],
                np.asarray(observation, np.float32)[None])
        safety = [fn(*args) for fn in safety_fns]
        for x in safety:
            assert x.shape == (1,), x.shape
        return np.append(sc, [x[0] for x in safety]).astype(np.float32)
    return traj, step


assert tuple(WALKER2D_CD_SAFE_KINDS) == WALKER2D_CD_SAFE_RESHAPES, (tuple(WALKER2D_CD_SAFE_KINDS), WALKER2D_CD_SAFE_RESHAPES)

# 스텝 safety 값의 범위 (lo, hi) — violog 의 위반량 D = Σ_산 스텝 (hi − s_t) + (T − n)(hi − lo) 에 쓴다 (D6.30).
SAFETY_STEP_RANGE = {
    _safety_posture:    (-2.0, 2.0),     # (1−2h̄²)+(1−2θ̄²)
    _safety_velocity:   (0.0, 1.0),      # 1[v ≤ v_max]
    _safety_posture_ok: (0.0, 1.0),      # 1[S0 ≥ s_th] (D6.31)
    _safety_margin_height: (0.0, 1.0),   # D6.37
    _safety_margin_angle:  (0.0, 1.0),   # D6.37
    _safety_height_ok:     (0.0, 1.0),   # D6.39
    _safety_margin:     (0.0, 1.0),      # clip(m, 0, 1)
    _safety_saturation: (0.0, 1.0),      # 1 − 포화 비율
}


def cd_safety_step_ranges(reshape):
    """safe-CD reshape → safety 채널마다 (lo, hi) 튜플 (채널 순서 = WALKER2D_CD_SAFE_KINDS)."""
    return tuple(SAFETY_STEP_RANGE[fn] for fn in WALKER2D_CD_SAFE_KINDS[reshape])
_CD_SAFE_PAIRS = {k: _make_cd_safe_reshapes(fn) for k, fn in WALKER2D_CD_SAFE_KINDS.items()}
reshape_walker2d_cd_safe_trajectory, reshape_walker2d_cd_safe_step = _CD_SAFE_PAIRS["walker2d_cd_safe"]


# ── Walker2d 성분 + 안전 (2026-10-25, docs/aet_objectives_worksheet.md) ──────────────────
# 사용자: "필요한 리워드·코스트·안전 항목은 컴포넌트별로 분리해 계산 가능하면 env 에서 추출해도 된다",
# "safe-CD 처럼 safety 를 따로 계산". environment/walker2d.py 의 두 목적: obj0 = vx + alive, obj1 = 4 − ‖a‖² + alive.
# ★ alive(상수 1)만 떼는 reshape 는 만들지 않는다 — 스텝별 min-max 정규화가 상수 이동을 지워서 정규화 return 이
#   원 목적(reshape=none)과 **완전히 같다** (중복 경로). 새 정보는 안전 채널뿐이라 그것만 붙인다.
# 안전 = _walker2d_quadratic_safety (safe-CD 와 같은 식, 다음 관측의 높이·각도).
WALKER2D_COMP_REWARD_DIM = 3


def reshape_walker2d_comp_trajectory(actions, raw_rewards, observations=None):
    """[T, 3] (speed obj0, energy obj1, safety) — 원 두 목적 + 안전 (observations 필수)."""
    if observations is None:
        raise ValueError("walker2d_comp needs observations (height, angle) for the safety channel")
    safety = _walker2d_quadratic_safety(observations[..., 0], observations[..., 1])
    return np.concatenate([np.asarray(raw_rewards, np.float32), safety[..., None]], axis=-1).astype(np.float32)


def reshape_walker2d_comp_step(action, info_obj, observation=None):
    if observation is None:
        raise ValueError("walker2d_comp needs the observation (height, angle) for the safety channel")
    return np.append(np.asarray(info_obj, np.float32),
                     _walker2d_quadratic_safety(observation[0], observation[1])).astype(np.float32)


# Dispatch tables — extend when new reshape modes are added. Keep the per-step
# and per-trajectory tables aligned so the buffer and evaluator never disagree.
RESHAPE_TRAJECTORY = {
    "antcd":             reshape_antcd_trajectory,
    "walker2dcd":        reshape_walker2dcd_trajectory,
    "half_cheetahcd":    reshape_half_cheetahcd_trajectory,
    "walker2d_cd_safe":  reshape_walker2d_cd_safe_trajectory,
    "walker2d_comp":     reshape_walker2d_comp_trajectory,
}
RESHAPE_STEP = {
    "antcd":             reshape_antcd_step,
    "walker2dcd":        reshape_walker2dcd_step,
    "half_cheetahcd":    reshape_half_cheetahcd_step,
    "walker2d_cd_safe":  reshape_walker2d_cd_safe_step,
    "walker2d_comp":     reshape_walker2d_comp_step,
}
RESHAPE_REWARD_DIM = {
    "antcd":             ANTCD_REWARD_DIM,
    "walker2dcd":        WALKER_CD_REWARD_DIM,
    "half_cheetahcd":    HALFCHEETAH_CD_REWARD_DIM,
    "walker2d_cd_safe":  WALKER2D_CD_SAFE_REWARD_DIM,
    "walker2d_comp":     WALKER2D_COMP_REWARD_DIM,
}
# S1–S3 safe-CD reshape 도 같은 세 표에 등록한다 (S0 는 위에 이름으로 있다).
for _name in WALKER2D_CD_SAFE_RESHAPES[1:]:
    RESHAPE_TRAJECTORY[_name], RESHAPE_STEP[_name] = _CD_SAFE_PAIRS[_name]
    RESHAPE_REWARD_DIM[_name] = 2 + len(WALKER2D_CD_SAFE_KINDS[_name])   # (speed, cost) + safety 채널 수


# ── trajectory-level scoring + top-K% 필터 (filtered-BC 바닥선용) ──────────────
# CobbDouglasUtility._transform 의 dim 0 과 **같은 수식**을 궤적 누적 return 에 적용한다:
#     CD(R) = Reward^ρ / Cost^(1−ρ)
# 여기서 Cost 는 reshape 단계에서 이미 per-step `1 + ‖a‖²` 로 만들어졌으므로 CD 단에
# 더할 오프셋이 없다 (CD_COST_UNIT 주석 참조).
#
# Use cases:
#   - Filtered behavior cloning: train BC on top-K% trajectories' (s,a) pairs
#     as an offline-RL baseline. See BC_learner.py.
#   - Sanity-checking dataset CD distribution against trained-policy CD scores.

def trajectory_cd_scores(trajectories_reshaped, reward_min_per_obj, reward_max_per_obj,
                          rho: float, reward_scaling_factor=None):
    """Compute per-trajectory CD scores using cumulative normalized (reward, cost).

    trajectories_reshaped:    list of [T_i, reward_dim] arrays (post-reshape, post-truncation)
    reward_min/max_per_obj:   [reward_dim]; bounds used for normalization. CD 경로에서는
                              setup.resolve_reward_bounds 가 (0, T) 로 고정한다 — shift 를
                              넣으면 CD 순위가 바뀌기 때문이다 (scale 은 무해).
    rho:                      Cobb-Douglas elasticity ∈ [0, 1].
    reward_scaling_factor:    optional [reward_dim]; defaults to ones.

    Returns: np.ndarray [num_traj] — CD score per trajectory.
    """
    rmin = np.asarray(reward_min_per_obj, dtype=np.float32)
    rmax = np.asarray(reward_max_per_obj, dtype=np.float32)
    rsf = (np.ones_like(rmin) if reward_scaling_factor is None
           else np.asarray(reward_scaling_factor, dtype=np.float32))
    cum = []
    for r in trajectories_reshaped:
        norm = rsf[None, :] * min_max_normalization(r.astype(np.float32), rmin[None, :], rmax[None, :])
        cum.append(norm.sum(axis=0))
    cum = np.stack(cum, axis=0)                       # [num_traj, reward_dim]
    return cd_score(cum[..., CD_REWARD_CH], cum[..., CD_COST_CH], rho, xp=np)


TRAJ_FILTER_KINDS = ("cd",)   # "corner" 제거 2026-09-20 — 아래 주석 참조


def trajectory_returns_normalized(trajectories_reshaped, reward_min_per_obj,
                                  reward_max_per_obj, reward_scaling_factor=None):
    """궤적별 **정규화 누적 return** [n_traj, D] — 모든 top-K 필터의 공통 입력 (SSOT).

    버퍼가 racc 를 쌓는 공간과 같아야 필터 결과가 학습/평가와 어긋나지 않는다.
    """
    rmin = np.asarray(reward_min_per_obj, dtype=np.float32)
    rmax = np.asarray(reward_max_per_obj, dtype=np.float32)
    rsf = (np.ones_like(rmin) if reward_scaling_factor is None
           else np.asarray(reward_scaling_factor, dtype=np.float32))
    cum = [ (rsf[None, :] * min_max_normalization(r.astype(np.float32),
                                                  rmin[None, :], rmax[None, :])).sum(axis=0)
            for r in trajectories_reshaped ]
    return np.stack(cum, axis=0)                      # [num_traj, D]


def filter_top_k(trajectories_reshaped, reward_min_per_obj, reward_max_per_obj,
                 k_pct: float, *, kind: str = "cd", rho: float = 0.5,
                 reward_scaling_factor=None):
    """상위 K% 궤적의 **원본 인덱스**를 돌려준다. `kind` 가 무엇으로 자를지 정한다.

    k_pct ∈ (0, 1].  0.1 → 상위 10% (50,000 중 5,000). 최소 1 개는 남긴다.

    kind="cd"      CD 점수 Reward^ρ/Cost^(1−ρ) 상위 K% — (reward, cost) 레이아웃 전용.
                   필터된 BC 를 CD 목적의 바닥선으로 쓸 때.

    ★ kind="corner" 는 제거했다 (2026-09-20, 사용자 지시). "목적별 상위 (K/D)% 의 합집합"
      이라는 규칙에 정당화가 없다 — 왜 목적당 절반인지, 왜 합집합인지, 왜 10% 인지 답할 수
      없고 목적이 3 개 이상이면 정의조차 모호하다. 무엇보다 **결과를 본 뒤에 고른 post-hoc
      휴리스틱**이라 리뷰어에게 설명할 수 없다. 대체 기준선은 "우리가 최적화하는 목적
      자신으로 부분집합을 고르는 것" (AET-greedy) 이다 — 임의 상수가 없고 목적 개수와
      무관하게 정의되며, 왜 그렇게 골랐냐는 질문에 한 줄로 답한다.
    """
    if not 0.0 < k_pct <= 1.0:
        raise ValueError(f"k_pct must be in (0, 1], got {k_pct}")
    if kind not in TRAJ_FILTER_KINDS:
        raise ValueError(f"kind must be in {TRAJ_FILTER_KINDS}, got {kind!r}")
    # ★ k_pct >= 1.0 은 **어느 kind 든 no-op** 이어야 한다. corner 는 목적당 (K/D)% 를
    #   합집합하므로 K=1.0 이어도 전체가 되지 않는다 (겹침 때문에 ~75%). 그러면
    #   (a) 기존 BC 바닥선(top100pct = 전 궤적 모방)의 의미가 조용히 바뀌고
    #   (b) setup._buffer_cache_dir 의 "k_pct>=1.0 이면 원본과 byte-identical" 가정이 깨져
    #       엉뚱한 캐시를 HIT 한다. 그래서 여기서 먼저 잘라낸다.
    if k_pct >= 1.0:
        return np.arange(len(trajectories_reshaped))
    cum = trajectory_returns_normalized(trajectories_reshaped, reward_min_per_obj,
                                        reward_max_per_obj, reward_scaling_factor)
    n, D = cum.shape
    if kind == "cd":
        scores_ = cd_score(np.maximum(cum[..., CD_REWARD_CH], 0.0),
                           cum[..., CD_COST_CH], rho, xp=np)
        return np.argsort(-scores_)[:max(1, int(round(n * k_pct)))]
    raise ValueError(f"unsupported kind={kind!r}")


def cd_score(reward, cost, rho: float, *, xp=jnp):
    """★ CD 의 **단일 소스** — 이 수식 외에는 어디에도 CD 를 쓰지 않는다 (SSOT).

        CD(Reward, Cost) = Reward^ρ / Cost^(1−ρ)

    Reward / Cost 는 **궤적 누적** 값이고, per-step 채널은 reshape 가 만든다:
        reward_t = (env 별 순수 reward, antcd 는 vx+vy)
        cost_t   = CD_COST_UNIT + ‖a_t‖²          ← `+1` 은 여기 한 곳에만 있다

    정규화: scale(나눗셈)은 CD 순위를 **전혀** 바꾸지 않는다 —
        R→R/a, C→C/b  ⇒  CD → CD/(a^ρ·b^(1−ρ)) 는 모든 정책에 공통인 양의 상수.
        (실측 2026-09-16: 50,000 궤적에서 배율 std 7.9e-17, Spearman 1.000000000000)
    반면 shift(−rmin)는 순위를 바꾼다 — rmin[reward]<0 이면 정지 정책이 공짜 이득을
    받아 ρ=0.5 에서 정지 21.7 vs 고속 1.36 으로 **정지가 1등**이 된다. 그래서 CD 경로의
    reward_min 은 0 으로 고정한다 (setup.resolve_reward_bounds).

    엣지케이스:
        Reward ≤ 0  → 0   (음수 밑의 실수 거듭제곱이 미정의. 후진 정책은 CD=0)
        Cost  == 0  → 0   (racc_0 = [0,0] 에서만 발생. 0/0 회피)
        ρ == 0      → 분자 상수 1 (순수 비용최소화)
    """
    num = cd_numerator_power(xp.maximum(reward, 0.0), rho, xp=xp)
    pos_cost = cost > 0.0
    den = xp.power(xp.where(pos_cost, cost, 1.0), 1.0 - rho)
    return xp.where(pos_cost, num / den, 0.0)


# CD 의 두 형태 (exp A, 2026-10-24 사용자 채택 — docs/CD_redesign_worksheet.md §3):
#   total  F(R)    = CD(R)                — 기존. 2ρ−1 차 동차라 telescoping 보상이 t^{2ρ−2} 로 몰린다
#                                            (ρ<0.5 초반 spike·이후 음수, ρ=0.5 는 t≈50 에 최종값).
#   rate   F(R, n) = n · CD(R·T/n)        — n = 지금까지 스텝 수, R·T/n = 스텝당 평균 (scale-only 정규화라
#                                            R = Σraw/T). 고정 길이(n=T)에서 F = T·CD(R) → 같은 최적 정책
#                                            (데이터 순위 Spearman 1.0). 스텝 보상 ≈ 지금까지의 효율 → 고르게 퍼진다.
#                                            조기 종료(n<T)는 남은 스텝의 효율을 잃는다 — 목적이 달라진다.
CD_FORMS = ("total", "rate")


def cd_rate_score(reward, cost, steps, horizon, rho: float, *, xp=jnp):
    """rate 형태의 **에피소드 점수** (n/T)·CD(R·T/n) — 길이 T 에피소드에서는 cd_score(R) 와 같다.

    reward/cost = scale-only 정규화 누적 (Σraw/T), steps = 누적한 스텝 수 n (≥0), horizon = T.
    n=0 이면 0 (빈 누적). 학습 F 는 이 값에 T 를 곱한 n·CD(평균) 이다 (CobbDouglasUtility).
    """
    n = xp.asarray(steps).astype(xp.asarray(reward).dtype)
    assert n.shape == xp.shape(reward), (n.shape, xp.shape(reward))
    pos = n > 0
    n_safe = xp.where(pos, n, 1.0)
    rate = cd_score(reward * horizon / n_safe, cost * horizon / n_safe, rho, xp=xp)
    return xp.where(pos, (n_safe / horizon) * rate, 0.0)


class CobbDouglasUtility:
    """(reward, cost) 목적을 가진 env 용 utility — 선택적으로 세 번째 "safety/floor"
    채널과 분리해 쓴다.

    Input layout (buffer 가 주는 누적 R):
        R[..., 0] = reward  (CD 분자 — 많을수록 좋다)
        R[..., 1] = cost    (CD 분모 — 적을수록 좋다. reshape 가 이미 1+‖a‖² 로 만든다)
        R[..., 2] = R2  (reward_dim ≥ 3 일 때만) — threshold 분기에 들어간다.

    Output dim 0: amp · Reward^ρ / Cost^(1−ρ)           ← cd_score (SSOT), form='total'
                  amp · n · CD(R·T/n)                   ← cd_rate_score × T, form='rate' (steps=n 필요)
    Output dim 1: piecewise_log(target − threshold) when threshold is set; 0 otherwise.

    rho ∈ [0, 1]:
      rho = 1    -> pure reward maximization     (= R_0)
      rho = 0    -> pure cost minimization       (= 1 / R_1)
      rho = 0.5  -> symmetric efficiency         (= sqrt(R_0 / R_1))   ← default

    amp 은 dim0 의 크기만 바꾸는 **양의 상수배**다 (G=linear 에서는 β 를 amp 로 나누는
    것과 수학적으로 같다). scale 이므로 정책 순위에 영향이 없다 — cd_score docstring 참조.

    ★ 2026-09-16: mode("original"/"modified"/"canonical"), denom_offset, offset_shift,
      horizon, denom_offset_step, numerator_eps 를 **전부 제거**했다. CD 는 위 한 식뿐이다.
      옛 `modified` 는 정규화 shift 를 되돌린 뒤 δ=T·δ_step 을 더한 형태였는데, 이는 지금의
      per-step `+1` 과 상수배 차이(Spearman 0.99999)일 뿐이면서 고정 T 를 가정해 조기종료
      에피소드(t<T/2)의 분자를 0 으로 뭉개는 버그가 있었다. 옛 `original` 은 정규화 shift 를
      그대로 둬서 정지 정책이 1등이 되는 치명적 오류였다.
    """

    def __init__(self, rho: float = 0.5, threshold=None, amp: float = 1.0, form: str = "total",
                 horizon=None, safety: str = "plog", kappa: float = 1.0, safety_range=None):
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"CobbDouglas rho must be in [0, 1], got {rho}")
        if form not in CD_FORMS:
            raise ValueError(f"CobbDouglas form must be in {CD_FORMS}, got {form!r}")
        if form == "rate" and not (horizon and horizon > 0):
            raise ValueError(f"form='rate' needs horizon T > 0 (got {horizon!r})")
        # safe-CD 결합 방식 (2026-11-03, docs/safe_cd_definitions.md §4, worksheet D6.20):
        #   plog    : dim1 = plog(S − th)          — 기존 (G=plog 와 쓰면 log 가 안팎에 걸린다)
        #   linear  : dim1 = T·S − th (raw 합 단위) — 논문 식 J = log E[CD] + log E[ΣS − th] (G=plog 가 밖의 log)
        #   product : dim0 ×= max(S, 0)^κ, dim1 = 0 — F = CD_rate·S̄^κ 한 궤적 안에서 둘을 함께 (G=linear)
        #   inlog   : F = plog([T·CD_rate, ΣS_k − th_k]) — linear 의 바깥 log 를 궤적 안으로 (G=linear, ESR).
        #             사용자 2026-11-06 (D6.28): "F에 cobbdouglas랑 safety utility가 걸리는게 맞지 … 그래야 한
        #             trajectory내에서 CD랑 safety를 지키도록 학습을 강제할 수 있잖아" / CD 항에도 log.
        # 사용자: "하나의 trajectory 내에서 동시에 cd rate과 safety가 balance있게 max되기를 원하는거야" →
        #   "multiplicative form ... might give the same results as outside log form. so let's test both."
        #   violog  : F = [plog(T·CD_rate), −log(1 + D_k)], D_k = 위반량 = Σ_산 스텝 (hi_k − s_t) + (T − n)(hi_k − lo_k)
        #             (못 산 스텝 = 최대 위반). 위반 0 근처에서 기울기가 가장 커서 문턱 위 포화가 없다 (G=linear, ESR).
        #             사용자 2026-11-06 (D6.30): "inlog에 들어가는 safety항을 좀 바꿔야할듯. 어떻게해야 포화가 안돼지?"
        #             → "위반량에 log" 선택.
        if safety not in CD_SAFETY_MODES:
            raise ValueError(f"CobbDouglas safety must be in {CD_SAFETY_MODES}, got {safety!r}")
        if safety == "violog":
            if threshold is not None:
                raise ValueError("safety='violog' 는 문턱이 없다 (위반량 D = 완벽 대비 부족분)")
            if not safety_range:
                raise ValueError("safety='violog' 는 safety_range ((lo, hi) 채널마다) 가 필요하다")
            safety_range = tuple((float(lo), float(hi)) for lo, hi in safety_range)
        elif safety_range is not None:
            raise ValueError(f"safety_range 는 safety='violog' 에서만 (got safety={safety!r})")
        if safety != "plog" and form != "rate":
            raise ValueError(f"safety={safety!r} 는 form='rate' 에서만 정의했다 (raw 합 단위에 horizon T 가 필요)")
        if safety in CD_SAFETY_THRESHOLDED and threshold is None:
            raise ValueError(f"safety={safety!r} 는 threshold (raw 합 단위 S_th) 가 필요하다")
        # 문턱은 safety 채널마다 하나 (2026-11-05, walker2d_cd_safe_pv 의 S0+S1 두 제약). 한 개면 float 로 둔다 —
        # 기존 런의 config (float) 와 캐시·디렉터리 이름이 그대로 맞게.
        if threshold is not None and np.ndim(threshold) > 0:
            threshold = [float(t) for t in np.ravel(threshold)]
            if len(threshold) == 1:
                threshold = threshold[0]
            elif safety not in CD_SAFETY_THRESHOLDED:
                raise ValueError(f"문턱 여러 개 ({threshold}) 는 safety ∈ {CD_SAFETY_THRESHOLDED} 에서만 (채널마다 ΣS_k − th_k)")
        if safety == "product" and threshold is not None:
            raise ValueError("safety='product' 는 문턱이 없다 (P1: F = CD_rate·S̄^κ)")
        if not kappa > 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        self.safety = safety
        self.safety_range = safety_range
        self.kappa = float(kappa)
        self.rho = float(rho)
        self.threshold = (None if threshold is None else
                          tuple(threshold) if isinstance(threshold, list) else float(threshold))
        self.amplifier = amp
        self.form = form
        self.horizon = None if horizon is None else float(horizon)

    @property
    def needs_steps(self):
        """rate 형태는 누적값 R 만으로 정해지지 않는다 — 누적한 스텝 수가 필요하다."""
        return self.form == "rate"

    @staticmethod
    def _piecewise_log_threshold(reward, threshold):
        """utility.piecewise_log 을 (reward − threshold) 에 적용 (단일 소스 재사용)."""
        return piecewise_log(reward - threshold)

    def _transform(self, R, steps=None):
        # ★ 채널 이름 규약 (2026-09-12): dim0 = `reward`(분자), dim1 = `cost`(분모).
        #   r0/r1 이라 부르지 않는다 — MO-Ant 원목적 (vx, vy) 로 오독되기 때문.
        reward = R[..., CD_REWARD_CH]
        cost   = R[..., CD_COST_CH]
        if self.form == "rate":
            if steps is None:
                raise ValueError("CobbDouglasUtility(form='rate') needs steps (누적 스텝 수)")
            steps = jnp.asarray(steps, dtype=reward.dtype).reshape(reward.shape)
            # × T: 학습 F = n·CD(평균) — 에피소드 점수(cd_rate_score)의 T 배라 스텝 보상이 O(CD).
            dim0 = self.amplifier * self.horizon * cd_rate_score(
                reward, cost, steps, self.horizon, self.rho)
        else:
            dim0 = self.amplifier * cd_score(reward, cost, self.rho)
        if self.safety != "plog":
            assert R.shape[-1] >= 3, f"safety={self.safety!r} 는 3 채널 (speed, cost, safety) 레이아웃이 필요하다"
            safe = R[..., 2:]                     # S̄_k = Σ s_t^(k) / T (scale-only 정규화 누적), 채널 k = 2..
            if self.safety == "product":
                # S0 는 스텝 값이 음수일 수 있다 → 0 에서 자른다 (S̄ ≤ 0 이면 F = 0). 여러 채널이면 곱한다.
                dim0 = dim0 * jnp.prod(jnp.power(jnp.maximum(safe, 0.0), self.kappa), axis=-1)
                return jnp.stack([dim0, jnp.zeros_like(dim0)], axis=-1)
            if self.safety == "violog":
                lo = jnp.asarray([r[0] for r in self.safety_range], dtype=safe.dtype)
                hi = jnp.asarray([r[1] for r in self.safety_range], dtype=safe.dtype)
                assert lo.shape[0] == safe.shape[-1], f"safety_range {lo.shape[0]} 개 ≠ safety 채널 {safe.shape[-1]} 개"
                n = steps[..., None]
                T = self.horizon
                D = jnp.maximum(T * hi - T * safe - (T - n) * lo, 0.0)   # = Σ_산 (hi − s_t) + (T − n)(hi − lo)
                return jnp.concatenate([piecewise_log(dim0)[..., None], -jnp.log1p(D)], axis=-1)
            # linear: raw 합 단위 (T·S̄ = Σ s_t) 에서 문턱을 뺀다 — 논문 R_th = 0.8H 와 같은 단위.
            # 채널마다 제약 하나: F = [T·CD_rate, ΣS_1 − th_1, ΣS_2 − th_2, …] → G=plog 가 각 제약에 바깥 log 를 건다.
            th = jnp.asarray(self.threshold if isinstance(self.threshold, tuple) else (self.threshold,),
                             dtype=safe.dtype)
            assert th.shape[0] == safe.shape[-1], f"문턱 {th.shape[0]} 개 ≠ safety 채널 {safe.shape[-1]} 개"
            vec = jnp.concatenate([dim0[..., None], self.horizon * safe - th], axis=-1)
            # inlog: 같은 벡터에 궤적마다 plog — G=linear 가 합하면 J = E_τ[Σ_k plog(F_k(τ))].
            return piecewise_log(vec) if self.safety == "inlog" else vec
        # threshold-shaping in dim 1. 3-D 레이아웃(walker2d_cd_safe = (speed, cost, safety))
        # 이면 세 번째 채널을 쓴다 — floor 를 CD 분자와 분리하기 위해서다. 2-D 면 reward
        # 채널로 되돌아간다 (halfcheetahcd/walker2dcd back-compat).
        target = R[..., 2] if R.shape[-1] >= 3 else reward
        assert not isinstance(self.threshold, tuple), "문턱 여러 개는 safety='linear' 에서만"
        if self.threshold is None:
            dim1 = jnp.zeros_like(reward)
        else:
            dim1 = self._piecewise_log_threshold(target, self.threshold)
        return jnp.stack([dim0, dim1], axis=-1)

    def __call__(self, R, *, keep_dims=False, steps=None):
        X = self._transform(R, steps)
        return X if keep_dims else jnp.sum(X, axis=-1)
