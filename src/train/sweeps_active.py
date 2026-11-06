"""Active SWEEPS list — the dispatcher's single source of "what to run right now".

This file is the toggle layer between the sweep dict definitions
(`sweep_defs/*.py`) and the dispatcher (`dispatcher.run`). Comment / uncomment
entries here to change what the next launch actually executes.

Adding a brand-new sweep:
    1. Define the dict in the appropriate `sweep_defs/<env>.py`.
    2. Import the name here (the `import *` below already brings every name from
       every env module — usually no edit needed).
    3. Add the name to the `SWEEPS = [
    # ── MO-Ant CD 본 격자 (idx 0) — ρ{0.2,0.3,0.4} × β{1e-5,1e-6} × seed{1,2,3} = 18 job
    ANT_EXPERT_CD_GRID,

    # ── BC 바닥선 (idx 1) — 데이터셋을 그대로 따라 하는 정책
    ANT_EXPERT_CD_BC,

    # ── BC 바닥선 (AET/ESR/SER 표용, idx 2) — reshape=none, base MDP 의 π(a|s)
    ANT_EXPERT_BC,

    # ── ★ racc-조건부 BC 바닥선 (idx 3) — augmented MDP 의 π(a|s, racc).
    #    알고리즘·오라클과 **같은 정책 클래스**의 바닥선. sweep_defs/ant.py 주석 참조.
    ANT_EXPERT_BC_RACC,



    # ══ mjx-native 데이터셋 실험 (idx 6~8) ═══════════════════════════════════
    # 데이터·알고리즘·오라클을 **mjx 한 물리**에 모아 sim-to-sim 격차를 없앤 첫 비교.
    # 데이터: preference 전문가 13 점을 원본 궤적의 초기상태에서 결정론 롤아웃 (1 대 1).
    #   품질 게이트 통과 (2026-09-17): raw R̄ 원본의 98~99%, 방향 평균 −1.0°,
    #   데이터셋 AET 13.204 vs 원본 13.508, 조기종료 1.69%.
    ANT_MJX_AETDICE,
    ANT_MJX_BC_RACC,
...]` list below.
"""
from src.train.sweep_defs import *  # noqa: F401,F403


# ============================================================
# ✏️  Active sweeps
# ============================================================
SWEEPS = [
    # ── MO-Ant CD 본 격자 (idx 0) — ρ{0.2,0.3,0.4} × β{1e-5,1e-6} × seed{1,2,3} = 18 job
    ANT_EXPERT_CD_GRID,

    # ── BC 바닥선 (idx 1) — 데이터셋을 그대로 따라 하는 정책
    ANT_EXPERT_CD_BC,

    # ── BC 바닥선 (AET/ESR/SER 표용, idx 2) — reshape=none, base MDP 의 π(a|s)
    ANT_EXPERT_BC,

    # ── ★ racc-조건부 BC 바닥선 (idx 3) — augmented MDP 의 π(a|s, racc).
    #    알고리즘·오라클과 **같은 정책 클래스**의 바닥선. sweep_defs/ant.py 주석 참조.
    ANT_EXPERT_BC_RACC,



    # ══ mjx-native 데이터셋 실험 (idx 6~8) ═══════════════════════════════════
    # 데이터·알고리즘·오라클을 **mjx 한 물리**에 모아 sim-to-sim 격차를 없앤 첫 비교.
    # 데이터: preference 전문가 13 점을 원본 궤적의 초기상태에서 결정론 롤아웃 (1 대 1).
    #   품질 게이트 통과 (2026-09-17): raw R̄ 원본의 98~99%, 방향 평균 −1.0°,
    #   데이터셋 AET 13.204 vs 원본 13.508, 조기종료 1.69%.
    ANT_MJX_AETDICE,
    ANT_MJX_BC_RACC,

    # ── R2: MO-PointMaze-v10 (10-obj) AET-DICE ──
    MO_POINTMAZE_EXPERT_AET,

    # ── Exp 0 (walking ρ figure). Launch: --sweeps 0 1 ──
    WALKER2D_AMATEUR_EXP0_CD,       # idx 0 — cd-only ρ{1.0,0.5,0.0}
    WALKER2D_AMATEUR_EXP0_SAFE_CD,  # idx 1 — cd+safety ρ{1.0,0.5,0.0}

    # ── Table 4 AET-DICE reproduction (MO-Ant Expert, G-plog × F-exp, 5 seeds) ──
    # Index 0 → launch with `--sweeps 0` to run ONLY this (leaves the walker2d
    # entries below active for the dispatcher's own use but untargeted).
    ANT_EXPERT_PLOG_EXP,      # idx 0 — AET-DICE (all β, done/clipped)
    ANT_EXPERT_PLOG_EXP_B6,   # idx 1 — AET-DICE β1e-6 ×5 seed, CLIP-OFF check
    # Table 4 baselines (5 seeds) — idx 1,2,3. Launch with `--sweeps 1 2 3`.
    ANT_EXPERT_OPTIDICE,      # idx 1 — LIN-[...] rows (OptiDICE)
    ANT_EXPERT_FAIRDICE,      # idx 2 — FAIRDICE row
    ANT_EXPERT_IQL_EXP,       # idx 3 — ESR-IQL rows

    # ══ amateur 3 종 (idx 17~19, 2026-09-17 사용자 지시) ═══════════════════
    # offline env 트랙. 실행: --sweeps 17 18 19
    ANT_AMATEUR_AETDICE,        # idx 17
    ANT_AMATEUR_BC_RACC,        # idx 18 — BC (필터 없음)
    ANT_AMATEUR_AETDICE_HP,     # idx 20 — β 6 × musf 5 = 30, seed 1 고정
    ANT_AMATEUR_AETDICE_RACCRATE, # idx 21 — racc 인코딩 rate 로 AETDICE 재학습
    ANT_AMATEUR_BC_ZCOND1,      # idx 22 — 잠재 z 조건부 단일 정책 (w 상위 1%)
    ANT_AMATEUR_BC_ZCOND5,      # idx 23 — 같은 것, 상위 5%
    ANT_AMATEUR_BC_WTOP10,      # idx 22 — 학습된 w 상위 10% 하드필터 + BC
    ANT_AMATEUR_BC_WTOP25,      # idx 22 — 상위 25%
    ANT_AMATEUR_BC_WTOP5,       # idx 23 — 상위 5%
    ANT_AMATEUR_BC_WTOP15,      # idx 24 — 상위 15%
    ANT_AMATEUR_BC_WTOP33,      # idx 25 — 상위 33% (= w>0 지지집합 크기)
    # 모드별 전문가 (idx 26~31) — 에피소드 단위 커밋. sweep_defs/ant.py 주석 참조.
    ANT_AMATEUR_BC_SPEC1_M0,   # idx 26
    ANT_AMATEUR_BC_SPEC1_M1,   # idx 27
    ANT_AMATEUR_BC_SPEC5_M0,   # idx 28
    ANT_AMATEUR_BC_SPEC5_M1,   # idx 29
    ANT_AMATEUR_BC_SPEC10_M0,   # idx 30
    ANT_AMATEUR_BC_SPEC10_M1,   # idx 31

    # ANT_EXPERT_IQL_PLOG,
    # ANT_AMATEUR_LINEAR_EXP,
    # ANT_AMATEUR_PLOG_EXP,
    # ANT_AMATEUR_LINEAR_PLOG,
    # ANT_AMATEUR_PLOG_PLOG,
    # ANT_AMATEUR_OPTIDICE,
    # ANT_AMATEUR_FAIRDICE,
    # ANT_AMATEUR_IQL_EXP,
    # ANT_AMATEUR_IQL_PLOG,

    # Filtered-BC baselines: top-K% by trajectory CD score, MLE on (s, a) pairs.
    # One BC policy per (cd_filter_rho, cd_filter_k_pct) — calibrates whether
    # the AETDICE / FairDICE machinery extracts more than simple imitation of the
    # highest-CD trajectories.
    # WALKER2D_AMATEUR_BC_CD_FILTER,
    # HALFCHEETAH_AMATEUR_BC_CD_FILTER,
    # CD × amp ∈ {1, 10, 100, 1000} — F-shaping signal-strength sweep on dim 0.
    # (The amp=1 cell reproduces the back-compat CD baseline.)
    # WALKER2D_AMATEUR_AETDICE_CD_AMP,
    # HALFCHEETAH_AMATEUR_AETDICE_CD_AMP,
    # IQL ESR baselines on the SAME (rho, amp, beta) grid as the AETDICE-CD
    # sweeps above. Calibrates whether DICE's gradient machinery beats the
    # state+racc Q + V + AWR baseline at extracting CD-optimal policies.
    # WALKER2D_AMATEUR_IQL_CD_AMP,
    # HALFCHEETAH_AMATEUR_IQL_CD_AMP,
    # Per-rho IQL HP sweeps. Targets regimes where default-HP IQL underperformed
    # in the IQL-CD comparison (walker2d ρ=0.9 crashed; halfcheetah ρ=0.1
    # missed the low-cost tail; halfcheetah ρ=0.9 close to dataset top-10%).
    # 8 + 8 + 4 = 20 jobs.
    # WALKER2D_AMATEUR_IQL_HP_RHO09,
    # HALFCHEETAH_AMATEUR_IQL_HP_RHO01,
    # HALFCHEETAH_AMATEUR_IQL_HP_RHO09,
    # CD-filtered AETDICE / IQL on top-10% trajectories. Same filter as BC; tests
    # whether the gradient-based learners on the filtered subset beat the same
    # learners on the full dataset, AND whether they beat filtered BC. 4 sweeps
    # × 9 jobs (3 train ρs × 3 filter ρs) = 36 jobs.
    # WALKER2D_AMATEUR_AETDICE_CD_FILTERED,
    # HALFCHEETAH_AMATEUR_AETDICE_CD_FILTERED,
    # WALKER2D_AMATEUR_IQL_CD_FILTERED,
    # HALFCHEETAH_AMATEUR_IQL_CD_FILTERED,
    # Safety-CD = 3-obj walker2d (speed, cost, alive) with threshold on the safety
    # dim. Encodes the log-balanced objective log(E[CD]) + log(E[plogT(R_safety)]).
    # WALKER2D_AMATEUR_AETDICE_SAFE_CD,
    # OTHER ENV
    # Halfcheetah is done!
    # HALFCHEETAH_EXPERT_LINEAR_EXP,
    # HALFCHEETAH_EXPERT_PLOG_EXP,
    # HALFCHEETAH_EXPERT_LINEAR_PLOG,
    # HALFCHEETAH_EXPERT_PLOG_PLOG,
    # HALFCHEETAH_EXPERT_OPTIDICE,
    # HALFCHEETAH_EXPERT_FAIRDICE,
    # HALFCHEETAH_EXPERT_IQL_EXP,
    # HALFCHEETAH_EXPERT_IQL_PLOG,
    # HALFCHEETAH_AMATEUR_LINEAR_EXP,
    # HALFCHEETAH_AMATEUR_PLOG_EXP,
    # HALFCHEETAH_AMATEUR_LINEAR_PLOG,
    # HALFCHEETAH_AMATEUR_PLOG_PLOG,
    # HALFCHEETAH_AMATEUR_OPTIDICE,
    # HALFCHEETAH_AMATEUR_FAIRDICE,
    # HALFCHEETAH_AMATEUR_IQL_EXP,
    # HALFCHEETAH_AMATEUR_IQL_PLOG,

    # ── Minimum CD analysis pair (walker2d_amateur) — DONE ──
    # WALKER2D_AMATEUR_CD_MIN_AET,
    # WALKER2D_AMATEUR_CD_MIN_BC,

    # Walker2d seed-extension (3 → 5) — done; commented for re-enable convenience.
    # WALKER2D_EXPERT_OPTIDICE_S45,
    # WALKER2D_EXPERT_FAIRDICE_S45,
    # WALKER2D_EXPERT_LINEAR_EXP_S45,
    # WALKER2D_EXPERT_PLOG_EXP_S45,
    # WALKER2D_EXPERT_IQL_EXP_S45,
    # Amateur seeds 4-5 — moved to the other server.
    # WALKER2D_AMATEUR_OPTIDICE_S45,
    # WALKER2D_AMATEUR_FAIRDICE_S45,
    # WALKER2D_AMATEUR_LINEAR_EXP_S45,
    # WALKER2D_AMATEUR_PLOG_EXP_S45,
    # WALKER2D_AMATEUR_IQL_EXP_S45,
    # Prior IQL HP search and original main walker2d sweeps — done; commented for re-enable convenience.
    # WALKER2D_EXPERT_IQL_EXP_E09_T3,
    # WALKER2D_EXPERT_IQL_EXP_E07_T10,
    # WALKER2D_EXPERT_LINEAR_EXP,
    # WALKER2D_EXPERT_PLOG_EXP,
    # WALKER2D_EXPERT_LINEAR_PLOG,
    # WALKER2D_EXPERT_PLOG_PLOG,
    # WALKER2D_EXPERT_OPTIDICE,
    # WALKER2D_EXPERT_FAIRDICE,
    # WALKER2D_EXPERT_IQL_EXP,
    # WALKER2D_EXPERT_IQL_PLOG,
    # WALKER2D_AMATEUR_LINEAR_EXP,
    # WALKER2D_AMATEUR_PLOG_EXP,
    # WALKER2D_AMATEUR_LINEAR_PLOG,
    # WALKER2D_AMATEUR_PLOG_PLOG,
    # WALKER2D_AMATEUR_OPTIDICE,
    # WALKER2D_AMATEUR_FAIRDICE,
    # WALKER2D_AMATEUR_IQL_EXP,
    # WALKER2D_AMATEUR_IQL_PLOG,
    # ── hopper ── under server 89
    # HOPPER_EXPERT_LINEAR_EXP,
    # HOPPER_EXPERT_PLOG_EXP,
    # HOPPER_EXPERT_LINEAR_PLOG,
    # HOPPER_EXPERT_PLOG_PLOG,
    # HOPPER_EXPERT_OPTIDICE,
    # HOPPER_EXPERT_FAIRDICE,
    # HOPPER_EXPERT_IQL_EXP,
    # HOPPER_EXPERT_IQL_PLOG,
    # HOPPER_AMATEUR_LINEAR_EXP,
    # HOPPER_AMATEUR_PLOG_EXP,
    # HOPPER_AMATEUR_LINEAR_PLOG,
    # HOPPER_AMATEUR_PLOG_PLOG,
    # HOPPER_AMATEUR_OPTIDICE,
    # HOPPER_AMATEUR_FAIRDICE,
    # HOPPER_AMATEUR_IQL_EXP,
    # HOPPER_AMATEUR_IQL_PLOG,

    # # ── swimmer ──
    # SWIMMER_EXPERT_LINEAR_EXP,
    # SWIMMER_EXPERT_PLOG_EXP,
    # SWIMMER_EXPERT_LINEAR_PLOG,
    # SWIMMER_EXPERT_PLOG_PLOG,
    # SWIMMER_EXPERT_OPTIDICE,
    # SWIMMER_EXPERT_FAIRDICE,
    # SWIMMER_EXPERT_IQL_EXP,
    # SWIMMER_EXPERT_IQL_PLOG,
    # SWIMMER_AMATEUR_LINEAR_EXP,
    # SWIMMER_AMATEUR_PLOG_EXP,
    # SWIMMER_AMATEUR_LINEAR_PLOG,
    # SWIMMER_AMATEUR_PLOG_PLOG,
    # SWIMMER_AMATEUR_OPTIDICE,
    # SWIMMER_AMATEUR_FAIRDICE,
    # SWIMMER_AMATEUR_IQL_EXP,
    # SWIMMER_AMATEUR_IQL_PLOG,
]
