"""Sweep dict definitions for MO-Walker2d-v2.

Each `<ENV>_<DATASET>_<METHOD>` constant is a sweep dict consumed by
`sweeps_active.SWEEPS` and expanded into per-job dicts by
`src.train.dispatcher.sweep_to_jobs`.
"""
from src.train.dispatch_settings import DIVERGENCE, MAX_SEQ_LEN
from src.train.sweep_builder import BETAS, PREFS_3, build_sweep


# 6 methods × 2 datasets = 12 sweeps. Mirrors the ant_* convention exactly.
WALKER2D_EXPERT_LINEAR_EXP   = build_sweep("walker2d", "expert",  "linear_exp",  learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3)

WALKER2D_EXPERT_PLOG_EXP     = build_sweep("walker2d", "expert",  "plog_exp",    learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None])

WALKER2D_EXPERT_LINEAR_PLOG  = build_sweep("walker2d", "expert",  "linear_plog", learner_type="aet",      g_type="linear",        f_type="piecewise_log", prefs=PREFS_3)

WALKER2D_EXPERT_PLOG_PLOG    = build_sweep("walker2d", "expert",  "plog_plog",   learner_type="aet",      g_type="piecewise_log", f_type="piecewise_log", prefs=[None])

WALKER2D_EXPERT_OPTIDICE     = build_sweep("walker2d", "expert",  "optidice",    learner_type="fairdice", g_type="linear",        f_type="linear",         prefs=PREFS_3)

WALKER2D_EXPERT_FAIRDICE     = build_sweep("walker2d", "expert",  "fairdice",    learner_type="fairdice", g_type="piecewise_log", f_type="linear",         prefs=[None])


WALKER2D_AMATEUR_LINEAR_EXP  = build_sweep("walker2d", "amateur", "linear_exp",  learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3)

WALKER2D_AMATEUR_PLOG_EXP    = build_sweep("walker2d", "amateur", "plog_exp",    learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None])

WALKER2D_AMATEUR_LINEAR_PLOG = build_sweep("walker2d", "amateur", "linear_plog", learner_type="aet",      g_type="linear",        f_type="piecewise_log", prefs=PREFS_3)

WALKER2D_AMATEUR_PLOG_PLOG   = build_sweep("walker2d", "amateur", "plog_plog",   learner_type="aet",      g_type="piecewise_log", f_type="piecewise_log", prefs=[None])

WALKER2D_AMATEUR_OPTIDICE    = build_sweep("walker2d", "amateur", "optidice",    learner_type="fairdice", g_type="linear",        f_type="linear",         prefs=PREFS_3)

WALKER2D_AMATEUR_FAIRDICE    = build_sweep("walker2d", "amateur", "fairdice",    learner_type="fairdice", g_type="piecewise_log", f_type="linear",         prefs=[None])

# IQL ESR baselines for walker2d (state+racc; G-linear only — μ fixed to pref_weights). One sweep per F:
WALKER2D_EXPERT_IQL_EXP      = build_sweep("walker2d", "expert",  "iql_exp",     learner_type="iql", g_type="linear", f_type="exponential_F",  prefs=PREFS_3)

# IQL HP search (walker2d_expert × iql_exp): only the prefs that underperformed
# linear-exp at default HP (0.7, 3.0). β is unused by IQL — pin to 1e-4 (best
# default-HP iql_exp β at extreme prefs). Each non-default (temp, expectile)
# tuple lands in its own save_dir via run_naming's `_temp{T}_exp{E}` suffix.
PREFS_HP_SEARCH = [[0.5, 0.5], [0.0, 1.0]]

WALKER2D_EXPERT_IQL_EXP_E09_T3 = build_sweep(
    "walker2d", "expert", "iql_exp_e0.9_t3",
    learner_type="iql", g_type="linear", f_type="exponential_F",
    prefs=PREFS_HP_SEARCH, beta=["1e-4"],
    expectiles=[0.9], temperatures=[3.0],
)

WALKER2D_EXPERT_IQL_EXP_E07_T10 = build_sweep(
    "walker2d", "expert", "iql_exp_e0.7_t10",
    learner_type="iql", g_type="linear", f_type="exponential_F",
    prefs=PREFS_HP_SEARCH, beta=["1e-4"],
    expectiles=[0.7], temperatures=[10.0],
)

# IQL pinned to single β (β is unused inside IQL — placeholder only). Other
# methods keep the 3-β sweep since β IS load-bearing for AET / FairDICE learners.
WALKER2D_EXPERT_OPTIDICE_S45    = build_sweep("walker2d", "expert",  "optidice",   learner_type="fairdice", g_type="linear",        f_type="linear",        prefs=PREFS_3, seeds=[4, 5])

WALKER2D_EXPERT_FAIRDICE_S45    = build_sweep("walker2d", "expert",  "fairdice",   learner_type="fairdice", g_type="piecewise_log", f_type="linear",        prefs=[None],   seeds=[4, 5])

WALKER2D_EXPERT_LINEAR_EXP_S45  = build_sweep("walker2d", "expert",  "linear_exp", learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3, seeds=[4, 5])

WALKER2D_EXPERT_PLOG_EXP_S45    = build_sweep("walker2d", "expert",  "plog_exp",   learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None],   seeds=[4, 5])

WALKER2D_EXPERT_IQL_EXP_S45     = build_sweep("walker2d", "expert",  "iql_exp",    learner_type="iql",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3, seeds=[4, 5], beta=["1e-4"])


WALKER2D_AMATEUR_OPTIDICE_S45   = build_sweep("walker2d", "amateur", "optidice",   learner_type="fairdice", g_type="linear",        f_type="linear",        prefs=PREFS_3, seeds=[4, 5])

WALKER2D_AMATEUR_FAIRDICE_S45   = build_sweep("walker2d", "amateur", "fairdice",   learner_type="fairdice", g_type="piecewise_log", f_type="linear",        prefs=[None],   seeds=[4, 5])

WALKER2D_AMATEUR_LINEAR_EXP_S45 = build_sweep("walker2d", "amateur", "linear_exp", learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3, seeds=[4, 5])

WALKER2D_AMATEUR_PLOG_EXP_S45   = build_sweep("walker2d", "amateur", "plog_exp",   learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None],   seeds=[4, 5])

WALKER2D_AMATEUR_IQL_EXP_S45    = build_sweep("walker2d", "amateur", "iql_exp",    learner_type="iql",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3, seeds=[4, 5], beta=["1e-4"])

WALKER2D_EXPERT_IQL_PLOG     = build_sweep("walker2d", "expert",  "iql_plog",    learner_type="iql", g_type="linear", f_type="piecewise_log",  prefs=PREFS_3)


WALKER2D_AMATEUR_IQL_EXP     = build_sweep("walker2d", "amateur", "iql_exp",     learner_type="iql", g_type="linear", f_type="exponential_F",  prefs=PREFS_3)

WALKER2D_AMATEUR_IQL_PLOG    = build_sweep("walker2d", "amateur", "iql_plog",    learner_type="iql", g_type="linear", f_type="piecewise_log",  prefs=PREFS_3)


# Cobb-Douglas: AET learner with F=cobbdouglas on the (forward_velocity, ‖a‖²) reshape.
# CD × amp sweep — amp ∈ {1, 10, 100, 1000} amplifies dim 0 of CobbDouglas only.
# Under G=linear this is mathematically equivalent to dividing β by amp; gives a
# clean read on whether stronger F-shaping signal pushes the policy further from
# the dataset toward the (low-cost / high-reward) tail. amp=1 reproduces the
# back-compat baseline.
WALKER2D_AMATEUR_AETDICE_CD_AMP = build_sweep(
    "walker2d", "amateur", "cd_amp_aet",
    learner_type="aet", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1],
    cobbdouglas_rhos=[0.9, 0.5, 0.3, 0.2, 0.1],
    cobbdouglas_amps=[1, 10, 100, 1000],
    beta=["1e-4", "1e-5", "1e-6"],
)


# ── Exp 0 (docs/design/design/0001-exp0-walking-rho.md) ──────────────────────
# Walking-quality ρ sweep, Walker2d amateur, β=1e-4, amp=1, seed{1,2,3}, clip-off.
# ρ∈{1.0,0.5,0.0} × {cd-only, cd+safety}. Renders → walking-vs-collapse figure.
WALKER2D_AMATEUR_EXP0_CD = build_sweep(
    "walker2d", "amateur", "exp0_cd",
    learner_type="aet", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1, 2, 3],
    cobbdouglas_rhos=[1.0, 0.5, 0.0], cobbdouglas_amps=[1], beta=["1e-4"],
)

WALKER2D_AMATEUR_EXP0_SAFE_CD = build_sweep(
    "walker2d", "amateur", "exp0_safe_cd",
    learner_type="aet", g_type="piecewise_log", f_type="cobbdouglas",
    prefs=[None], reward_reshape="walker2d_cd_safe", seeds=[1, 2, 3],
    cobbdouglas_rhos=[1.0, 0.5, 0.0], reward_thresholds=[250],
    cobbdouglas_amps=[1], beta=["1e-4"],
)


# IQL ESR baseline for walker2d CD: SAME (rho, amp) grid as the AETDICE-CD sweep
# above, but learner_type=iql (state+racc; Q + V + AWR; no DICE dual; uses
# config.expectile / temperature / awr_exp_clip / tau, NOT config.beta).
# IQL requires utility_G_type=linear with fixed pref_weights — matches AETDICE's
# [[1.0, 0.0]]. β is a placeholder (the runner threads it for uniform CLI), so
# we sweep ONE arbitrary value to avoid 3× duplicate training.
WALKER2D_AMATEUR_IQL_CD_AMP = build_sweep(
    "walker2d", "amateur", "cd_amp_iql",
    learner_type="iql", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1],
    cobbdouglas_rhos=[0.9, 0.5, 0.3, 0.2, 0.1],
    cobbdouglas_amps=[1, 10, 100, 1000],
    beta=["1e-5"],   # placeholder — IQL ignores config.beta
)


# Walker2d CD + safety: 3-D reshape (speed, cost, safety) — fully decoupled.
# CobbDouglasUtility runs CD over (R0=speed, R1=cost) for output dim 0, and the
# piecewise_log threshold over R2=safety for output dim 1. F̃ output stays 2-D,
# so under G=piecewise_log the population objective is
#     log(E[CD(R_speed, R_cost)]) + log(E[plogT(R_safety − T)])
# Drop-and-die at low ρ is punished on both channels (it kills cumulative speed
# AND cumulative safety), unlike bare CD on (speed, cost) where minimizing R1
# by terminating early was a valid optimum.
#
# Per-step safety is a normalized symmetric quadratic in (height, angle):
#     r_safety = (−2·h_norm² + 1) + (−2·a_norm² + 1)
# where h_norm = (h − 1.4)/0.6 and a_norm = a/1.0. Each dim contributes +1 at
# center, 0 at |x_norm|=1/√2, −1 at its env bound. Per-step safety ∈ [-2, +2]
# in-bounds (max +2 at perfect balance, min −2 at both env bounds).
#
# Thresholds operate on accumulated NORMALIZED R2 (raw per-step ∈ [-2, +2] gets
# min-max normalized to [0, 1] per step; with rsf=1 and max_seq_len=500 the
# cumulative range is [0, 500]). [100, 250, 400] = lax (~20% of max) / moderate
# (~50%) / strict (~80%). Recalibrate after seeing empirical normalized
# R_safety percentiles if these don't differentiate well across rho.
WALKER2D_AMATEUR_AETDICE_SAFE_CD  = build_sweep(
    "walker2d", "amateur", "safe_cd_aet",
    learner_type="aet", g_type="piecewise_log", f_type="cobbdouglas",
    prefs=[None], reward_reshape="walker2d_cd_safe", seeds=[1],
    cobbdouglas_rhos=[0.9, 0.5, 0.3, 0.2, 0.1],
    reward_thresholds=[100, 250, 400],          # cum-normalized R_safety floor (dim 2 only)
    cobbdouglas_amps=[1, 10, 100, 1000],
    beta=["1e-4", "1e-5", "1e-6"],
)

# Filtered-BC baseline for walker2d amateur CD. Same idea as halfcheetah's
# bc_cdfilter — one MLE-trained policy per (cd_filter_rho, cd_filter_k_pct)
# combo. β is unused by BC's loss but the runner threads it; pick one arbitrary
# value to keep one run per knob combo.
WALKER2D_AMATEUR_BC_CD_FILTER = build_sweep(
    "walker2d", "amateur", "bc_cdfilter",
    learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1],
    cd_filter_rhos=[0.1, 0.5, 0.9],
    cd_filter_k_pcts=[0.1],
    beta=["1e-4"],
)


# ── Minimum CD analysis sweep — for data_analysis/cd_aet_vs_bc.ipynb ──
# Three ρ values span the trade-off corners:
#   ρ=0.1 → cost-favoring  (F ≈ R0^0.1 / (1+R1)^0.9 — energy minimization dominates)
#   ρ=0.5 → symmetric      (F = sqrt(R0) / sqrt(1+R1))
#   ρ=1.0 → pure reward    (F = R0 — R1 ignored, degenerates to fwd-velocity max)
# Each ρ paired with same-ρ BC (filtered top-10% by CD score under that ρ) so
# clouds are directly comparable per-ρ.
#   AET: 3 ρ × 1 amp × 3 β × 3 seeds = 27 jobs (~3h each on 1 GPU)
#   BC : 3 ρ × 1 k_pct × 3 seeds     = 9 jobs (~40min each — BC is 200K steps)
WALKER2D_AMATEUR_CD_MIN_AET = build_sweep(
    "walker2d", "amateur", "cd_min_aet",
    learner_type="aet", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1, 2, 3],
    cobbdouglas_rhos=[0.1, 0.5, 1.0],
    cobbdouglas_amps=[1],
    beta=["1e-4", "1e-5", "1e-6"],
)

WALKER2D_AMATEUR_CD_MIN_BC = build_sweep(
    "walker2d", "amateur", "cd_min_bc",
    learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1, 2, 3],
    cd_filter_rhos=[0.1, 0.5, 1.0],
    cd_filter_k_pcts=[0.1],
    beta=["1e-4"],
)


# IQL has different optimal hyperparameters per regime. With defaults
# (expectile=0.7, temperature=3.0):
#   - walker2d ρ=0.9 crashed (CD=55 vs AETDICE 92, BC 107) — temp=3 too peaky for
#     reward-dominated regime + walker2d's strict `done` → unstable. Try lower temp.
#   - halfcheetah ρ=0.1 underperformed (CD=1.5 vs AETDICE 12.8) — V's 70th
#     percentile expectile misses the rare extreme-low-cost trajs that AETDICE
#     finds via F-shaping. Try higher expectile + higher temperature.
#   - halfcheetah ρ=0.9 close but +3pt below dataset top-10% (159.7 vs 163).
#     Try slight expectile bump.
# Each cell varies a small (temp × expectile) grid at the best amp from the
# default-HP sweep. β stays placeholder. Dirs distinguish via _temp/_exp tokens.

# walker2d ρ=0.9: lower temperature fights mode collapse. amp=1000 was the best
# from the default-HP sweep at ρ=0.9 (CD=55).
WALKER2D_AMATEUR_IQL_HP_RHO09 = build_sweep(
    "walker2d", "amateur", "iql_hp_rho0.9",
    learner_type="iql", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1],
    cobbdouglas_rhos=[0.9],
    cobbdouglas_amps=[1, 1000],   # best two amps from prior IQL sweep at ρ=0.9
    beta=["1e-5"],
    temperatures=[0.3, 1.0],       # vs default 3.0 — softer AWR
    expectiles=[0.5, 0.7],         # vs default 0.7 — less optimistic V
)

# Same idea as filtered-BC, but applied to the gradient-based learners. The
# buffer keeps only the top-10% trajectories by CD score under cd_filter_rho;
# AETDICE / IQL then trains on this subset. Tests whether DICE's gradient
# machinery on filtered data beats either:
#   - DICE on full data (its current best)
#   - filtered BC on the same subset (the imitation lower bound)
#
# Diagonal (cd_filter_rho == cobbdouglas_rho) — most natural pairing: filter the
# data by best-CD-at-rho, train at the same rho. Cartesian over 3 rhos = 3 jobs
# per (env × learner × beta) combo. Per-rho amp is the best from the unfiltered
# CD sweep. Filter k=10% matches the BC pattern.

WALKER2D_AMATEUR_AETDICE_CD_FILTERED = build_sweep(
    "walker2d", "amateur", "cd_filtered_aet",
    learner_type="aet", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1],
    cobbdouglas_rhos=[0.1, 0.5, 0.9],
    cobbdouglas_amps=[100],   # best amp from unfiltered AET-CD at most rhos
    beta=["1e-5"],
    cd_filter_rhos=[0.1, 0.5, 0.9],   # 3×3=9 combos; user can drop off-diagonals if too many
    cd_filter_k_pcts=[0.1],
)

WALKER2D_AMATEUR_IQL_CD_FILTERED = build_sweep(
    "walker2d", "amateur", "cd_filtered_iql",
    learner_type="iql", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="walker2dcd", seeds=[1],
    cobbdouglas_rhos=[0.1, 0.5, 0.9],
    cobbdouglas_amps=[100],
    beta=["1e-5"],   # placeholder — IQL ignores config.beta
    cd_filter_rhos=[0.1, 0.5, 0.9],
    cd_filter_k_pcts=[0.1],
)
