"""Sweep dict definitions for MO-HalfCheetah-v2.

Each `<ENV>_<DATASET>_<METHOD>` constant is a sweep dict consumed by
`sweeps_active.SWEEPS` and expanded into per-job dicts by
`src.train.dispatcher.sweep_to_jobs`.
"""
from src.train.dispatch_settings import DIVERGENCE, MAX_SEQ_LEN
from src.train.sweep_builder import BETAS, PREFS_3, build_sweep

HALFCHEETAH_EXPERT_LINEAR_EXP   = build_sweep("half_cheetah", "expert",  "linear_exp",  learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3)

HALFCHEETAH_EXPERT_PLOG_EXP     = build_sweep("half_cheetah", "expert",  "plog_exp",    learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None])

HALFCHEETAH_EXPERT_LINEAR_PLOG  = build_sweep("half_cheetah", "expert",  "linear_plog", learner_type="aet",      g_type="linear",        f_type="piecewise_log", prefs=PREFS_3)

HALFCHEETAH_EXPERT_PLOG_PLOG    = build_sweep("half_cheetah", "expert",  "plog_plog",   learner_type="aet",      g_type="piecewise_log", f_type="piecewise_log", prefs=[None])

HALFCHEETAH_EXPERT_OPTIDICE     = build_sweep("half_cheetah", "expert",  "optidice",    learner_type="fairdice", g_type="linear",        f_type="linear",         prefs=PREFS_3)

HALFCHEETAH_EXPERT_FAIRDICE     = build_sweep("half_cheetah", "expert",  "fairdice",    learner_type="fairdice", g_type="piecewise_log", f_type="linear",         prefs=[None])


HALFCHEETAH_AMATEUR_LINEAR_EXP  = build_sweep("half_cheetah", "amateur", "linear_exp",  learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3)

HALFCHEETAH_AMATEUR_PLOG_EXP    = build_sweep("half_cheetah", "amateur", "plog_exp",    learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None])

HALFCHEETAH_AMATEUR_LINEAR_PLOG = build_sweep("half_cheetah", "amateur", "linear_plog", learner_type="aet",      g_type="linear",        f_type="piecewise_log", prefs=PREFS_3)

HALFCHEETAH_AMATEUR_PLOG_PLOG   = build_sweep("half_cheetah", "amateur", "plog_plog",   learner_type="aet",      g_type="piecewise_log", f_type="piecewise_log", prefs=[None])

HALFCHEETAH_AMATEUR_OPTIDICE    = build_sweep("half_cheetah", "amateur", "optidice",    learner_type="fairdice", g_type="linear",        f_type="linear",         prefs=PREFS_3)

HALFCHEETAH_AMATEUR_FAIRDICE    = build_sweep("half_cheetah", "amateur", "fairdice",    learner_type="fairdice", g_type="piecewise_log", f_type="linear",         prefs=[None])

# IQL ESR baselines for half_cheetah (state+racc; G-linear only — μ fixed to pref_weights). One sweep per F:
HALFCHEETAH_EXPERT_IQL_EXP     = build_sweep("half_cheetah", "expert",  "iql_exp",     learner_type="iql", g_type="linear", f_type="exponential_F",  prefs=PREFS_3)

HALFCHEETAH_EXPERT_IQL_PLOG    = build_sweep("half_cheetah", "expert",  "iql_plog",    learner_type="iql", g_type="linear", f_type="piecewise_log",  prefs=PREFS_3)

HALFCHEETAH_AMATEUR_IQL_EXP    = build_sweep("half_cheetah", "amateur", "iql_exp",     learner_type="iql", g_type="linear", f_type="exponential_F",  prefs=PREFS_3)

HALFCHEETAH_AMATEUR_IQL_PLOG   = build_sweep("half_cheetah", "amateur", "iql_plog",    learner_type="iql", g_type="linear", f_type="piecewise_log",  prefs=PREFS_3)



HALFCHEETAH_AMATEUR_AETDICE_CD_AMP = build_sweep(
    "half_cheetah", "amateur", "cd_amp_aet",
    learner_type="aet", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="half_cheetahcd", seeds=[1],
    cobbdouglas_rhos=[0.9, 0.5, 0.1],
    cobbdouglas_amps=[1, 10, 100, 1000],
    beta=["1e-5", "1e-6"],
)


# IQL ESR baseline for halfcheetah CD: SAME (rho, amp) grid as the AETDICE-CD
# sweep above, learner_type=iql. β is a placeholder (IQL ignores it; see
# walker2d IQL-CD comment) — single arbitrary value to avoid duplicates.
HALFCHEETAH_AMATEUR_IQL_CD_AMP = build_sweep(
    "half_cheetah", "amateur", "cd_amp_iql",
    learner_type="iql", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="half_cheetahcd", seeds=[1],
    cobbdouglas_rhos=[0.9, 0.5, 0.1],
    cobbdouglas_amps=[1, 10, 100, 1000],
    beta=["1e-5"],   # placeholder — IQL ignores config.beta
)


# halfcheetah ρ=0.1: higher expectile + temperature to find rare low-cost tail.
# amp=100 was the best at ρ=0.1 (CD=1.5).
HALFCHEETAH_AMATEUR_IQL_HP_RHO01 = build_sweep(
    "half_cheetah", "amateur", "iql_hp_rho0.1",
    learner_type="iql", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="half_cheetahcd", seeds=[1],
    cobbdouglas_rhos=[0.1],
    cobbdouglas_amps=[100, 1000],
    beta=["1e-5"],
    temperatures=[10.0, 30.0],     # sharpen AWR → focus on rare-extreme advantages
    expectiles=[0.9, 0.95],         # V optimistically targets high-Q tail
)


# halfcheetah ρ=0.9: small expectile bump to close 3-pt gap to dataset top-10%.
# amp=1 was the best (CD=159.7); the regime is saturated so amp barely matters.
HALFCHEETAH_AMATEUR_IQL_HP_RHO09 = build_sweep(
    "half_cheetah", "amateur", "iql_hp_rho0.9",
    learner_type="iql", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="half_cheetahcd", seeds=[1],
    cobbdouglas_rhos=[0.9],
    cobbdouglas_amps=[1],
    beta=["1e-5"],
    temperatures=[3.0, 10.0],
    expectiles=[0.85, 0.9],
)

HALFCHEETAH_AMATEUR_AETDICE_CD_FILTERED = build_sweep(
    "half_cheetah", "amateur", "cd_filtered_aet",
    learner_type="aet", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="half_cheetahcd", seeds=[1],
    cobbdouglas_rhos=[0.1, 0.5, 0.9],
    cobbdouglas_amps=[1000],   # best amp from unfiltered halfcheetah AET-CD at low rho
    beta=["1e-5"],
    cd_filter_rhos=[0.1, 0.5, 0.9],
    cd_filter_k_pcts=[0.1],
)

HALFCHEETAH_AMATEUR_IQL_CD_FILTERED = build_sweep(
    "half_cheetah", "amateur", "cd_filtered_iql",
    learner_type="iql", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="half_cheetahcd", seeds=[1],
    cobbdouglas_rhos=[0.1, 0.5, 0.9],
    cobbdouglas_amps=[100],
    beta=["1e-5"],
    cd_filter_rhos=[0.1, 0.5, 0.9],
    cd_filter_k_pcts=[0.1],
)


# Filtered-BC baseline for halfcheetah amateur CD. State-only MoG MLE on the
# top-K% trajectories ranked by per-trajectory CD with cd_filter_rho. β is
# unused by BC's loss but the runner threads it; pick one arbitrary value
# (1e-5) to get a single run per (cd_filter_rho, cd_filter_k_pct).
HALFCHEETAH_AMATEUR_BC_CD_FILTER = build_sweep(
    "half_cheetah", "amateur", "bc_cdfilter",
    learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], reward_reshape="half_cheetahcd", seeds=[1],
    cd_filter_rhos=[0.1, 0.5, 0.9],
    cd_filter_k_pcts=[0.1],
    beta=["1e-4"],   # unused by BC loss; one arbitrary value to avoid duplicate runs
)
