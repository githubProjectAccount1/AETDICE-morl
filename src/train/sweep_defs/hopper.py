"""Sweep dict definitions for MO-Hopper-v2.

Each `<ENV>_<DATASET>_<METHOD>` constant is a sweep dict consumed by
`sweeps_active.SWEEPS` and expanded into per-job dicts by
`src.train.dispatcher.sweep_to_jobs`.
"""
from src.train.dispatch_settings import DIVERGENCE, MAX_SEQ_LEN
from src.train.sweep_builder import BETAS, PREFS_3, build_sweep

HOPPER_EXPERT_LINEAR_EXP   = build_sweep("hopper", "expert",  "linear_exp",  learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3)

HOPPER_EXPERT_PLOG_EXP     = build_sweep("hopper", "expert",  "plog_exp",    learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None])

HOPPER_EXPERT_LINEAR_PLOG  = build_sweep("hopper", "expert",  "linear_plog", learner_type="aet",      g_type="linear",        f_type="piecewise_log", prefs=PREFS_3)

HOPPER_EXPERT_PLOG_PLOG    = build_sweep("hopper", "expert",  "plog_plog",   learner_type="aet",      g_type="piecewise_log", f_type="piecewise_log", prefs=[None])

HOPPER_EXPERT_OPTIDICE     = build_sweep("hopper", "expert",  "optidice",    learner_type="fairdice", g_type="linear",        f_type="linear",         prefs=PREFS_3)

HOPPER_EXPERT_FAIRDICE     = build_sweep("hopper", "expert",  "fairdice",    learner_type="fairdice", g_type="piecewise_log", f_type="linear",         prefs=[None])


HOPPER_AMATEUR_LINEAR_EXP  = build_sweep("hopper", "amateur", "linear_exp",  learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3)

HOPPER_AMATEUR_PLOG_EXP    = build_sweep("hopper", "amateur", "plog_exp",    learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None])

HOPPER_AMATEUR_LINEAR_PLOG = build_sweep("hopper", "amateur", "linear_plog", learner_type="aet",      g_type="linear",        f_type="piecewise_log", prefs=PREFS_3)

HOPPER_AMATEUR_PLOG_PLOG   = build_sweep("hopper", "amateur", "plog_plog",   learner_type="aet",      g_type="piecewise_log", f_type="piecewise_log", prefs=[None])

HOPPER_AMATEUR_OPTIDICE    = build_sweep("hopper", "amateur", "optidice",    learner_type="fairdice", g_type="linear",        f_type="linear",         prefs=PREFS_3)

HOPPER_AMATEUR_FAIRDICE    = build_sweep("hopper", "amateur", "fairdice",    learner_type="fairdice", g_type="piecewise_log", f_type="linear",         prefs=[None])

# IQL ESR baselines for hopper (state+racc; G-linear only — μ fixed to pref_weights). One sweep per F:
HOPPER_EXPERT_IQL_EXP      = build_sweep("hopper", "expert",  "iql_exp",     learner_type="iql", g_type="linear", f_type="exponential_F",  prefs=PREFS_3)

HOPPER_EXPERT_IQL_PLOG     = build_sweep("hopper", "expert",  "iql_plog",    learner_type="iql", g_type="linear", f_type="piecewise_log",  prefs=PREFS_3)

HOPPER_AMATEUR_IQL_EXP     = build_sweep("hopper", "amateur", "iql_exp",     learner_type="iql", g_type="linear", f_type="exponential_F",  prefs=PREFS_3)

HOPPER_AMATEUR_IQL_PLOG    = build_sweep("hopper", "amateur", "iql_plog",    learner_type="iql", g_type="linear", f_type="piecewise_log",  prefs=PREFS_3)
