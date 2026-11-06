"""Sweep dict definitions for MO-PointMaze-v10 (Reviewer 2, many-objective scaling).

10-objective toy: AET-DICE (G-plog × F-exp) should learn a policy that visits all 10
objectives. reward_dim=10 comes from env.obj_dim (build_env); the sweep must pass a
10-length reward_scaling_factor and max_seq_len=200 (env horizon).
"""
from src.train.sweep_builder import build_sweep

# AET-DICE, G-plog × F-exp, χ², β1e-6. prefs=[None] (racc-conditioned, mu learnable).
MO_POINTMAZE_EXPERT_AET = build_sweep(
    "mo_pointmaze", "expert", "plog_exp", learner_type="aet",
    g_type="piecewise_log", f_type="exponential_F", prefs=[None],
    seeds=[1, 2, 3, 4, 5], beta=["1e-6"],
    reward_scaling_factors=[[1.0] * 10], max_seq_lens=[200])
