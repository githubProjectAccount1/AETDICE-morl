"""Sweep dict factory + canonical β / preference vectors.

`_build_sweep` is the single source of truth for sweep-dict shape. Adding a
new axis means: (1) add a kwarg here, (2) add a row in dispatcher._AXIS_KEYS
and _AXIS_DEFAULTS, (3) thread it into job_runner.run_job's argparse forward.

Naming convention used by all sweep dicts (set by `_build_sweep`):
    name = "{env}_{dataset}_{method}"
where method encodes both networks and utilities, e.g.:
    linear_exp : --learner_type aet      G=linear        F=exponential_F  (3 prefs)
    plog_exp   : --learner_type aet      G=piecewise_log F=exponential_F  (mu learnable)
    optidice   : --learner_type fairdice G=linear        F=linear         (3 prefs, no F shaping)
    fairdice   : --learner_type fairdice G=piecewise_log F=linear         (mu learnable, no F shaping)
    iql_*      : --learner_type iql      G=linear        F={linear|exp|plog}  (3 prefs; ESR baseline, μ fixed)

OptiDICE / FairDICE baselines use `FairDICE_learner` (state-only critic + policy);
they differ from each other only in --utility_G_type. IQL is an ESR baseline
(state+racc; Q + V + AWR; no DICE dual) — G must be linear.
"""
from src.train.dispatch_settings import DIVERGENCE, MAX_SEQ_LEN

# Canonical β grid used by virtually every sweep entry.
BETAS = ["1e-4", "1e-5", "1e-6"]

# Canonical 3-pref triplet for 2-objective envs.
PREFS_3 = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]

def build_sweep(env, dataset, method, *, learner_type, g_type, f_type, prefs,
                reward_reshape: str = "none",
                reward_scaling_factors=None,
                # musf 는 오래 전역 상수(1e4)로 못박혀 있었다. β 와 함께 DICE dual 의
                # 수치 스케일을 정하는 축이라 실제로는 스윕 대상이다 (_AXIS_KEYS 에는
                # 이미 항목이 있었고 값만 [1e4] 로 고정돼 있었다).
                mu_scaling_factors=(1e4,),
                seeds=(1, 2, 3),
                beta=BETAS,
                bc_conditionings=("state",),
                # 조건부 racc 입력의 형태 (기본 ftilde = 기존 동작). cli.py 주석 참조.
                # 학습용 exponential_F 의 스케일 S. 평가 지표 AET 는 S=50 고정이므로,
                # 50 이 아닌 값은 **tempered surrogate** 로 학습한다는 뜻이다.
                exp_f_scales=(50.0,),
                racc_encodings=("ftilde",),
                z_keeplists=(None,),
                cd_filter_kinds=("cd",),
                cobbdouglas_rhos=(0.5,),
                max_seq_lens=(MAX_SEQ_LEN,),
                reward_thresholds=(None,),
                cobbdouglas_amps=(1.0,),
                cd_filter_rhos=(None,),
                cd_filter_k_pcts=(None,),
                divergences=(DIVERGENCE,),
                w_maxes=(float("inf"),),
                nu_output_transforms=("none",),
                temperatures=(None,),
                expectiles=(None,),
                beta_schedules=("constant",),
                beta_inits=(None,),
                beta_finals=(None,),
                beta_warmup_steps_list=(0,),
                standardize_e_list=(False,),
                traj_filter_paths=(None,),
                transition_filter_paths=(None,),
                global_mixtures=(False,),
                extractions=("original",),
                wbc_weights=("raw",),
                traj_mixtures=(False,)):
    """Construct a sweep dict from method-level overrides; the rest is pinned to
    project-wide defaults (entropy=uniform, 3 seeds, rsf=[1,1], musf=1e4, BETAS).

    `method` is the suffix appended to the human-readable sweep name (and matches
    the variable-name suffix used by callers, by convention).

    `reward_reshape` selects an optional buffer/eval reshape ('none' default,
    'antv5' for the 5-obj MO-Ant decomposition). `reward_scaling_factors`
    defaults to a 1-vector of length matching the reshape's reward_dim.

    `cobbdouglas_rhos` is a list of rho values to sweep when f_type=='cobbdouglas'.
    Default [0.5] — the symmetric efficiency. Ignored at run time when f_type
    != 'cobbdouglas' (CLI flag is dropped).

    `max_seq_lens` is a list of episode-length caps. Default (MAX_SEQ_LEN,).
    Sets BOTH the eval rollout cap AND the sinusoidal time-embedding T_max in
    policy/critic — but does NOT truncate the training dataset (add truncation
    in buffer.load_d4morl_dataset for that).

    `reward_thresholds` is a list of T values to sweep when f_type ==
    'piecewise_log_threshold'. Default (None,) — placeholder for non-thresholded
    F types. When f_type IS piecewise_log_threshold, ALL entries must be non-None.

    보려면 modified 를 써야 한다. 이 축이 없으면 sweep 은 항상 original 로 돈다.

    `cobbdouglas_amps` is a list of dim-0 amplifiers for cobbdouglas. Default
    (1.0,). When f_type == 'cobbdouglas', a value != 1.0 emits an `_amp<value>`
    token in the run name. Under G=linear this knob ≡ dividing β by amp;
    under G=piecewise_log it has no gradient effect (log absorbs it).
    """
    if reward_scaling_factors is None:
        from src.data.reward_transform import WALKER2D_CD_SAFE_RESHAPES
        if reward_reshape in WALKER2D_CD_SAFE_RESHAPES:
            # 3-D layout: (speed, cost, safety) — speed/safety stay decoupled
            # so CobbDouglasUtility runs CD over (speed, cost) and threshold-log
            # over safety alone.
            reward_scaling_factors = [[1.0, 1.0, 1.0]]
        else:
            # 2-D reshapes (none / walker2dcd / half_cheetahcd).
            reward_scaling_factors = [[1.0, 1.0]]
    return {
        "name":                   f"{env}_{dataset}_{method}",
        "dataset":                dataset,
        "entropy_levels":         ["uniform"],
        "seeds":                  list(seeds),
        "envs":                   [env],
        "learner_types":          [learner_type],
        "utility_G_types":        [g_type],
        "utility_F_types":        [f_type],
        "power_ps":               [None],
        "exp_f_scales":           list(exp_f_scales),
        "pref_weights_list":      prefs,
        "reward_scaling_factors": reward_scaling_factors,
        "mu_scaling_factor":      list(mu_scaling_factors),
        "reward_reshapes":        [reward_reshape],
        "betas":                  beta,
        "bc_conditionings":       list(bc_conditionings),
        "racc_encodings":         list(racc_encodings),
        "z_keeplists":            list(z_keeplists),
        "cd_filter_kinds":        list(cd_filter_kinds),
        "cobbdouglas_rhos":       list(cobbdouglas_rhos),
        "max_seq_lens":           list(max_seq_lens),
        "reward_thresholds":      list(reward_thresholds),
        "cobbdouglas_amps":       list(cobbdouglas_amps),
        "cd_filter_rhos":         list(cd_filter_rhos),
        "cd_filter_k_pcts":       list(cd_filter_k_pcts),
        "divergences":            list(divergences),
        "w_maxes":                list(w_maxes),
        "nu_output_transforms":   list(nu_output_transforms),
        "temperatures":           list(temperatures),
        "expectiles":             list(expectiles),
        "beta_schedules":         list(beta_schedules),
        "beta_inits":             list(beta_inits),
        "beta_finals":            list(beta_finals),
        "beta_warmup_steps_list": list(beta_warmup_steps_list),
        "standardize_e_list":     list(standardize_e_list),
        "traj_filter_paths":      list(traj_filter_paths),
        "transition_filter_paths": list(transition_filter_paths),
        "global_mixtures":        list(global_mixtures),
        "extractions":            list(extractions),
        "wbc_weights":            list(wbc_weights),
        "traj_mixtures":          list(traj_mixtures),
    }
