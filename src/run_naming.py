"""Shared run-naming logic.

main.py (which creates save_dirs) and run_AET_d4morl.py
(which auto-resumes crashed jobs by finding matching save_dirs) must agree on
the name format. Kept pure-python — no jax/numpy imports — so the sweep runner
can import it without triggering cuda/jax startup.

Layout of a save_dir:
    <save_path>/<subfolder>/<basename>
where:
    <save_path> — base directory (typically env+dataset level, e.g. "./results_transformed/ant_expert")
    <subfolder> — scaling knobs only, emitted by make_subfolder_name(). Groups runs that
                  share the same reward/mu scaling so they sit together on disk.
    <basename> — run-differentiating knobs (G, F, beta, seed, tag), emitted by make_run_name().
                 Does NOT include scaling knobs.

Splitting the scaling knobs out of the basename lets sweep tooling construct the full
save_dir from hyperparameters alone, without parsing config.json.
"""


def fmt_threshold(th) -> str:
    """문턱 토큰: 하나면 '490', 여러 개 (safe-CD 채널마다, 2026-11-05) 면 '490-166'. 디렉터리 이름·버퍼 캐시 키 공용."""
    if isinstance(th, (list, tuple)):
        return "-".join(f"{float(t):g}" for t in th)
    return f"{float(th):g}"


def fmt_sci(x) -> str:
    """Scientific notation without leading zeros in the exponent: 1e-4 not 1e-04.
    Accepts either str ("1e-4") or float (1e-4)."""
    s = f"{float(x):.0e}"
    m, e = s.split("e")
    return f"{m}e{int(e)}"


def fmt_to(values) -> str:
    """Join numeric values with a 'to' separator, preserving default float repr
    (so 1.0 stays '1.0', not stripped to '1'). Example: [1.0, 0.0] -> '1.0to0.0'.

    High-dim reshapes (e.g. antpref100, reward_dim=100) would blow past the OS
    filename limit if every value were listed, so vectors longer than 4 are
    summarized: all-equal -> '<v>x<n>' (e.g. '1.0x100'); otherwise '<v0>..<n>d'."""
    values = list(values)
    if len(values) > 4:
        if all(v == values[0] for v in values):
            return f"{values[0]}x{len(values)}"
        return f"{values[0]}..{len(values)}d"
    return "to".join(f"{v}" for v in values)


def wfilter_token(traj_filter_path) -> str:
    """Strategy id from a w-filter keep-list path (Exp3-03 coverage ablation):
    '/x/keep_droptop_5pct.npy' -> 'droptop_5pct'. Empty when no filter (baseline
    naming unchanged). Pure-string (no os import) to keep this module jax/numpy-free."""
    if not traj_filter_path:
        return ""
    stem = str(traj_filter_path).replace("\\", "/").rsplit("/", 1)[-1]  # basename
    if stem.endswith(".npy"):
        stem = stem[:-4]
    if stem.startswith("keep_"):
        stem = stem[len("keep_"):]
    return stem


def make_run_name(
    utility_G_type,
    utility_F_type,
    power_p,
    pref_weights,
    beta,
    seed,
    tag,
    learner_type="aet",
    bc_conditioning=None,
    cobbdouglas_rho=None,
    max_seq_len=None,
    reward_threshold=None,
    cobbdouglas_amp=None,
    cobbdouglas_form=None,
    cobbdouglas_safety=None,
    cobbdouglas_kappa=None,
    f_tau=None,
    f_sigmoid_scale=None,
    g_iso_p=None,
    cd_filter_rho=None,
    cd_filter_k_pct=None,
    cd_filter_kind=None,
    divergence=None,
    w_max=None,
    nu_output_transform=None,
    temperature=None,
    expectile=None,
    beta_schedule="constant",
    beta_init=None,
    beta_final=None,
    beta_warmup_steps=0,
    standardize_e=False,
    traj_filter_path=None,
    transition_filter_path=None,
    global_mixture: bool = False,
    extraction: str = "original",
    wbc_weight: str = "raw",
    traj_mixture: bool = False,
    racc_encoding: str = "ftilde",
    z_keeplist=None,
    exp_f_scale=None,
) -> str:
    """Deterministic run BASENAME (not full path) derived from knobs that distinguish
    one sweep entry from another — excluding the scaling knobs, which go in the subfolder.
    Used as-is for the leaf directory name — no UUID suffix, so the sweep runner can
    resume crashed jobs by reconstructing the path from hyperparameters alone.

    `learner_type` is prefixed only when != "aet", so existing aet-learner save_dirs
    keep their original names (back-compat for the 36+ runs already on disk). Non-aet
    learners (fairdice baselines: optidice / fairdice) get a `<learner>_` prefix that
    makes their save_dir distinct from the AETDICE-style runs that share the same
    (G, F, beta, seed, prefs) — which would otherwise collide on the same path.

    `cobbdouglas_rho` is suffixed onto the F-tag only when utility_F_type=='cobbdouglas'
    AND rho != 0.5 (the default). Same back-compat trick as learner_type: the first
    batch of CD runs trained at the default symmetric rho=0.5 keep their existing
    `F-cobbdouglas` dir names, while a rho=0.7 run gets `F-cobbdouglas_rho0.7` and
    sits in a separate dir with no risk of overwriting the rho=0.5 runs."""
    if utility_G_type == "linear":
        g_tag = f"G-linear_w{fmt_to(pref_weights)}"
    elif utility_G_type == "piecewise_log":
        g_tag = "G-plog"
    elif utility_G_type in ("weighted_log", "isoelastic", "isoelastic_ce"):
        # 2026-10-25 concave G — 가중치·지수 p 가 G 를 바꾸므로 이름에 넣는다.
        short = {"weighted_log": "wlog", "isoelastic": "iso", "isoelastic_ce": "isoce"}[utility_G_type]
        g_tag = f"G-{short}"
        if utility_G_type != "weighted_log":
            g_tag += f"_p{float(g_iso_p):g}"
        if pref_weights is not None:
            g_tag += f"_w{fmt_to(pref_weights)}"
    else:
        g_tag = f"G-{utility_G_type}"

    if utility_F_type == "power_F":
        f_tag = f"F-power_p-{power_p:g}"
    elif utility_F_type == "exponential_F":
        f_tag = "F-exp_F"
        # 학습용 F 의 스케일 S. 기본 50 은 토큰을 생략해 기존 런 이름과 충돌하지 않게 한다
        # (cobbdouglas rho 와 같은 back-compat 규칙). S=100 → `F-exp_F_S100`.
        if exp_f_scale is not None and float(exp_f_scale) != 50.0:
            f_tag = f"{f_tag}_S{float(exp_f_scale):g}"
    elif utility_F_type == "cobbdouglas":
        f_tag = "F-cobbdouglas"
        # ★ 2026-09-16: rho 는 **항상** 이름에 넣는다. 예전에는 기본값 0.5 일 때 토큰을
        #   생략해서 `F-cobbdouglas` 가 ρ=0.5 를 뜻하는 함정이 있었다 (디렉토리 이름만
        #   보고 ρ 를 알 수 없었다). CD 가 단일 수식이 된 지금 ρ 가 유일한 축이므로
        #   생략하면 안 된다. 옛 런은 전부 폐기했으므로 back-compat 부담도 없다.
        if cobbdouglas_rho is not None:
            f_tag = f"{f_tag}_rho{float(cobbdouglas_rho):g}"
        # threshold is None by default (back-compat for runs predating the knob);
        # only suffix when it's actually set, so existing CD runs without threshold
        # keep their dir names. When set, the threshold-shaping populates dim 1 of
        # the CD output, giving log(E[CD]) + log(E[plogT]) under G=piecewise_log.
        if reward_threshold is not None:
            f_tag = f"{f_tag}_T{fmt_threshold(reward_threshold)}"
        # amp == 1.0 (default) → no suffix; pre-amp CD runs keep their dir names.
        # Otherwise emit `_amp<value>` in scientific form (1e2, 1e3, 1e4, …).
        if cobbdouglas_amp is not None and float(cobbdouglas_amp) != 1.0:
            f_tag = f"{f_tag}_amp{fmt_sci(cobbdouglas_amp)}"
        # form: total(기본)은 토큰 없이 기존 이름 유지, rate(exp A) → `_cdrate`.
        if cobbdouglas_form not in (None, "total"):
            f_tag = f"{f_tag}_cd{cobbdouglas_form}"
        # safe-CD 결합 방식 (2026-11-03): plog(기본) 는 토큰 없이 기존 이름 유지.
        if cobbdouglas_safety not in (None, "plog"):
            f_tag = f"{f_tag}_sf{cobbdouglas_safety}"
        if cobbdouglas_kappa is not None and float(cobbdouglas_kappa) != 1.0:
            f_tag = f"{f_tag}_k{float(cobbdouglas_kappa):g}"
    elif utility_F_type == "sigmoid":
        f_tag = f"F-sig_tau{'-'.join(f'{float(t):g}' for t in f_tau)}_s{float(f_sigmoid_scale):g}"
    elif utility_F_type == "piecewise_log_threshold":
        # Threshold is REQUIRED for this F type, so it always appears in the name —
        # no back-compat default to elide. Different thresholds → different dirs.
        thr = float(reward_threshold) if reward_threshold is not None else 0.0
        f_tag = f"F-plogT_T-{thr:g}"
    else:
        f_tag = f"F-{utility_F_type}"

    # Horizon suffix — only when != 500 (the project-wide default). Different
    # max_seq_len values yield different time-embedding T_max, so checkpoints are
    # not interchangeable across horizons; the dir name has to reflect this.
    horizon_suffix = ""
    if max_seq_len is not None and int(max_seq_len) != 500:
        horizon_suffix = f"_T{int(max_seq_len)}"

    # CD-filter tokens — emitted whenever cd_filter is active, regardless of
    # learner. Lets AET / IQL / BC on top-K% data each land in distinct dirs.
    bc_suffix = ""
    # ★ racc-조건부 BC 는 **다른 정책 클래스**(augmented MDP)라 반드시 이름으로 갈린다.
    #   기본 state 는 토큰 없음 — 2026-09-16 이전 런 디렉토리 이름을 보존한다.
    if bc_conditioning is not None and str(bc_conditioning) != "state":
        bc_suffix += f"_{bc_conditioning}cond"
    # kind 는 **어느 궤적이 남는지**를 바꾸므로 필터가 실제로 작동할 때(k_pct<1) 이름에 넣는다.
    # k_pct=1.0 은 no-op 이라 kind 와 무관하게 같은 데이터셋 → 토큰을 붙이지 않는다
    # (2026-09-16 이전 top100pct 런 이름 보존).
    _filter_active = cd_filter_k_pct is not None and float(cd_filter_k_pct) < 1.0
    if cd_filter_rho is not None and (not _filter_active or str(cd_filter_kind) == "cd"):
        bc_suffix += f"_cdrho{float(cd_filter_rho):g}"
    if _filter_active and cd_filter_kind is not None and str(cd_filter_kind) != "cd":
        bc_suffix += f"_{cd_filter_kind}"
    if cd_filter_k_pct is not None:
        bc_suffix += f"_top{float(cd_filter_k_pct) * 100:g}pct"

    # Divergence + w_max tokens — only emitted when non-default ('CHI' is the
    # project default; w_max=∞ is no-clip). Keeps existing CHI runs unchanged
    # while letting CLIP_CHI / w_max sweeps land in distinct dirs.
    div_suffix = ""
    if divergence is not None and divergence != "CHI":
        div_suffix += f"_div{divergence}"
        if divergence in ("CLIP_CHI", "SOFT_CLIP_CHI") and w_max is not None and w_max != float("inf"):
            div_suffix += f"_wmax{float(w_max):g}"

    # ν-output reparameterization token — emitted only for non-default ('symexp').
    nu_suffix = ""
    if nu_output_transform is not None and nu_output_transform != "none":
        nu_suffix = f"_nu{nu_output_transform}"

    # IQL HP tokens — only emitted when explicitly set (axes default to None →
    # main argparse defaults). Lets per-rho IQL HP sweeps land in
    # distinct dirs from the default-HP IQL run at the same (rho, amp, β).
    iql_suffix = ""
    if temperature is not None:
        iql_suffix += f"_temp{float(temperature):g}"
    if expectile is not None:
        iql_suffix += f"_exp{float(expectile):g}"

    # β token: constant → unchanged 'betaX' for back-compat. Non-constant
    # schedules emit 'beta-<sched>_<init>to<final>' so each schedule shape lands
    # in a distinct save_dir from default-CHI/CHI runs that share other axes.
    if beta_schedule == "constant":
        beta_token = f"beta{fmt_sci(beta)}"
    else:
        beta_token = f"beta-{beta_schedule}_{fmt_sci(beta_init)}to{fmt_sci(beta_final)}"
        if beta_warmup_steps and int(beta_warmup_steps) > 0:
            w = int(beta_warmup_steps)
            beta_token += f"_w{w // 1000}k" if w >= 1000 else f"_w{w}"

    # Bellman-residual standardization token — emitted only when active so the
    # default-OFF runs keep their existing dir names.
    estd_suffix = "_estd" if standardize_e else ""

    # w-filter strategy token (Exp3-03) — emitted only when a keep-list is active
    # so baseline/other runs keep their existing dir names.
    wf_suffix = f"_wf-{wfilter_token(traj_filter_path)}" if traj_filter_path else ""
    # IDRL transition 마스크는 granularity 가 달라 별도 토큰(_tf-)을 쓴다 — 같은 _wf- 를
    # 재사용하면 두 필터를 이름으로 구분할 수 없다.
    wf_suffix += f"_tf-{wfilter_token(transition_filter_path)}" if transition_filter_path else ""
    wf_suffix += "_gmix" if global_mixture else ""
    # CLAUDE.md 명명 규칙: timecond 는 이름 뒤에 붙인다. 접미사가 없으면 original 이다.
    wf_suffix += f"_{extraction}" if extraction != "original" else ""
    wf_suffix += f"_w{wbc_weight}" if wbc_weight != "raw" else ""
    wf_suffix += "_tmix" if traj_mixture else ""

    # 조건부 racc 인코딩 토큰 — 기본('ftilde')이면 생략해 기존 런 이름을 유지한다.
    racc_suffix = "" if racc_encoding in (None, "ftilde") else f"_racc{racc_encoding}"
    # 잠재 z 토큰 — z_keeplist 가 있을 때만 (기존 런 이름 보존).
    z_suffix = "_z" if z_keeplist else ""

    base = (f"{g_tag}_{f_tag}_{beta_token}_seed{seed}"
            f"{horizon_suffix}{bc_suffix}{div_suffix}{nu_suffix}{iql_suffix}"
            f"{estd_suffix}{racc_suffix}{z_suffix}{wf_suffix}_{tag}")
    return base if learner_type == "aet" else f"{learner_type}_{base}"


def make_experiment_subfolder(
    divergence: str = "CHI",
    beta_schedule: str = "constant",
    standardize_e: bool = False,
    w_max=None,
    nu_output_transform: str = "none",
    traj_filter_path=None,
    transition_filter_path=None,
) -> str:
    """Token identifying which 'new technique' a run exercises. Combined with
    `make_save_subdirs` to route ablation runs under `<save_path>/ablation_study/<token>/`,
    parallel to (not nested in) the `rsf_musf` subfolder used for baselines.
    Default knobs (CHI / constant / no stde) return "" → caller routes to legacy
    rsf_musf layout unchanged.

    Examples:
      baseline                            →  ""
      stde only                           →  "stde"
      CLIP_CHI w=10                       →  "clipchi_w10"
      β-sched (exp)                       →  "bsched-exp"
      CLIP_CHI w=10 + β-sched + stde      →  "clipchi_w10_bsched-exp_stde"
      SOFT_CLIP_CHI w=100                 →  "softclipchi_w100"

    Tokens are duplicated between this subfolder and the basename suffixes
    (e.g., basename also has `_estd` / `_divCLIP_CHI_wmax10`). That's fine —
    the subfolder is for browsability ("show me everything under stde");
    the basename suffix is for unambiguous resume / dedup.
    """
    parts = []
    if divergence == "CLIP_CHI":
        if w_max is not None and w_max != float("inf"):
            parts.append(f"clipchi_w{int(w_max)}")
        else:
            parts.append("clipchi")
    elif divergence == "SOFT_CLIP_CHI":
        if w_max is not None and w_max != float("inf"):
            parts.append(f"softclipchi_w{int(w_max)}")
        else:
            parts.append("softclipchi")
    elif divergence not in ("CHI", None):
        parts.append(divergence.lower())

    if beta_schedule and beta_schedule != "constant":
        parts.append(f"bsched-{beta_schedule}")

    if standardize_e:
        parts.append("stde")

    if nu_output_transform and nu_output_transform != "none":
        parts.append(f"nu{nu_output_transform}")

    # w-filter coverage ablation (Exp3-03): all strategies share the 'wfilter' tree
    # (strategy id lives in the run basename via wfilter_token), keeping one
    # browsable ablation_study/wfilter/ subtree instead of 9 sibling folders.
    if traj_filter_path or transition_filter_path:
        parts.append("wfilter")

    return "_".join(parts)


def make_save_subdirs(
    reward_scaling_factor,
    mu_scaling_factor,
    reward_reshape: str = "none",
    divergence: str = "CHI",
    beta_schedule: str = "constant",
    standardize_e: bool = False,
    w_max=None,
    nu_output_transform: str = "none",
    traj_filter_path=None,
    transition_filter_path=None,
) -> tuple:
    """Path components between `save_path` and the run basename, returned as a
    tuple so the caller can `os.path.join(save_path, *parts, basename)`.

    Two layouts:
      - Baseline (no new techniques active):
          ('rsf<X>_musf<Y>[_<reshape>]',)
          → <save_path>/rsf<X>_musf<Y>/<basename>
      - Ablation (any of CLIP_CHI / SOFT_CLIP_CHI / β-schedule / standardize_e):
          ('ablation_study', '<technique_token>', 'rsf<X>_musf<Y>[_<reshape>]')
          → <save_path>/ablation_study/<technique>/rsf<X>_musf<Y>/<basename>

    Ablation runs are intentionally NOT under the top-level rsf_musf — they get
    their own ablation_study/ tree. Within an ablation, rsf_musf is preserved
    as a third level so runs at different scaling don't collide on the same path.
    """
    exp_token = make_experiment_subfolder(
        divergence=divergence,
        beta_schedule=beta_schedule,
        standardize_e=standardize_e,
        w_max=w_max,
        nu_output_transform=nu_output_transform,
        traj_filter_path=traj_filter_path,
        transition_filter_path=transition_filter_path,
    )
    rsf_token = make_subfolder_name(reward_scaling_factor, mu_scaling_factor, reward_reshape)
    if exp_token:
        return ("ablation_study", exp_token, rsf_token)
    return (rsf_token,)


def make_subfolder_name(reward_scaling_factor, mu_scaling_factor, reward_reshape: str = "none") -> str:
    """Subfolder encoding the scaling knobs: rsf (per-objective reward scaling vector)
    and musf (mu scaling for conjugate_G). Sits between save_path and the run basename
    so runs that share scaling are grouped together on disk, and the full save_dir is
    predictable from hyperparameters alone (no config.json parsing needed).

    `reward_reshape` is appended as a token (e.g. '_antv5') when != 'none', so
    reshape-mode runs sit in distinct subfolders from the base 2-obj runs and can
    never overwrite them on disk."""
    base = f"rsf{fmt_to(reward_scaling_factor)}_musf{fmt_sci(mu_scaling_factor)}"
    if reward_reshape and reward_reshape != "none":
        return f"{base}_{reward_reshape}"
    return base


def make_env_dataset_subfolder(env, dataset) -> str:
    """Env+dataset folder — groups runs that share environment and data source.
    Full save_dir layout: <base_save_path>/<env_dataset>/<rsf_musf>/<basename>.
    Example: "ant_amateur"."""
    return f"{env}_{dataset}"


def make_project_name(prefix, env, dataset, version) -> str:
    """wandb project name. Format: <prefix>_<env>_<dataset>_<version>.
    Example: "transformed_ant_amateur_v0424"."""
    return f"{prefix}_{env}_{dataset}_{version}"


def make_tag_name(prefix, env, dataset) -> str:
    """Run-identifying tag, appended to the save_dir basename.
    Format: <prefix>_<env>_<dataset>. Prefix is independent of make_project_name's
    prefix — set them separately in the sweep runner if you want them to differ.
    Example: "transformed_ant_amateur"."""
    return f"{prefix}_{env}_{dataset}"
