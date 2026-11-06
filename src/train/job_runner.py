"""Per-job execution — turns one expanded job dict into a subprocess call.

Public:
    run_job(gpu, job, dry_run=False) → bool   # True on success / SKIP

Resume logic and CLI-flag assembly live here so the dispatcher can stay
small: queue + worker + threading only.
"""
import os
import subprocess
import sys

from src.run_naming import (
    make_env_dataset_subfolder,
    make_project_name,
    make_run_name,
    make_save_subdirs,
    make_tag_name,
)
from src.train.dispatch_settings import (
    BATCH_SIZE,
    CHUNK_SIZE,
    DIVERGENCE,
    EVAL_EPISODES,
    EVAL_EVERY_N_LOGS,
    HIDDEN_DIM,
    LEARNER_TRAIN_STEPS,
    LOG_INTERVAL,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    PROJECT_PREFIX,
    SAVE_PATH,
    TAG_PREFIX,
    TOTAL_TRAIN_STEPS,
    USE_WANDB,
    VERSION,
)

# Picked up by mujoco_py / EGL at subprocess startup. Set once at module import
# (which is the dispatcher process), then inherited by every child subprocess.
os.environ["LD_LIBRARY_PATH"] = (
    "/root/.mujoco/mujoco210/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
)


def get_num_gpus() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        lines = [ln for ln in result.stdout.strip().splitlines() if ln]
        return max(len(lines), 1)
    except FileNotFoundError:
        return 1


def env_to_mo(env: str) -> str:
    """e.g. half_cheetah -> MO-HalfCheetah-v2"""
    camel = "".join(part.capitalize() for part in env.split("_"))
    return f"MO-{camel}-v2"


# Dataset naming override: key -> (mo_name, size_token). 키는 `(env, dataset)` 을 먼저 보고
# 없으면 `env` 로 되짚는다. 어느 쪽도 없으면 기본 (env_to_mo(env), "50000") = d4morl 50k 관례.
# 새 데이터셋은 **한 줄만** 추가한다 (dispatch-table, if-tree 금지).
#   ("ant", "mjx") — mjx-native 오프라인 데이터셋 (2026-09-16). 같은 env 인데 크기가 달라
#   env 키만으로는 표현이 안 돼 (env, dataset) 축을 열었다. 만든 경위는
#   experiments/oracle/build_mjx_dataset.py 참조 — sim-to-sim 격차를 없애기 위한 것이다.
_ENV_DATASET_OVERRIDE = {
    "mo_pointmaze":  ("MO-PointMaze-v10", "92400"),
    ("ant", "mjx"):  ("MO-Ant-v2", "10000"),
}


def dataset_path(env: str, dataset: str, entropy_level: str) -> str:
    mo_name, size = _ENV_DATASET_OVERRIDE.get(
        (env, dataset), _ENV_DATASET_OVERRIDE.get(env, (env_to_mo(env), "50000")))
    return f"./environment/dataset/{mo_name}/{mo_name}_{size}_{dataset}_{entropy_level}.pkl"


def _resume_status(save_dir: str) -> str:
    """Inspect a prospective save_dir. Returns:
      'done'   — final_returns.npy present, training already completed.
      'resume' — a live checkpoint (model/ + resume_key.npy) exists.
      'fresh'  — nothing usable; start from scratch (may overwrite an empty dir).
    main.py writes final_returns.npy only on clean completion, so its presence
    is a reliable 'done' marker. model/ + resume_key.npy are written together
    by save_checkpoint every log_interval."""
    if not os.path.isdir(save_dir):
        return "fresh"
    if os.path.exists(os.path.join(save_dir, "eval", "final_returns.npy")):
        return "done"
    if (os.path.exists(os.path.join(save_dir, "model"))
            and os.path.exists(os.path.join(save_dir, "resume_key.npy"))):
        return "resume"
    return "fresh"


def run_job(gpu: int, job: dict, dry_run: bool = False) -> bool:
    """Returns True on success (or SKIP); False on subprocess failure."""
    env                    = job["env"]
    learner_type           = job["learner_type"]
    utility_G_type         = job["utility_G_type"]
    utility_F_type         = job["utility_F_type"]
    power_p                = job["power_p"]
    exp_f_scale            = job.get("exp_f_scale", 50.0)
    pref_weights           = job["pref_weights"]
    reward_scaling_factor  = job["reward_scaling_factor"]
    mu_scaling_factor      = job["mu_scaling_factor"]
    reward_reshape         = job.get("reward_reshape", "none")
    beta                   = job["beta"]
    seed                   = job["seed"]
    dataset                = job["dataset"]
    entropy_level          = job["entropy_level"]
    cobbdouglas_rho        = job.get("cobbdouglas_rho", 0.5)
    max_seq_len            = job.get("max_seq_len", MAX_SEQ_LEN)
    reward_threshold       = job.get("reward_threshold")  # None unless f_type=plogT
    cobbdouglas_amp        = job.get("cobbdouglas_amp", 1.0)
    cobbdouglas_form       = job.get("cobbdouglas_form", "total")
    bc_conditioning        = job.get("bc_conditioning", "state")   # learner=bc 전용 축
    racc_encoding          = job.get("racc_encoding", "ftilde")    # 조건부 racc 입력 형태
    z_keeplist             = job.get("z_keeplist")                 # 에피소드 단위 잠재 z 라벨
    cd_filter_kind         = job.get("cd_filter_kind", "cd")   # 상위 K% 필터 기준
    cd_filter_rho          = job.get("cd_filter_rho")     # None unless learner=bc
    cd_filter_k_pct        = job.get("cd_filter_k_pct")   # None unless learner=bc
    divergence             = job.get("divergence", DIVERGENCE)
    w_max                  = job.get("w_max", float("inf"))
    nu_output_transform    = job.get("nu_output_transform", "none")
    temperature            = job.get("temperature")  # None = main default
    expectile              = job.get("expectile")
    beta_schedule          = job.get("beta_schedule", "constant")
    beta_init              = job.get("beta_init")     # None for constant
    beta_final             = job.get("beta_final")    # None for constant
    beta_warmup_steps      = int(job.get("beta_warmup_steps", 0))
    standardize_e          = bool(job.get("standardize_e", False))
    traj_filter_path       = job.get("traj_filter_path")  # None unless w-filter ablation
    transition_filter_path = job.get("transition_filter_path")
    global_mixture         = bool(job.get("global_mixture", False))
    extraction             = job.get("extraction", "original")
    wbc_weight             = job.get("wbc_weight", "raw")
    traj_mixture           = bool(job.get("traj_mixture", False))  # IDRL zero-w transition 마스크

    # Per-job naming — derived from env + dataset + module-level constants so sweep
    # dicts don't have to repeat project/tag/save_path across every entry.
    project   = make_project_name(PROJECT_PREFIX, env, dataset, VERSION)
    tag       = make_tag_name(TAG_PREFIX, env, dataset)
    save_path = os.path.join(SAVE_PATH, make_env_dataset_subfolder(env, dataset))

    # power_p is None iff utility_F_type != "power_F" (main.py rejects
    # --power_p for non-power_F F types). Skip the flag entirely in that case.
    p_label = f"(p={power_p:g})" if power_p is not None else ""
    label = (
        f"env={env} G={utility_G_type} F={utility_F_type}{p_label} "
        f"pref={pref_weights} scale={reward_scaling_factor} beta={beta} seed={seed} "
        f"[{dataset}/{entropy_level}]"
    )

    # Predict save_dir so we can detect a previous run at that location and
    # either skip it (done) or resume it (crashed mid-way). Must match the
    # path that main.py itself will compute — both sides share
    # run_naming.make_run_name for exactly this reason.
    run_name = make_run_name(
        utility_G_type=utility_G_type,
        utility_F_type=utility_F_type,
        power_p=power_p,
        exp_f_scale=exp_f_scale,
        pref_weights=pref_weights,
        beta=beta,
        seed=seed,
        tag=tag,
        learner_type=learner_type,
        cobbdouglas_rho=cobbdouglas_rho,
        max_seq_len=max_seq_len,
        reward_threshold=reward_threshold,
        cobbdouglas_amp=cobbdouglas_amp,
        cobbdouglas_form=cobbdouglas_form,
        f_tau=job.get("f_tau"),
        f_sigmoid_scale=job.get("f_sigmoid_scale"),
        g_iso_p=job.get("g_iso_p"),
        bc_conditioning=bc_conditioning,
        racc_encoding=racc_encoding,
        z_keeplist=z_keeplist,
        cd_filter_kind=cd_filter_kind,
        cd_filter_rho=cd_filter_rho,
        cd_filter_k_pct=cd_filter_k_pct,
        divergence=divergence,
        w_max=w_max,
        nu_output_transform=nu_output_transform,
        temperature=temperature,
        expectile=expectile,
        beta_schedule=beta_schedule,
        beta_init=beta_init,
        beta_final=beta_final,
        beta_warmup_steps=beta_warmup_steps,
        standardize_e=standardize_e,
        traj_filter_path=traj_filter_path,
        transition_filter_path=transition_filter_path,
        global_mixture=global_mixture,
        extraction=extraction,
        wbc_weight=wbc_weight,
        traj_mixture=traj_mixture,
    )
    # Routes to either rsf_musf/ (baseline) or ablation_study/<technique>/
    # (any new-technique knob active). Sibling trees under save_path; never mix.
    save_subdirs = make_save_subdirs(
        reward_scaling_factor=reward_scaling_factor,
        mu_scaling_factor=mu_scaling_factor,
        reward_reshape=reward_reshape,
        divergence=divergence,
        beta_schedule=beta_schedule,
        standardize_e=standardize_e,
        w_max=w_max,
        nu_output_transform=nu_output_transform,
        traj_filter_path=traj_filter_path,
        transition_filter_path=transition_filter_path,
    )
    save_dir = os.path.join(save_path, *save_subdirs, run_name)
    status = _resume_status(save_dir)

    if status == "done":
        print(f"[GPU {gpu}] SKIP (already completed) {label}", flush=True)
        return True

    # Per-job override for total_train_steps with cascading fallback:
    #   1. Sweep-level metadata "total_train_steps" (set per-sweep for one-off
    #      long-train experiments) wins if present.
    #   2. Per-learner default (LEARNER_TRAIN_STEPS) — e.g., BC converges fast.
    #   3. Project-wide TOTAL_TRAIN_STEPS as the final fallback.
    total_train_steps = job.get("total_train_steps") \
        or LEARNER_TRAIN_STEPS.get(learner_type, TOTAL_TRAIN_STEPS)
    cmd = [
        sys.executable, "-m", "src.train.main",
        "--env_name",             env,
        "--learner_type",         learner_type,
        "--utility_G_type",       utility_G_type,
        "--utility_F_type",       utility_F_type,
        "--divergence",           divergence,
        "--beta",                 beta,
        "--seed",                 str(seed),
        "--num_layers",           str(NUM_LAYERS),
        "--eval_episodes",        str(EVAL_EPISODES),
        "--batch_size",           str(BATCH_SIZE),
        "--total_train_steps",    str(total_train_steps),
        "--log_interval",         str(LOG_INTERVAL),
        "--chunk_size",           str(CHUNK_SIZE),
        "--eval_every_n_logs",    str(EVAL_EVERY_N_LOGS),
        "--hidden_dim",           str(HIDDEN_DIM),
        "--max_seq_len",          str(max_seq_len),
        "--tag",                  tag,
        "--wandb_project",        project,
        "--save_path",            save_path,
        "--dataset_path",         dataset_path(env, dataset, entropy_level),
        "--reward_scaling_factor", *[str(s) for s in reward_scaling_factor],
        "--mu_scaling_factor",    str(mu_scaling_factor),
    ]
    if pref_weights is not None:
        cmd += ["--pref_weights", *[str(w) for w in pref_weights]]
    if power_p is not None:
        cmd += ["--power_p", str(power_p)]
    if float(exp_f_scale) != 50.0:
        # 기본 50 이면 플래그를 생략해 기존 런과 cmd 가 동일하게 유지된다.
        cmd += ["--exp_f_scale", str(exp_f_scale)]
    # 2026-10-25 새 F/G 모수 — job 에 있을 때만 넘긴다 (make_run_name 인자와 짝).
    if job.get("f_tau") is not None:
        cmd += ["--f_tau", *[str(t) for t in job["f_tau"]], "--f_sigmoid_scale", str(job["f_sigmoid_scale"])]
    if job.get("g_iso_p") is not None:
        cmd += ["--g_iso_p", str(job["g_iso_p"])]
    if reward_reshape != "none":
        # MUST be present for antv5 — main.py overrides config.reward_dim
        # to 5 only when --reward_reshape=antv5; without this flag the env's obj_dim=2
        # carries through and the assertion `len(pref_weights) == reward_dim` fails.
        cmd += ["--reward_reshape", reward_reshape]
    if divergence in ("CLIP_CHI", "SOFT_CLIP_CHI") and w_max != float("inf"):
        # w_max meaningful for CLIP_CHI and SOFT_CLIP_CHI (both clip the upper
        # tail of w); skip for divergences that don't reference w_max.
        cmd += ["--w_max", str(w_max)]
    if nu_output_transform != "none":
        cmd += ["--nu_output_transform", nu_output_transform]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if expectile is not None:
        cmd += ["--expectile", str(expectile)]
    # β-schedule flags: only forward for non-constant schedules. main's
    # validator REJECTS --beta_init / --beta_final for constant schedule, so silently
    # omitting them keeps the constant-case cmd line back-compat-clean.
    if beta_schedule != "constant":
        cmd += ["--beta_schedule", beta_schedule]
        cmd += ["--beta_init", str(beta_init)]
        cmd += ["--beta_final", str(beta_final)]
        if beta_warmup_steps > 0:
            cmd += ["--beta_warmup_steps", str(beta_warmup_steps)]
    # Bellman residual standardization — forwarded only when active so default
    # cmd lines stay byte-identical to pre-feature behavior (back-compat).
    if standardize_e:
        cmd += ["--standardize_e"]
    if traj_filter_path is not None:
        cmd += ["--traj_filter_path", traj_filter_path]
    if transition_filter_path is not None:
        cmd += ["--transition_filter_path", transition_filter_path]
    if global_mixture:
        cmd += ["--global_mixture"]
    if extraction != "original":
        cmd += ["--extraction", extraction]
    if wbc_weight != "raw":
        cmd += ["--wbc_weight", wbc_weight]
    if traj_mixture:
        cmd += ["--traj_mixture"]
    if utility_F_type == "cobbdouglas":
        # Only forward when F is actually cobbdouglas — main accepts the
        # flag in any context (it's only consumed at CobbDouglasUtility build time)
        # but keeping it gated drops noise from non-CD command lines.
        cmd += ["--cobbdouglas_rho", str(cobbdouglas_rho)]
        # amp == 1.0 is the default; only forward when != 1 to keep cmd lines clean
        # for back-compat single-amp sweeps. (main validator REJECTS
        # amp != 1.0 for non-CD F types, but here we're under f_type=cobbdouglas.)
        if float(cobbdouglas_amp) != 1.0:
            cmd += ["--cobbdouglas_amp", str(cobbdouglas_amp)]
        if cobbdouglas_form != "total":
            cmd += ["--cobbdouglas_form", cobbdouglas_form]
    if utility_F_type == "piecewise_log_threshold":
        # Threshold is REQUIRED at main's argparse time when this F is
        # used; sweep entries that pair this F with reward_thresholds=[None] (the
        # default placeholder) will get rejected explicitly there.
        if reward_threshold is None:
            raise ValueError(
                f"sweep '{job.get('name','?')}' uses f_type=piecewise_log_threshold "
                f"but reward_thresholds is [None]; supply explicit threshold(s) in _build_sweep."
            )
        cmd += ["--reward_threshold", str(reward_threshold)]
    elif utility_F_type == "cobbdouglas" and reward_threshold is not None:
        # Threshold is OPTIONAL for cobbdouglas — when set, fills dim 1 of the CD
        # output with piecewise_log(R0 − T), enabling the log-balanced
        # CD + plogT objective via G=piecewise_log.
        cmd += ["--reward_threshold", str(reward_threshold)]
    if learner_type == "bc":
        # Filtered-BC requires both flags — main validates pairing.
        if cd_filter_rho is None or cd_filter_k_pct is None:
            raise ValueError(
                f"sweep '{job.get('name','?')}' uses learner_type=bc but cd_filter_rhos / "
                f"cd_filter_k_pcts contains None; both must be set in _build_sweep."
            )
        # 목적함수가 아니라 **정책 클래스**를 바꾸는 축이라 기본값이라도 config.json 에
        # 남겨 reload 경로가 같은 구조를 복원하게 한다.
        cmd += ["--bc_conditioning", str(bc_conditioning)]
        cmd += ["--cd_filter_kind", str(cd_filter_kind)]
        cmd += ["--cd_filter_rho",   str(cd_filter_rho),
                "--cd_filter_k_pct", str(cd_filter_k_pct)]
    # 조건부 racc 인코딩 — 기본(ftilde)이면 생략해 기존 런 커맨드를 그대로 유지한다.
    if racc_encoding and racc_encoding != "ftilde":
        cmd += ["--racc_encoding", str(racc_encoding)]
    if z_keeplist:
        cmd += ["--z_keeplist", str(z_keeplist)]
    if USE_WANDB:
        cmd += ["--use_wandb"]
    if status == "resume":
        cmd += ["--resume_dir", save_dir]

    if dry_run:
        tag_verb = "RESUME" if status == "resume" else "FRESH"
        print(f"[DRY-RUN GPU {gpu}] {tag_verb} {label}", flush=True)
        print(f"  cmd: {' '.join(cmd)}", flush=True)
        return True

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu)

    verb = "RESUME" if status == "resume" else "START"
    print(f"[GPU {gpu}] {verb} {label}", flush=True)
    proc = subprocess.Popen(cmd, env=env_vars)
    returncode = proc.wait()

    if returncode != 0:
        print(f"[GPU {gpu}] FAILED (exit {returncode}) {label}", flush=True)
        return False

    print(f"[GPU {gpu}] DONE  {label}", flush=True)
    return True
