"""Setup phase functions for the train entrypoint.

Each function does one resolvable step in turning a validated `config` into
the runtime state needed by the training loop. All are called in `main.py` in
top-down order; none of them implicitly call each other (they share state only
through `config` mutations and explicit return values).
"""
import argparse
import os
import random

import jax
import numpy as np
import wandb

from environment.ant import AntEnv
from environment.half_cheetah import HalfCheetahEnv
from environment.hopper import HopperEnv
from environment.hopper_v3 import HopperEnv as HopperEnvV3
from environment.mo_pointmaze_v10 import MOPointMazeV10Env
from environment.swimmer import SwimmerEnv
from environment.walker2d import Walker2dEnv

from src.data.buffer import Buffer, load_d4morl_dataset, inject_buffer_stats, bufcache_key
from src.learners import LEARNER_MODULES
from src.run_naming import make_run_name, make_save_subdirs
from src.train.beta_schedule import make_beta_schedule
from src.train.config_io import dump_config_json
from src.data.reward_transform import CD_RESHAPES, RESHAPE_REWARD_DIM, cd_reward_bounds
from src.utility import Utility, make_utility_F_from_config


_ENV_MAP = {
    "walker2d":     Walker2dEnv,
    "ant":          AntEnv,
    "half_cheetah": HalfCheetahEnv,
    "hopper":       HopperEnv,
    "hopper_v3":    HopperEnvV3,
    "swimmer":      SwimmerEnv,
    "mo_pointmaze": MOPointMazeV10Env,
}


# Each reshape is mathematically tied to a specific env's reward layout —
# cross-pairing (e.g. walker2dcd on MO-Ant data) silently produces garbage.
_RESHAPE_TO_ENV = {
    "antcd":             "ant",
    "walker2dcd":        "walker2d",
    "half_cheetahcd":    "half_cheetah",
    "walker2d_cd_safe":  "walker2d",
    "walker2d_comp":     "walker2d",
    "walker2d_cd_safe_vel":    "walker2d",     # safe-CD S1–S3 (2026-11-03)
    "walker2d_cd_safe_margin": "walker2d",
    "walker2d_cd_safe_sat":    "walker2d",
    "walker2d_cd_safe_pv":     "walker2d",     # S0 + S1 두 제약 (2026-11-05)
    "walker2d_cd_safe_pok":    "walker2d",     # S0ok 스텝별 자세 문턱 (D6.31)
    "walker2d_cd_safe_pokv":   "walker2d",     # S0ok + S1 (D6.31)
    "walker2d_cd_safe_ha":     "walker2d",     # 높이 + 각도 여유 (D6.37)
    "walker2d_cd_safe_h":      "walker2d",     # 높이 여유만 (D6.38)
    "walker2d_cd_safe_hm":     "walker2d",     # 채점용 높이 여유 + 높이 준수 (D6.39)
}


def select_learner(config):
    """Resolve `config.learner_type` → learner module (the one exposing the
    init_train_state / train_step / get_model / save_model / load_model /
    save_checkpoint / load_checkpoint surface)."""
    return LEARNER_MODULES[config.learner_type]


def build_env(config):
    """Instantiate the env, then write env-derived dims back onto `config`
    (state_dim, action_dim, reward_dim, action_scale, hidden_dims)."""
    if config.env_name not in _ENV_MAP:
        raise ValueError(
            f"Invalid environment name: {config.env_name}. "
            f"Choose from {list(_ENV_MAP)}"
        )
    env = _ENV_MAP[config.env_name]()
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.reward_dim = env.obj_dim
    config.action_scale = np.asarray(
        getattr(env, "action_scale", np.ones(config.action_dim, dtype=np.float32)),
        dtype=np.float32,
    )
    config.hidden_dims = [config.hidden_dim] * config.num_layers
    return env


def apply_reward_reshape(config):
    """If `--reward_reshape != none`, validate the (env, reshape) pair and
    override `config.reward_dim` to the reshape's output dim. Networks must
    size their racc input from the post-reshape dim, NOT raw env.reward_dim."""
    if config.reward_reshape == "none":
        return
    # Imported lazily — only relevant when a reshape is actually requested,
    # and reward_transform.py has its own large module-level side effects.
    from src.data.reward_transform import RESHAPE_REWARD_DIM

    required_env = _RESHAPE_TO_ENV.get(config.reward_reshape)
    if required_env is not None and config.env_name != required_env:
        # Late argparse-error replacement (parser is already consumed); raise
        # an explicit ValueError so the misuse is loud at startup.
        raise SystemExit(
            f"--reward_reshape={config.reward_reshape} requires "
            f"--env_name={required_env} (got {config.env_name!r})."
        )
    if config.reward_reshape not in RESHAPE_REWARD_DIM:
        raise SystemExit(
            f"Unknown --reward_reshape={config.reward_reshape!r}. "
            f"Supported: 'none' | {sorted(RESHAPE_REWARD_DIM.keys())}."
        )
    config.reward_dim = RESHAPE_REWARD_DIM[config.reward_reshape]


def assert_dim_consistency(config):
    """Length / divisibility checks that only make sense AFTER env build +
    reshape are settled (need final config.reward_dim, etc.)."""
    if config.pref_weights is not None:
        # pref_weights are dotted with F̃-telescoped rewards, so the expected
        # length is F̃-dim, not raw reward_dim. They coincide for pointwise F;
        # CobbDouglas emits 2-D F̃, except the per-channel threshold modes (linear / inlog), which
        # emit [CD, ΣS_1 − th_1, …] = 1 + (reward_dim − 2) dims (2026-11-06: inlog is the first of
        # these used with G=linear, so the first to reach this check).
        from src.data.reward_transform import CD_SAFETY_PER_CHANNEL
        if config.utility_F_type != "cobbdouglas":
            expected_mu_dim = config.reward_dim
        elif getattr(config, "cobbdouglas_safety", None) in CD_SAFETY_PER_CHANNEL:
            expected_mu_dim = config.reward_dim - 1
        else:
            expected_mu_dim = 2
        assert len(config.pref_weights) == expected_mu_dim, (
            f"--pref_weights length ({len(config.pref_weights)}) must equal "
            f"F̃-dim ({expected_mu_dim}) for utility_F_type={config.utility_F_type!r} "
            f"(raw reward_dim={config.reward_dim})"
        )
    assert len(config.reward_scaling_factor) == config.reward_dim, (
        f"--reward_scaling_factor length ({len(config.reward_scaling_factor)}) "
        f"must equal reward_dim ({config.reward_dim})"
    )
    assert config.log_interval % config.chunk_size == 0, (
        f"--log_interval ({config.log_interval}) must be divisible by "
        f"--chunk_size ({config.chunk_size})"
    )


def build_cd_filter(config):
    """Resolve the optional top-K% trajectory filter (역사적 이름: cd_filter).

    Applies for ANY learner. `--cd_filter_kind` 는 현재 "cd"(CD 점수 상위 K%) 하나뿐이다
    ("corner" 는 정당화 불가한 post-hoc 휴리스틱이라 2026-09-20 제거).
    reward_min/max stay computed on the FULL dataset BEFORE filtering, so
    eval-time normalization is comparable across filtered/unfiltered runs.
    BC requires this filter (k_pct=1.0 이면 no-op = 전 궤적 모방)."""
    if config.cd_filter_rho is not None or config.cd_filter_k_pct is not None:
        if config.cd_filter_rho is None or config.cd_filter_k_pct is None:
            raise ValueError(
                "--cd_filter_rho and --cd_filter_k_pct must be provided together "
                "(or both omitted)."
            )
        return {"rho": float(config.cd_filter_rho),
                "k_pct": float(config.cd_filter_k_pct),
                "kind": str(getattr(config, "cd_filter_kind", "cd") or "cd")}
    if config.learner_type == "bc":
        raise ValueError(
            "--learner_type=bc requires both --cd_filter_rho and --cd_filter_k_pct."
        )
    return None


def build_utility_F(config):
    """inner utility F 선택·인스턴스화, `config.utility_F` 에 주입. 생성 로직은 `make_utility_F`
    단일 소스 (eval-reload `rollout._rebuild_utility_F` 와 공유 → 학습/평가 F 일치)."""
    # ★ CD 인자는 utility.cd_kwargs_from_config 한 곳에서만 온다 (SSOT). 여기서 손으로
    #   나열하면 reload/eval 경로와 어긋나 같은 버그가 또 난다 (2026-09-14 부채 정리).
    config.utility_F = make_utility_F_from_config(config)


def resolve_save_dir(config):
    """Resume: reuse the given save_dir as-is. Fresh: deterministic save_dir
    from hyperparameters (no uuid suffix — keeps auto-resume by the dispatcher
    unambiguous). config.json is written only on fresh starts so a resume with
    mismatched args doesn't silently overwrite the original record.

    Returns (run_name, save_dir). Creates save_dir/eval/ as a side effect."""
    if config.resume_dir is not None:
        save_dir = os.path.abspath(config.resume_dir)
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"--resume_dir does not exist: {save_dir}")
        run_name = os.path.basename(save_dir.rstrip("/"))
        os.makedirs(os.path.join(save_dir, "eval"), exist_ok=True)
        return run_name, save_dir

    if config.run_name is not None:
        run_name = config.run_name
        save_dir = os.path.join(config.save_path, run_name)
    else:
        run_name = make_run_name(
            utility_G_type=config.utility_G_type,
            utility_F_type=config.utility_F_type,
            power_p=config.power_p,
            exp_f_scale=getattr(config, "exp_f_scale", 50.0),
            pref_weights=config.pref_weights,
            beta=config.beta,
            seed=config.seed,
            tag=config.tag,
            learner_type=config.learner_type,
            cobbdouglas_rho=config.cobbdouglas_rho,
            max_seq_len=config.max_seq_len,
            reward_threshold=config.reward_threshold,
            cobbdouglas_amp=config.cobbdouglas_amp,
            cobbdouglas_form=getattr(config, "cobbdouglas_form", None),
            cobbdouglas_safety=getattr(config, "cobbdouglas_safety", None),
            cobbdouglas_kappa=getattr(config, "cobbdouglas_kappa", None),
            f_tau=getattr(config, "f_tau", None),
            f_sigmoid_scale=getattr(config, "f_sigmoid_scale", None),
            g_iso_p=getattr(config, "g_iso_p", None),
            bc_conditioning=getattr(config, "bc_conditioning", None),
            racc_encoding=getattr(config, "racc_encoding", "ftilde"),
            z_keeplist=getattr(config, "z_keeplist", None),
            # ★ job_runner 의 make_run_name 호출과 **인자가 완전히 같아야** 한다 — 어긋나면
            #   dispatcher 의 "이미 완료" 판정이 실제 저장 경로와 다른 곳을 본다
            #   (2026-09-16: cd_filter_kind 를 여기만 빠뜨려 corner 런이 cd 런 이름으로 저장됐다).
            cd_filter_kind=getattr(config, "cd_filter_kind", None),
            cd_filter_rho=getattr(config, "cd_filter_rho", None),
            cd_filter_k_pct=getattr(config, "cd_filter_k_pct", None),
            divergence=getattr(config, "divergence", None),
            w_max=getattr(config, "w_max", None),
            nu_output_transform=getattr(config, "nu_output_transform", None),
            # Only thread IQL HPs into the basename when they differ from main's
            # argparse defaults (else default IQL runs would needlessly bloat
            # their dir names).
            temperature=(getattr(config, "temperature", None)
                          if config.learner_type == "iql"
                             and float(getattr(config, "temperature", 3.0)) != 3.0
                          else None),
            expectile=(getattr(config, "expectile", None)
                        if config.learner_type == "iql"
                           and float(getattr(config, "expectile", 0.7)) != 0.7
                        else None),
            # β-schedule + standardize_e tokens — these were missing in an
            # earlier version and caused name-collision overwrites between
            # constant-β baselines and standardize_e / β-sched runs. Always
            # thread them; run_naming.make_run_name omits the suffix when
            # values match the back-compat defaults.
            beta_schedule=getattr(config, "beta_schedule", "constant"),
            beta_init=getattr(config, "beta_init", None),
            beta_final=getattr(config, "beta_final", None),
            beta_warmup_steps=getattr(config, "beta_warmup_steps", 0),
            standardize_e=bool(getattr(config, "standardize_e", False)),
            traj_filter_path=getattr(config, "traj_filter_path", None),
            transition_filter_path=getattr(config, "transition_filter_path", None),
            global_mixture=bool(getattr(config, "global_mixture", False)),
            extraction=str(getattr(config, "extraction", "original")),
            wbc_weight=str(getattr(config, "wbc_weight", "raw")),
            traj_mixture=bool(getattr(config, "traj_mixture", False)),
        )
        # Routes to either rsf_musf/ (baseline) or ablation_study/<technique>/
        # (any new technique active). Layouts are siblings under save_path,
        # never nested — baselines and ablations live on entirely separate
        # directory trees.
        save_subdirs = make_save_subdirs(
            reward_scaling_factor=config.reward_scaling_factor,
            mu_scaling_factor=config.mu_scaling_factor,
            reward_reshape=config.reward_reshape,
            divergence=getattr(config, "divergence", "CHI"),
            beta_schedule=getattr(config, "beta_schedule", "constant"),
            standardize_e=bool(getattr(config, "standardize_e", False)),
            w_max=getattr(config, "w_max", None),
            nu_output_transform=getattr(config, "nu_output_transform", "none"),
            traj_filter_path=getattr(config, "traj_filter_path", None),
            transition_filter_path=getattr(config, "transition_filter_path", None),
        )
        save_dir = os.path.join(config.save_path, *save_subdirs, run_name)

    os.makedirs(os.path.join(save_dir, "eval"), exist_ok=True)
    return run_name, save_dir


def _buffer_cache_dir(config, cd_filter, traj_filter=None):
    """처리된-버퍼 캐시 경로 (없으면 None). build_buffer 와 resolve_reward_bounds 가 **같은**
    키를 써야 하므로 한 곳에 둔다 (SSOT).

    k_pct >= 1.0 인 cd_filter 는 전 궤적을 남기므로(top-100%) 필터된 데이터셋이 원본과
    byte-identical 이라 공유 캐시가 유효하다. (Full BC 가 cd_filter={k_pct:1.0} 을 쓰는데,
    이 예외가 없으면 캐시를 우회해 antpref100 의 ~100GB/30분 빌드를 매번 다시 한다.)
    """
    cd_is_noop = (cd_filter is None) or (float(cd_filter.get("k_pct", 0.0)) >= 1.0)
    if config.reward_reshape != "none" and cd_is_noop and traj_filter is None:
        return bufcache_key(config, config.reward_reshape)
    return None


def resolve_reward_bounds(config, cd_filter):
    """CD 전용 선해결 — `config.reward_min/max` 를 **utility_F 생성 전에** 채운다.

    CD 의 정규화 상수는 데이터셋 통계가 아니라 (0, T) 상수다(cd_reward_bounds). 그래서
    데이터셋을 미리 읽을 필요가 없다 — 예전 modified CD 는 offset_shift = rsf·T·rmin/span
    를 만들려고 여기서 데이터셋을 한 번 더 읽었지만, shift 자체를 없앴으므로 그 경로가
    통째로 사라졌다 (2026-09-16).

    CD 가 아니면 아무것도 하지 않는다 — 기존 경로의 동작을 바꾸지 않기 위해서다.
    """
    if getattr(config, "utility_F_type", None) != "cobbdouglas":
        return
    reshape = getattr(config, "reward_reshape", "none")
    if reshape not in CD_RESHAPES:
        raise ValueError(
            f"utility_F_type='cobbdouglas' 인데 reward_reshape={reshape!r} 이다. "
            f"CD 는 (reward, cost) 레이아웃이 필요하다 — {CD_RESHAPES} 중 하나를 주라.")
    reward_dim = RESHAPE_REWARD_DIM[reshape]
    config.reward_min, config.reward_max = cd_reward_bounds(reward_dim, config.max_seq_len)
    print(f"[CD] scale-only 정규화 상수: min={config.reward_min}, max={config.reward_max}",
          flush=True)


def build_buffer(config, cd_filter):
    """Load d4morl pkl + construct Buffer. Mutates `config` with the buffer's
    derived stats (state_mean/std, max_racc/min_racc, racc_mean/std/dim,
    reward_min/max) — these are consumed downstream by networks and eval."""
    print(f"Loading dataset from {config.dataset_path} ...")
    # w-based trajectory filter (Exp3-03): keep only trajectories in the .npy list.
    traj_filter = None
    tf_path = getattr(config, "traj_filter_path", None)
    if tf_path:
        traj_filter = {"keep_ids_path": tf_path,
                       # 에피소드 단위 잠재 z (관측 라벨). 이 목록에 든 궤적이 z=1.
                       "z_keeplist": getattr(config, "z_keeplist", None)}
    # Processed-buffer cache: only for reshaped datasets (heavy per-traj accumulation +
    # F̃-transform; antpref100's 100-dim build is ~30 min / 100 GB). The standard 2-obj
    # (reshape=none) path is unchanged. Key captures every determinant of the processed
    # arrays so an HP sweep (varies only network/mixture) reuses one cache.
    cache_dir = _buffer_cache_dir(config, cd_filter, traj_filter)

    if cache_dir is not None and os.path.exists(cache_dir):
        print(f"Buffer cache HIT -> {cache_dir} (skipping dataset load/processing)")
        buffer = Buffer(config, None, config.utility_F, cache_path=cache_dir)
    else:
        # resolve_reward_bounds 가 이미 읽어 뒀으면 그걸 쓴다 (같은 인자로 두 번 읽지 않는다).
        data = getattr(config, "_preloaded_dataset", None)
        if data is None:
            data = load_d4morl_dataset(
                config.dataset_path, config.env_name, config.reward_scaling_factor,
                reward_reshape=config.reward_reshape,
                max_seq_len=config.max_seq_len,
                cd_filter=cd_filter,
                traj_filter=traj_filter,
                max_trajectories=getattr(config, "max_trajectories", None),
                reward_norm_scale=getattr(config, "reward_norm_scale", None),
            )
        else:
            print("[buffer] resolve_reward_bounds 가 로드한 데이터셋 재사용", flush=True)
            config._preloaded_dataset = None
        print(f"Dataset loaded: {data['states'].shape[0]} transitions")
        print(f"  reward_min={data['reward_min']}, reward_max={data['reward_max']}")
        buffer = Buffer(config, data, config.utility_F, cache_path=cache_dir)
    inject_buffer_stats(config, buffer)
    return buffer


def attach_beta_schedule(config):
    """Build the β schedule callable once. AET/FairDICE train_step reads
    `config.beta_schedule_fn(step)` instead of the constant `config.beta`.
    Constant schedule degrades to a closure that always returns config.beta
    (back-compat for runs that never set --beta_schedule)."""
    config.beta_schedule_fn = make_beta_schedule(config)


def maybe_dump_config(config, save_dir):
    """Write config.json on fresh starts only. Skip on resume so the original
    record isn't clobbered by a mismatched run."""
    if config.resume_dir is None:
        dump_config_json(config, save_dir)


def seed_everything(config):
    """Seed all RNG sources from `config.seed`. Returns the JAX key (the only
    one carried explicitly through the train loop)."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    return jax.random.PRNGKey(config.seed)


def init_run_state(config, save_dir, learner):
    """Build the initial train_state (fresh) or restore from checkpoint (resume).
    Returns (train_state, key, start_iter)."""
    key = seed_everything(config)
    if config.resume_dir is not None:
        train_state, key, start_iter = learner.load_checkpoint(save_dir, config)
        print(
            f"Resumed from iter {start_iter} "
            f"(step {start_iter * config.log_interval} / {config.total_train_steps})"
        )
    else:
        train_state = learner.init_train_state(config)
        if getattr(config, "critic_from", None):
            # Fresh runs only: a resumed checkpoint already holds the borrowed ν/μ.
            train_state = learner.load_frozen_dual(train_state, config)
        start_iter = 0
    return train_state, key, start_iter



def attach_wbc_weights(config, buffer, train_state, learner):
    """--wbc_weight traj | action: 고정 critic 에서 궤적 가중표 / 상태 가중 회귀를 만들어 config 에 붙인다.
    그 외 설정에서는 no-op (src/learners/wbc_state_weight.py)."""
    from src.learners import wbc_state_weight
    wbc_state_weight.attach(config, buffer, train_state, learner)

def init_wandb(config, save_dir, run_name):
    """No-op when --use_wandb is unset. On resume reuses the run id saved on
    first init; 'allow' makes a missing/stale id silently fall through to a
    new run instead of erroring."""
    if not config.use_wandb:
        return
    wandb_project = config.wandb_project or config.tag
    # Strip non-serializable values (callables like beta_schedule_fn) before
    # handing config to wandb — keeps the wandb config snapshot clean while
    # leaving the original `config` SimpleNamespace intact for train_step.
    wandb_config = {k: v for k, v in vars(config).items() if not callable(v)}
    wandb_id_path = os.path.join(save_dir, "wandb_run_id.txt")
    if config.resume_dir is not None and os.path.exists(wandb_id_path):
        with open(wandb_id_path) as fp:
            prev_wandb_id = fp.read().strip()
        wandb.init(project=wandb_project, id=prev_wandb_id, resume="allow",
                   config=wandb_config, settings=wandb.Settings(quiet=True))
    else:
        wandb.init(project=wandb_project, name=run_name, config=wandb_config,
                   settings=wandb.Settings(quiet=True))
        with open(wandb_id_path, "w") as fp:
            fp.write(wandb.run.id)
