"""Rollout a saved src checkpoint for N episodes and save returns.

Usage:
    python -m src.rollout \
        --run_dir ./results_transformed/ant_amateur/<RUN_NAME> \
        --num_episodes 100

Reads <run_dir>/config.json (written by main.py) and <run_dir>/model/,
writes <run_dir>/eval/returns_<N>.npy and eval/returns_<N>_normalized.npy.





## ant sf1e4
python -m src.rollout \
    --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e4/G-plog_F-exp_F_beta1e-6_seed1_tfd4morl_ant_expert \
    --num_episodes 100

## ant sf1e6
python -m src.rollout \
    --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e6/G-plog_F-exp_F_beta1e-6_seed1_tfd4morl_ant_expert \
    --num_episodes 100


"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

# Force JAX onto CPU before it is imported. Eval is CPU-only anyway
# (see evaluate.py), and this avoids contending with a GPU
# that may already be held by a training process.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from src.utility import Utility, make_utility_F_from_config
from environment.walker2d import Walker2dEnv
from environment.ant import AntEnv
from environment.half_cheetah import HalfCheetahEnv
from environment.hopper import HopperEnv
from environment.hopper_v3 import HopperEnv as HopperEnvV3
from environment.swimmer import SwimmerEnv
from environment.mo_pointmaze_v10 import MOPointMazeV10Env
import orbax.checkpoint as orbax
from orbax.checkpoint import checkpoint_utils as ocp_utils
# 학습과 같은 dispatch 표 (SSOT). src.learners 는 main 의 argparse / wandb 를 끌어오지 않는다.
# 예전 복사본에는 IQL 이 빠져 있어 ESR-IQL 체크포인트를 재평가하지 못했다 (2026-11-05).
from src.learners import LEARNER_MODULES
from src.eval.evaluate import evaluate as evaluate_GF


def load_model_cpu(path, config, init_train_state):
    """Like <learner>.load_model, but force-restore arrays onto CPU.

    The checkpoint records each array's save-time device (cuda:0). When we boot
    with JAX_PLATFORMS=cpu, orbax can't resolve cuda:0 and refuses to deserialize.
    construct_restore_args walks the target tree (which init_train_state built
    on CPU because of JAX_PLATFORMS) and generates ArrayRestoreArgs whose shardings
    point at CPU — overriding the save-time sharding metadata.

    `init_train_state` is passed in so we can dispatch on config.learner_type
    without coupling this helper to a specific learner module.
    """
    train_state = init_train_state(config)
    restore_args = ocp_utils.construct_restore_args(train_state)
    checkpointer = orbax.PyTreeCheckpointer()
    return checkpointer.restore(path, item=train_state, restore_args=restore_args)


ENV_MAP = {
    "walker2d": Walker2dEnv,
    "ant": AntEnv,
    "half_cheetah": HalfCheetahEnv,
    "hopper": HopperEnv,
    "hopper_v3": HopperEnvV3,
    "swimmer": SwimmerEnv,
    "mo_pointmaze": MOPointMazeV10Env,
}


def _rebuild_utility_F(cfg_dict):
    """checkpoint 의 config.json → 학습 때와 동일한 F 재구성. 생성 로직은 `make_utility_F` 단일
    소스 (setup.build_utility_F 와 공유). back-compat 기본값으로 legacy config 도 동일 reload;
    strict_threshold 로 piecewise_log_threshold 의 threshold 누락은 loud 실패 (kink 어긋남 방지).
    exp_f_scale 도 config 에서 읽어 exponential_F 가 학습 때 scale 로 reload 된다 (이전엔 50 고정 버그)."""
    # CD 인자 목록은 utility.cd_kwargs_from_config 가 단독으로 소유한다 (SSOT) —
    # 예전엔 여기서 rho/amp 만 넘겨 modified CD 체크포인트가 original 로 복원됐다.
    return make_utility_F_from_config(cfg_dict, strict_threshold=True)


def load_config(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        raw = json.load(f)
    # numpy-ify arrays the evaluator/buffer-consumers expect
    for k in ("state_mean", "state_std", "action_scale",
              "reward_min", "reward_max", "reward_scaling_factor",
              "max_racc", "min_racc", "racc_mean", "racc_std"):
        if k in raw and isinstance(raw[k], list):
            raw[k] = np.asarray(raw[k], dtype=np.float32)
    config = SimpleNamespace(**raw)
    config.utility_F = _rebuild_utility_F(raw)
    config.use_wandb = False  # never log eval-only runs to wandb
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Path to a run directory (contains config.json and model/).")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--mixture_commit", action="store_true",
                   help="혼합 성분을 **에피소드 시작에 한 번** 뽑아 끝까지 고정한다. AET 는 "
                        "비선형 목적이라 최적 정책이 정책들의 혼합이고(Zahavy et al. 2021) "
                        "매 스텝 재샘플링하는 Markov 확률정책은 준최적이다. 재학습 불필요 — "
                        "같은 체크포인트를 롤아웃 방식만 바꿔 굴린다.")
    p.add_argument("--model_subdir", type=str, default="model",
                   help="run_dir 아래 체크포인트 디렉토리. 'model_best' 로 주면 학습 중 최고 "
                        "점수 체크포인트를 같은 100ep 프로토콜로 다시 굴린다 — score_curve.csv "
                        "의 best 값은 30ep 평가 10 점 중 최대라 **선택 편향**이 있어 final 과 "
                        "같은 표에 못 쓴다.")
    p.add_argument("--z_value", type=float, default=None,
                   help="에피소드 단위 잠재 z 를 이 값으로 **고정**해 롤아웃한다 (z 조건부 "
                        "정책 전용). z=0/1 로 각각 100ep 굴린 뒤 p 로 해석적 혼합하면 "
                        "전문가 2 개 실험과 같은 규약이 된다.")
    p.add_argument("--cfg_scale", type=float, default=None,
                   help="extraction=diffusion 전용: racc guidance 스케일 λ 를 덮어써 같은 "
                        "체크포인트를 재학습 없이 다른 λ 로 굴린다. --diffusion_uncond_prob>0 "
                        "으로 학습된 모델에만 허용 (null 분기가 학습돼 있어야 한다).")
    p.add_argument("--seed", type=int, default=0,
                   help="PRNG seed for action sampling (env seeds are iter-based).")
    p.add_argument("--out_name", type=str, default=None,
                   help="Basename for saved arrays (default: returns_<N>).")
    p.add_argument(
        "--action_noise_std", type=float, default=0.0,
        help="σ of Gaussian noise injected onto the executed action at each "
             "step. 0 (default) = clean eval. Use 0.05–0.2 for distribution-"
             "shift robustness comparison (BC's narrow imitation vs DICE's "
             "wider data-supported policy).",
    )
    # 결정론 평가 (2026-11-03): 학습 중 최종 평가는 매 스텝 MoG 에서 행동을 샘플한다 (evaluate 기본값).
    # 오라클은 결정론 정책으로 재므로, 오프라인 런을 같은 조건으로 다시 재서 낙상 중 샘플링 노이즈 몫을 분리한다.
    p.add_argument("--deterministic", choices=("off", "mean", "mode"), default="off",
                   help="off = 샘플링 (기본, 학습 중 평가와 같다); mean = 혼합 평균; mode = 가중치 최대 성분의 평균.")
    p.add_argument(
        "--disable_done", action="store_true", default=False,
        help="Ignore env's `done` flag — episode always runs the full "
             "max_steps. Exposes 'drop & die' exploits that win short-horizon "
             "CD score by early termination.",
    )
    p.add_argument(
        "--score_reshape", type=str, default="none",
        help="채점 전용 reshape (예: walker2d_cd_safe_pv). 정책 입력은 학습 reshape 그대로, "
             "이 reshape 의 per-episode 합을 eval/<out>_score.npy 로 따로 저장한다 (2026-11-05, D6.24).",
    )
    args = p.parse_args()
    # z 값은 config 에 실어 evaluate 가 읽게 한다 (평가 경로 SSOT 유지).

    config = load_config(args.run_dir)
    if args.z_value is not None:
        config.z_eval_value = float(args.z_value)
    if args.cfg_scale is not None:
        if getattr(config, "extraction", None) != "diffusion":
            raise ValueError("--cfg_scale is only valid for extraction=diffusion checkpoints")
        if float(getattr(config, "diffusion_uncond_prob", 0.0)) <= 0.0 and args.cfg_scale != 1.0:
            raise ValueError("--cfg_scale != 1 needs a model trained with diffusion_uncond_prob > 0")
        config.diffusion_cfg_scale = float(args.cfg_scale)

    if config.env_name not in ENV_MAP:
        raise ValueError(f"Unknown env_name={config.env_name!r}")
    env = ENV_MAP[config.env_name]()

    # Pick the learner module that this checkpoint was trained with. Default
    # "aet" preserves behavior for older config.json files (pre-learner_type).
    learner_type = getattr(config, "learner_type", "aet")
    if learner_type not in LEARNER_MODULES:
        raise ValueError(f"Unknown learner_type={learner_type!r}; expected one of {list(LEARNER_MODULES)}")
    _learner = LEARNER_MODULES[learner_type]
    init_train_state = _learner.init_train_state
    get_model = _learner.get_model

    model_path = os.path.abspath(os.path.join(args.run_dir, args.model_subdir))
    print(f"Loading checkpoint from: {model_path}  (learner_type={learner_type})")
    train_state = load_model_cpu(model_path, config, init_train_state)
    policy = get_model(train_state.policy_state)[0]

    eval_dir = os.path.join(args.run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    z_tag = "" if args.z_value is None else f"_z{args.z_value:g}"
    z_tag += "" if args.cfg_scale is None else f"_cfg{args.cfg_scale:g}"
    perturb_tag = ""
    if args.action_noise_std > 0.0:
        perturb_tag += f" action_noise={args.action_noise_std}"
    if args.disable_done:
        perturb_tag += " disable_done"
    print(f"Rolling out {args.num_episodes} episodes on {config.env_name}{perturb_tag} ...")
    info = evaluate_GF(
        config,
        policy,
        env,
        save_dir=eval_dir,
        num_episodes=args.num_episodes,
        max_steps=config.max_seq_len,
        t_env=None,
        key=jax.random.PRNGKey(args.seed),
        reward_reshape=getattr(config, "reward_reshape", "none"),
        action_noise_std=args.action_noise_std,
        disable_done=args.disable_done,
        mixture_commit=args.mixture_commit,
        deterministic=args.deterministic != "off",
        det_mode=args.deterministic if args.deterministic != "off" else "mean",
        score_reshape=args.score_reshape,
    )

    return_array = info["return_array"]
    return_array_norm = info["return_array_normalized"]

    base = args.out_name or f"returns_{args.num_episodes}{z_tag}"
    raw_path = os.path.join(eval_dir, f"{base}.npy")
    norm_path = os.path.join(eval_dir, f"{base}_normalized.npy")
    np.save(raw_path, return_array)
    np.save(norm_path, return_array_norm)
    len_path = os.path.join(eval_dir, f"{base}_episode_lengths.npy")
    np.save(len_path, info["episode_lengths"])      # 낙상·rate CD 재계산용 (loop.py 의 final_episode_lengths 와 같은 값)
    if args.score_reshape != "none":
        score_path = os.path.join(eval_dir, f"{base}_score.npy")
        np.save(score_path, info["reshape_score_returns"])   # [N, score_dim] raw — 앞 2 채널 = 학습 reshape 와 같음
        print(f"  {score_path}  shape={info['reshape_score_returns'].shape}  ({args.score_reshape})")

    print(f"\nSaved:\n  {raw_path}  shape={return_array.shape}\n  {norm_path}  shape={return_array_norm.shape}")
    print(f"Mean raw returns : {np.mean(return_array, axis=0)}")
    print(f"Std  raw returns : {np.std(return_array, axis=0)}")
    for name, val in info.items():
        if name.startswith("score_"):
            print(f"{name}: {val:.4f}")


if __name__ == "__main__":
    main()
