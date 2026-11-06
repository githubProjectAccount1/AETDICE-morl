"""Train loop + finalization for the train entrypoint.

`train_loop` runs the chunked-scan training for-loop with eval gating, ckpt
write, and wandb log. `finalize` writes the final model + eval npy files.
Both are called from `main.py` after `setup.py` has built env/buffer/state.
"""
import csv
import json
import os

import jax
import numpy as np
import wandb
from tqdm import tqdm

from src.eval.evaluate import evaluate as evaluate_GF


# 중간 평가에서 가장 좋았던 정책을 따로 남기기 위한 장치.
#
# 왜 (사용자 지시 2026-09-17 "overfit 이 일어나고 있는지 확인할 겸 매 100k 마다 롤아웃"):
# `save_checkpoint` 는 매 로그 창마다 **같은 model/ 을 덮어쓴다** (crash-recovery 용도).
# 그래서 학습 곡선이 중간에서 정점을 찍고 내려와도 그 시점 정책을 되찾을 수 없었다.
# model_best/ 를 따로 쓰면 되찾을 수 있고, score_curve.csv 로 곡선 자체도 남는다.
#
# ★ 주의: 이것은 **online rollout 점수로 고르는 model selection** 이다. 순수 offline
#   프로토콜이 아니므로, 표에 쓸 때는 final 인지 best-checkpoint 인지 반드시 밝힌다.
_BEST_SCORE_KEY = "score_G-log_F-exp"          # AET 지표 (G=plog × F=exp).
# ESR 계열(score_linear-*)은 preference 별로 `_w{...}` 접미사가 붙어 키가 고정이 아니다.
# 곡선에는 접미사 없는 두 지표만 쓰고, 수익 수준은 정규화 return 평균으로 따로 적는다.
_CURVE_SCORES = (_BEST_SCORE_KEY, "score_FairDICE")   # AET, SER
_CURVE_COLS = ("step",) + _CURVE_SCORES + ("R0_mean", "R1_mean")


def _eval_curve_row(save_dir, step, eval_info):
    """중간 평가 점수를 eval/score_curve.csv 에 한 줄 덧붙인다 (overfit 진단용 곡선)."""
    path = os.path.join(save_dir, "eval", "score_curve.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(_CURVE_COLS)
        Rn = np.asarray(eval_info.get("return_array_normalized", np.empty((0, 2))))
        means = Rn.mean(0) if Rn.size else []
        w.writerow([step] + [eval_info.get(c, "") for c in _CURVE_SCORES]
                   + [f"{m:.2f}" for m in means])


def _load_best(save_dir):
    """resume 시 이전 best 를 되살린다 — 안 하면 재개 후 첫 평가가 무조건 best 가 된다."""
    path = os.path.join(save_dir, "eval", "best_meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _update_best(learner, train_state, eval_info, save_dir, step, best):
    score = eval_info.get(_BEST_SCORE_KEY)
    if score is None or not np.isfinite(score):
        return best
    if best is not None and float(score) <= float(best["score"]):
        return best
    learner.save_model(train_state, os.path.abspath(os.path.join(save_dir, "model_best")),
                       force=True)
    best = {"score": float(score), "step": int(step), "metric": _BEST_SCORE_KEY}
    with open(os.path.join(save_dir, "eval", "best_meta.json"), "w") as f:
        json.dump(best, f, indent=1)
    return best


def train_loop(config, env, buffer, train_state, key, save_dir, run_name,
               start_iter, learner):
    """Run the training for-loop. Returns (train_state, eval_info).

    Resume edge case: when start_iter == train_iterations on entry, the
    for-loop is a no-op and `eval_info` is None. We still need a populated
    `eval_info` for `finalize` to write final_returns.npy — so we run one
    final eval below using the restored train_state."""
    # ── chunked-scan: fuse `chunk_size` train steps into one compiled call ──
    # The chunk arrives as a Batch with each leaf shaped [chunk_size, batch_size, ...].
    # Scan iterates over the leading axis. Key splitting happens inside the scan body
    # (cheap on-device) so we only thread one carry-key per call. `config` is closed
    # over as static (its values become XLA constants in the compiled program).
    train_step = learner.train_step
    get_model = learner.get_model
    save_checkpoint = learner.save_checkpoint

    @jax.jit
    def run_chunk(train_state, key, chunk):
        def body(carry, batch_step):
            train_state, key = carry
            key, subkey = jax.random.split(key)
            train_state, info = train_step(config, train_state, batch_step, subkey)
            return (train_state, key), info
        (train_state, key), info_stacked = jax.lax.scan(body, (train_state, key), chunk)
        return train_state, key, info_stacked

    chunks_per_log = config.log_interval // config.chunk_size
    train_iterations = config.total_train_steps // config.log_interval

    print(f"Run name: {run_name}")
    # Pre-init so the post-loop final-eval write is safe when the loop is a
    # no-op (start_iter == train_iterations on resume).
    eval_info = None
    best = _load_best(save_dir)
    for iter in tqdm(range(start_iter, train_iterations), desc="Training ...",
                     initial=start_iter, total=train_iterations):
        step = (iter + 1) * config.log_interval
        update_info = None
        for _ in range(chunks_per_log):
            chunk = buffer.sample_chunk(config.chunk_size, config.batch_size)
            train_state, key, update_info = run_chunk(train_state, key, chunk)

        # Persist progress immediately after the train chunks. Placed BEFORE
        # eval/log so a crashing eval or wandb hiccup never loses a completed
        # log window.
        save_checkpoint(train_state, key, save_dir)

        # Eval gating: always on the final log window (final_returns.npy
        # depends on eval_info being populated); otherwise every
        # `eval_every_n_logs` windows.
        is_final = (iter + 1) == train_iterations
        should_eval = is_final or ((iter + 1) % config.eval_every_n_logs == 0)
        if should_eval:
            policy = get_model(train_state.policy_state)[0]
            # Heavier eval at the final iter; mid-training evals stay light to
            # keep the loop fast.
            num_eps = config.final_eval_episodes if is_final else config.eval_episodes
            eval_info = evaluate_GF(
                config,
                policy,
                env,
                save_dir + "/eval",
                num_episodes=num_eps,
                max_steps=config.max_seq_len,
                t_env=step,
                reward_reshape=config.reward_reshape,
                eval_batch=getattr(config, "eval_batch", 0),
            )
            _eval_curve_row(save_dir, step, eval_info)
            best = _update_best(learner, train_state, eval_info, save_dir, step, best)

        # update_info is scan-stacked (leading axis = chunk_size of the LAST
        # chunk in this log window). [-1] picks the very last train step's
        # values for logging.
        for k, value in update_info.items():
            if "loss" in k or "grad" in k or "debug" in k:
                if config.use_wandb:
                    wandb.log({f"{k}": value[-1]}, step=step)
                else:
                    print(f"Step: {step}, {k}: {value[-1]}")
            else:
                for i in range(config.reward_dim):
                    if config.use_wandb:
                        wandb.log({f"{k}_{i}": value[-1][i]}, step=step)
                    else:
                        print(f"Step: {step}, {k}_{i}: {value[-1][i]}")

    # Resume edge case: the for-loop above didn't run because start_iter ==
    # train_iterations (training was done previously, but final_returns.npy
    # never got written — e.g. the prior run's wandb finalize crashed after
    # save_checkpoint). Run one final eval now using the restored train_state.
    if eval_info is None:
        print(f"  Loop was a no-op (start_iter={start_iter} == "
              f"train_iterations={train_iterations}). Running one final "
              f"eval to populate final_returns.npy ...")
        policy = get_model(train_state.policy_state)[0]
        eval_info = evaluate_GF(
            config,
            policy,
            env,
            save_dir + "/eval",
            num_episodes=config.final_eval_episodes,
            max_steps=config.max_seq_len,
            t_env=train_iterations * config.log_interval,
            reward_reshape=config.reward_reshape,
            eval_batch=getattr(config, "eval_batch", 0),
        )

    if best is not None:
        print(f"  [best-ckpt] {_BEST_SCORE_KEY}={best['score']:.4f} @ step {best['step']:,} "
              f"→ {os.path.join(save_dir, 'model_best')}")
    return train_state, eval_info


def finalize(config, buffer, train_state, eval_info, save_dir, learner):
    """Close the buffer prefetch thread, write the final model + eval artifacts.

    save_checkpoint() inside the loop already wrote model/ + resume_key.npy for
    the final iter; save_model here is a safe idempotent overwrite (also
    covers the edge case of a run that completes 0 iters)."""
    if config.use_wandb:
        wandb.finish()
    # Explicit close() avoids the JAX-teardown-in-thread race at interpreter
    # shutdown. daemon=True on the prefetch thread is a backstop, not a
    # substitute.
    buffer.close()
    if not config.save_path:
        return
    learner.save_model(train_state, os.path.abspath(save_dir + "/model"), force=True)
    return_array = eval_info["return_array"]
    np.save(os.path.abspath(save_dir + "/eval/final_returns.npy"), return_array)
    # Normalized returns — what the 4 training-side scores operate on.
    # Comparer recomputes scores from these to stay robust to score-formula
    # changes.
    return_array_norm = eval_info["return_array_normalized"]
    np.save(os.path.abspath(save_dir + "/eval/final_returns_normalized.npy"),
            return_array_norm)
    # 에피소드 길이 — 조기 종료(Walker 낙상 등)와 rate 형태 CD 점수 재계산용 (2026-10-24).
    np.save(os.path.abspath(save_dir + "/eval/final_episode_lengths.npy"),
            eval_info["episode_lengths"])
