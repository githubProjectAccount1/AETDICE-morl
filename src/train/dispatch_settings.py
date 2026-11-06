"""Dispatcher-level training defaults.

These constants drive `run_job`'s subprocess command line AND `_build_sweep`'s
default axes. They're isolated here so neither the job-runner nor the sweep
builder needs to import from `run_AET_d4morl.py` — avoids the circular import
that would otherwise arise once sweep definitions live under `sweep_defs/`.

Edit a constant here = behavior change for every job dispatched. Edit a flag
inside `_build_sweep(...)` = behavior change for ONE sweep entry only.
"""
import os

# ── Training args shared across every dispatched job ──────────────────────
DIVERGENCE        = "CHI"
NUM_LAYERS        = 4
EVAL_EPISODES     = 30
BATCH_SIZE        = 512
# 환경변수로 덮을 수 있다 — 같은 런을 더 돌려 **연장**할 때 쓴다 (예: 1M 완주 후
# AET_TOTAL_TRAIN_STEPS=2000000 으로 재실행하면 체크포인트에서 이어 2M 까지 간다).
# 연장하려면 eval/final_returns.npy 를 먼저 치워야 한다 — 그 파일이 'done' 표식이라
# 디스패처가 SKIP 한다 (job_runner._resume_status 참조).
TOTAL_TRAIN_STEPS = int(os.environ.get("AET_TOTAL_TRAIN_STEPS", 1_000_000))
LOG_INTERVAL      = 100_000
HIDDEN_DIM        = 512
USE_WANDB         = True

# Per-learner override for total_train_steps. BC converges much faster than
# the DICE methods (no Bellman, no dual variable — just MLE on filtered (s,a)
# pairs), so 200K is more than enough; 1M would just overfit to the top-K%
# subset. Other learners default to TOTAL_TRAIN_STEPS via .get(...) in run_job.
LEARNER_TRAIN_STEPS = {
    "bc": 200_000,
}

# Default eval-rollout cap and policy/critic time-embedding range. Sweepable
# per job via the `max_seq_lens` axis on `_build_sweep`. Caveat: this knob does
# NOT truncate the training dataset — it only caps eval rollouts and sets
# T_max for the sinusoidal time embedding inside policy/critic. Buffer-side
# truncation is a separate change (see buffer.load_d4morl_dataset).
MAX_SEQ_LEN = 500

# ── System / performance config (no effect on training outcome) ───────────
# Number of train steps fused into one jax.lax.scan call. Larger = less Python
# dispatch overhead (better GPU util), more device memory per chunk. Must
# divide LOG_INTERVAL. Default 5000 was chosen empirically for the active sweeps;
# tune up for low GPU util, down for tight device memory.
CHUNK_SIZE = 5000
assert LOG_INTERVAL % CHUNK_SIZE == 0, (
    f"LOG_INTERVAL ({LOG_INTERVAL}) must be divisible by CHUNK_SIZE ({CHUNK_SIZE})"
)

# How often to eval, as a multiple of LOG_INTERVAL. 1 = every log window;
# 2 = every other (so every 200K with LOG_INTERVAL=100K). Final log window
# always evals regardless, since final_returns.npy is written from the last eval.
EVAL_EVERY_N_LOGS = 1

# ── Naming constants (consumed by run_naming.py helpers + run_job) ────────
# Shared across every sweep. Each sweep dict no longer carries save_path /
# project / tag — they're computed per-job from these constants + the job's
# env + dataset.
# 환경변수로 덮을 수 있다 — **탐색용 런을 본 결과 트리에서 격리**할 때 쓴다.
# 왜 (2026-09-17): HP 스윕의 (musf=1e4, β=1e-6, seed1) 셀은 경로가 본 런과 완전히 같아서,
# 그대로 띄우면 살아 있는 런의 디렉토리에 'resume' 로 붙어 no-op 최종 eval 을 덮어쓴다.
# 또 5 월 stale 런이 남아 있는 셀은 'done' 으로 조용히 SKIP 된다. 트리를 갈라 둘 다 막는다.
# base dir; env_dataset 은 job 마다 뒤에 붙는다.
SAVE_PATH = os.environ.get("AET_SAVE_PATH", "./results_Transformed")
PROJECT_PREFIX = "tfd4morl"   # wandb project: <prefix>_<env>_<dataset>_<VERSION>
TAG_PREFIX     = "tfd4morl"   # run tag:       <prefix>_<env>_<dataset>
VERSION        = "v0501"      # bump to isolate a batch of runs from prior results
