"""Multi-GPU sweep dispatcher.

Takes a list of sweep dicts → cartesian-expands them into per-job dicts →
distributes across a worker thread per GPU. Each worker pulls jobs off a
shared queue and runs them via `job_runner.run_job` (one subprocess per job).

Public:
    run(sweeps)   # CLI-driven entrypoint; called from run_AET_d4morl.py
"""
import argparse
import itertools
import queue
import sys
import threading
from datetime import datetime

from src.train.dispatch_settings import DIVERGENCE, MAX_SEQ_LEN
from src.train.job_runner import get_num_gpus, run_job

# ── Sweep axis schema ────────────────────────────────────────────────────
# Axis field name (in a sweep dict) → per-job field name (singular).
_AXIS_KEYS = {
    "entropy_levels":         "entropy_level",
    "seeds":                  "seed",
    "envs":                   "env",
    "learner_types":          "learner_type",
    "utility_G_types":        "utility_G_type",
    "utility_F_types":        "utility_F_type",
    "power_ps":               "power_p",
    "exp_f_scales":           "exp_f_scale",
    "pref_weights_list":      "pref_weights",
    "reward_scaling_factors": "reward_scaling_factor",
    "mu_scaling_factor":      "mu_scaling_factor",
    "reward_reshapes":        "reward_reshape",
    "betas":                  "beta",
    "bc_conditionings":       "bc_conditioning",
    "racc_encodings":         "racc_encoding",
    "z_keeplists":            "z_keeplist",
    "cd_filter_kinds":        "cd_filter_kind",
    "cobbdouglas_rhos":       "cobbdouglas_rho",
    "max_seq_lens":           "max_seq_len",
    "reward_thresholds":      "reward_threshold",
    "cobbdouglas_amps":       "cobbdouglas_amp",
    "cd_filter_rhos":         "cd_filter_rho",
    "cd_filter_k_pcts":       "cd_filter_k_pct",
    "divergences":            "divergence",
    "w_maxes":                "w_max",
    "nu_output_transforms":   "nu_output_transform",
    "temperatures":           "temperature",
    "expectiles":             "expectile",
    "beta_schedules":         "beta_schedule",
    "beta_inits":             "beta_init",
    "beta_finals":            "beta_final",
    "beta_warmup_steps_list": "beta_warmup_steps",
    "standardize_e_list":     "standardize_e",
    "traj_filter_paths":      "traj_filter_path",
    "transition_filter_paths": "transition_filter_path",
    "global_mixtures":        "global_mixture",
    "extractions":            "extraction",
    "wbc_weights":            "wbc_weight",
    "traj_mixtures":          "traj_mixture",
}

_METADATA_KEYS = ("name", "dataset", "total_train_steps")

# Axes that may be omitted from hand-rolled sweep dicts (e.g. older walker2d /
# hopper entries written before the axis existed). Each missing axis is filled
# with this default; absence here means the axis is required and a KeyError is
# the right signal.
_AXIS_DEFAULTS = {
    "learner_types":          ["aet"],
    "reward_reshapes":        ["none"],
    "bc_conditionings":       ["state"],
    "cd_filter_kinds":        ["cd"],
    "cobbdouglas_rhos":       [0.5],
    "max_seq_lens":           [MAX_SEQ_LEN],
    "reward_thresholds":      [None],
    "cobbdouglas_amps":       [1.0],
    "cd_filter_rhos":         [None],
    "cd_filter_k_pcts":       [None],
    "divergences":            [DIVERGENCE],
    "w_maxes":                [float("inf")],
    "nu_output_transforms":   ["none"],
    "temperatures":           [None],
    "expectiles":             [None],
    "beta_schedules":         ["constant"],
    "beta_inits":             [None],
    "beta_finals":            [None],
    "beta_warmup_steps_list": [0],
    "standardize_e_list":     [False],
    "traj_filter_paths":      [None],
    "transition_filter_paths": [None],
    "global_mixtures":        [False],
    "extractions":            ["original"],
    "wbc_weights":            ["raw"],
    "traj_mixtures":          [False],
}


def sweep_to_jobs(sweep: dict) -> list:
    """Expand a sweep dict into a list of per-job dicts via cartesian product
    over every axis-list. Sweep-level metadata is attached to every job."""
    axis_values = [sweep.get(ax, _AXIS_DEFAULTS.get(ax)) for ax in _AXIS_KEYS]
    missing = [ax for ax, v in zip(_AXIS_KEYS, axis_values) if v is None]
    if missing:
        raise KeyError(
            f"Sweep '{sweep.get('name','<unnamed>')}' missing required axis(es): {missing}"
        )
    job_keys = list(_AXIS_KEYS.values())
    metadata = {k: sweep[k] for k in _METADATA_KEYS if k in sweep}
    jobs = []
    for combo in itertools.product(*axis_values):
        job = dict(zip(job_keys, combo))
        job.update(metadata)
        jobs.append(job)
    return jobs


def _worker(gpu: int, job_queue: queue.Queue, failed_jobs: list,
            lock: threading.Lock, completed: list, total: int,
            dry_run: bool) -> None:
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break
        success = run_job(gpu, job, dry_run=dry_run)
        if not success:
            with lock:
                failed_jobs.append(job)
        with lock:
            completed.append(1)
            print(f"  Progress: {len(completed)}/{total} done", flush=True)
        job_queue.task_done()


def run_all_jobs(jobs: list, gpu_indices, dry_run: bool = False) -> list:
    """Run all jobs across a shared GPU pool. Returns the list of failed jobs.

    `gpu_indices` is a list of physical GPU indices to use (one worker per
    index). All workers consume from a single queue, so a GPU that finishes
    its job early picks up the next available one instead of idling."""
    print(f"\n{'='*60}", flush=True)
    print(f"  Total jobs={len(jobs)}  |  GPUs={gpu_indices}"
          + ("  [DRY-RUN]" if dry_run else ""), flush=True)
    print(f"{'='*60}\n", flush=True)

    job_queue: queue.Queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    failed_jobs: list = []
    completed:   list = []
    lock = threading.Lock()

    threads = [
        threading.Thread(
            target=_worker,
            args=(gpu, job_queue, failed_jobs, lock, completed, len(jobs), dry_run),
            daemon=True,
        )
        for gpu in gpu_indices
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return failed_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transformed AETDICE sweeps (d4morl).")
    parser.add_argument(
        "--sweeps", nargs="+", type=int, metavar="IDX",
        help="0-based indices of sweeps to run (default: all)",
    )
    parser.add_argument(
        "--gpus", type=int, default=None,
        help="Number of GPUs to use (default: auto-detect via nvidia-smi). "
             "Picks indices 0..N-1. To target specific GPUs, use --gpu_indices.",
    )
    parser.add_argument(
        "--gpu_indices", type=str, default=None,
        help="Comma-separated list of physical GPU indices to use, e.g. '2,3'. "
             "Overrides --gpus. Use when sharing the box with other experiments to "
             "route around busy GPUs instead of stepping on them.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def run(sweeps: list) -> None:
    """CLI entrypoint. Parses args, filters sweeps, dispatches, prints summary,
    `sys.exit(1)` on any failure."""
    args = parse_args()
    if args.gpu_indices is not None:
        gpu_indices = [int(x) for x in args.gpu_indices.split(",") if x.strip()]
    else:
        num_gpus = args.gpus if args.gpus is not None else get_num_gpus()
        gpu_indices = list(range(num_gpus))
    dry_run = args.dry_run

    sweeps_to_run = [sweeps[i] for i in args.sweeps] if args.sweeps else sweeps

    all_jobs: list = []
    for sweep in sweeps_to_run:
        sweep_jobs = sweep_to_jobs(sweep)
        print(f"  Sweep '{sweep.get('name', '(unnamed)')}' → {len(sweep_jobs)} jobs", flush=True)
        all_jobs.extend(sweep_jobs)

    all_failed = run_all_jobs(all_jobs, gpu_indices, dry_run=dry_run)

    print(f"\n{'='*60}", flush=True)
    if all_failed:
        print(f"  WARNING: {len(all_failed)} job(s) failed:", flush=True)
        for job in all_failed:
            print(f"    {job}", flush=True)
    else:
        print("  All sweeps completed successfully.", flush=True)
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*60}", flush=True)

    if all_failed:
        sys.exit(1)
