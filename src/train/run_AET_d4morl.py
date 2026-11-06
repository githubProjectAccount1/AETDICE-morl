"""
Top-level sweep dispatcher — thin shim that wires the active SWEEPS list
to the dispatcher's queue/worker engine.

Run all active sweeps:
    python src/train/run_AET_d4morl.py
    python -m src.train.run_AET_d4morl

Run specific sweeps by index (0-based into SWEEPS):
    python -m src.train.run_AET_d4morl --sweeps 0 1

Dry-run (print commands without executing):
    python -m src.train.run_AET_d4morl --dry-run

Route to specific GPUs (shared box):
    python -m src.train.run_AET_d4morl --gpu_indices 2,3

Kill jobs:
    pkill -9 -f "src.train.main"     # child training processes
    pkill -9 -f "run_AET_d4morl"     # the dispatcher itself

Toggle which sweeps run: edit src/train/sweeps_active.py.
Add a new sweep definition: drop a `_build_sweep(...)` into the appropriate
src/train/sweep_defs/<env>.py.
"""
import os
import sys

# Invoked as a plain script (`python src/train/run_AET_d4morl.py`), so
# sys.path[0] is src/train/ — neither `src.` nor the repo root is importable
# by default. Walk up three levels and prepend.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.dispatcher import run
from src.train.sweeps_active import SWEEPS


if __name__ == "__main__":
    run(SWEEPS)
