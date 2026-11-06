"""Config → JSON serialization helpers.

`comparer` (and any downstream analysis) reads `<run_dir>/config.json` to
identify runs without parsing filenames. Keep this module small and
side-effect-free — the only public entry is `dump_config_json(config, dir)`.
"""
import json
import os

import numpy as np

from src.utility import Utility


def _json_safe(v):
    """Recursively convert a value to a JSON-serializable form.
    Unknown objects fall back to a "<ClassName>" stub (identifiable, lossy)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    # Utility instances: record their configuration so comparer can tell them apart.
    if isinstance(v, Utility):
        return {"__class__": "Utility", "type": v.type,
                "p": getattr(v, "p", None)}
    return f"<{type(v).__name__}>"


def config_as_json(config) -> str:
    """One key per line for scannability; each value rendered compactly so long
    lists don't blow up across many lines (which would be noisy in terminals
    and in diffs of config.json)."""
    items = {k: _json_safe(v) for k, v in vars(config).items()}
    lines = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in items.items()]
    return "{\n" + ",\n".join(lines) + "\n}"


def dump_config_json(config, save_dir):
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        f.write(config_as_json(config))
