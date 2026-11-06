"""Shared boilerplate for the src/learners/ modules (SSOT for mechanical scaffolding).

Every learner module previously carried byte-identical copies of the state
namedtuples, `get_model`, the cosine-Adam optimizer recipe, `_mu_is_learnable`,
and the orbax checkpoint I/O helpers. Those copies are the single source of
truth here now; each learner imports from this module.

This module holds ONLY mechanical boilerplate — no DICE `e`/`w` math and no
ν/μ dual bodies live here (those stay per-learner on purpose).

Public surface preserved: importing these names into a learner's namespace keeps
`<learner>.get_model`, `<learner>.save_model`, etc. resolvable for the runner
and eval code (which access them as module attributes).
"""
import os
from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as orbax


# ── State containers (shared by AET / FairDICE / BC) ──
# IQL has a DIFFERENT 5-field TrainState (q_state / v_state) defined in its own
# module, but still reuses NetworkState / Model from here.
NetworkState = namedtuple("NetworkState", ["graphdef", "state", "target_params"])
TrainState = namedtuple("TrainState", ["policy_state", "nu_state", "mu_state", "step"])
Model = namedtuple("Model", ["network", "optimizer", "target_network"])


def _mu_is_learnable(config) -> bool:
    # linear G → mu fixed to preference weights. concave G → mu is a learnable dual variable.
    from src.utility import LEARNABLE_G_TYPES   # SSOT (2026-10-25: weighted_log / isoelastic* 추가)
    return config.utility_G_type in LEARNABLE_G_TYPES


def get_model(state: NetworkState) -> Model:
    """Merge (graphdef, state) into a live (network, optimizer) pair plus an
    optional target network. None-safe: when `target_params is None` (IQL's
    v/policy/mu slots), the target network is None instead of being rebuilt.
    For learners whose `target_params` is always a filtered Param state
    (AET / FairDICE / BC), the guard is never taken, so behavior is identical
    to the pre-refactor per-module `get_model`."""
    network, optimizer = nnx.merge(state.graphdef, state.state)
    target_network = None
    if state.target_params is not None:
        _, other_variables = state.state.split(nnx.Param, ...)
        target_network, _ = nnx.merge(state.graphdef, state.target_params, other_variables)
    return Model(network, optimizer, target_network)


def make_cosine_adam_tx(lr, total_steps):
    """Cosine-decayed Adam optimizer chain — the un-clipped recipe every learner uses
    (matches the original pre-refactor runs that produced the paper's Table 4/5/6).
    Gradient clipping was removed (2026-09-02): the always-on clip introduced during the
    refactor lowered AET-DICE returns, and an opt-in flag was too error-prone to keep."""
    return optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            optax.cosine_decay_schedule(-lr, total_steps, alpha=1e-6 / lr)
        ),
    )


# ── Checkpoint I/O ────────────────────────────────────────────────────────────
# save_model / save_checkpoint are init-agnostic and imported directly.
# load_model / load_checkpoint need the caller's `init_train_state` to build the
# restore template, so each learner passes its own via a thin wrapper.
def save_model(train_state, path: str, force: bool = False):
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(path, train_state, force=force)


def load_model(path: str, config, init_train_state):
    checkpointer = orbax.PyTreeCheckpointer()
    train_state = init_train_state(config)
    train_state = checkpointer.restore(path, item=train_state)
    return train_state


def save_checkpoint(train_state, key, save_dir: str):
    """Persist train_state + JAX prng key for crash-recovery.
    Overwrites any prior checkpoint. Orbax save runs first so a failed save
    never leaves a stale key sidecar pointing at a now-corrupt model."""
    model_path = os.path.abspath(os.path.join(save_dir, "model"))
    save_model(train_state, model_path, force=True)
    key_path = os.path.abspath(os.path.join(save_dir, "resume_key.npy"))
    np.save(key_path, np.asarray(key))


def load_checkpoint(save_dir: str, config, init_train_state):
    """Restore a checkpoint written by save_checkpoint.
    Returns (train_state, key, next_iter). next_iter is derived from
    train_state.step // log_interval so it can't drift out of sync with the
    restored model's actual step count."""
    model_path = os.path.abspath(os.path.join(save_dir, "model"))
    key_path = os.path.abspath(os.path.join(save_dir, "resume_key.npy"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No orbax checkpoint at {model_path}")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"No resume_key.npy at {key_path}")
    train_state = load_model(model_path, config, init_train_state)
    key = jnp.asarray(np.load(key_path))
    next_iter = int(train_state.step) // config.log_interval
    return train_state, key, next_iter
