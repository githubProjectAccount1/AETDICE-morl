"""MO-PointMaze-v10 — wall-free continuous point-mass, 10-objective, finite-horizon.

Production env for the training pipeline (Reviewer 2, many-objective scaling).
Prototype/design: mo-pointmaze_src/ + docs/rebuttal/reviewer2/W2_scaling/mo-pointmaze_v10/DESIGN.md.

Interface matches setup.build_env / eval.evaluate expectations:
    obj_dim, observation_space.shape, action_space.shape, action_scale,
    reset() -> obs, step(a) -> (obs, scalar_reward, done, {'obj': reward_vec}).

- M objective points on a circle (radius R); reward_i = exp(-||pos-p_i||^2 / 2 sigma^2) in (0,1].
- NO terminal state: done only at t == horizon. Box-clip boundary (no internal walls).
"""
import numpy as np

_NUM_OBJ = 10
_BOX = 2.0            # square boundary [-BOX, BOX]^2
_RADIUS = 1.6         # objective points on this circle
_SIGMA = 0.25         # Gaussian reward width (sharp enough that neighbor tails ~0 over H)
_HORIZON = 200
_DT = 0.1
_DAMPING = 0.1
_MAX_SPEED = 2.0
_START = np.zeros(2, dtype=np.float64)


class _Space:
    """Minimal gym.spaces.Box stand-in exposing `.shape` (build_env reads shape[0])."""
    def __init__(self, dim: int):
        self.shape = (dim,)


def _circle_positions(num_obj: int, radius: float) -> np.ndarray:
    ang = 2.0 * np.pi * np.arange(num_obj) / num_obj
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)  # (M, 2)


class MOPointMazeV10Env:
    """Finite-horizon MO point-mass. Objective vector via info['obj'] (raw, in (0,1])."""

    def __init__(self):
        self.obj_dim = _NUM_OBJ
        self.obj_pos = _circle_positions(_NUM_OBJ, _RADIUS)   # (M, 2)
        self.horizon = _HORIZON
        self.observation_space = _Space(4)                    # (pos_x, pos_y, vel_x, vel_y)
        self.action_space = _Space(2)                         # force (ax, ay)
        self.action_scale = np.ones(2, dtype=np.float32)
        self._pos = None
        self._vel = None
        self._t = 0

    # --- gym-style API ---
    def seed(self, seed=None):
        """gym-compat: eval (evaluate.py) calls env.seed(iter). Env dynamics are
        deterministic (fixed start, no process noise) — episode variety comes from
        the stochastic policy's action sampling — so this just stores an RNG."""
        self._np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self._pos = _START.copy()
        self._vel = np.zeros(2, dtype=np.float64)
        self._t = 0
        return self._obs()

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)   # force ∈ [-1,1]^2
        assert a.shape == (2,), f"action shape {a.shape} != (2,)"
        self._vel = (1.0 - _DAMPING) * self._vel + a * _DT
        self._vel = np.clip(self._vel, -_MAX_SPEED, _MAX_SPEED)
        self._pos = np.clip(self._pos + self._vel * _DT, -_BOX, _BOX)  # walls X → box-clip
        obj = self._reward_vec()                                  # (M,) raw per-step reward
        self._t += 1
        done = self._t >= self.horizon                            # NO terminal; horizon only
        return self._obs(), float(obj.sum()), done, {"obj": obj}

    # --- internals ---
    def _obs(self):
        return np.concatenate([self._pos, self._vel]).astype(np.float32)

    def _reward_vec(self):
        d2 = np.sum((self.obj_pos - self._pos[None, :]) ** 2, axis=1)  # (M,)
        return np.exp(-d2 / (2.0 * _SIGMA ** 2)).astype(np.float32)    # (M,) in (0,1]
