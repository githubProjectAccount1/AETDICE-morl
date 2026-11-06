"""D4MORL MO-Ant 데이터셋 궤적의 torso xy 경로 재구성 (2026-11-06, Fig 5a 데이터셋 coverage 층).

왜: 데이터셋 observation 은 qpos[2:] + qvel (environment/ant.py _get_obs) 이라 절대 x, y 가 없다.
방법: environment/ant.py 의 보상 정의를 그대로 뒤집는다 —
    obj_k = Δ{x,y}/dt + (1 - 0.5‖clip(a, -1, 1)‖²)   (k = 0: x, 1: y; dt = 0.01 × frame_skip 5 = 0.05)
    ⇒ Δ{x,y}_t = (raw_rewards[t, k] - 1 + 0.5‖clip(a_t)‖²) · dt,   xy_0 = (0, 0)  (reset 잡음 ±1e-3 무시)
검증 (2026-11-06, 무작위 8 궤적): 데이터셋 상태(qpos[2:], qvel; x, y = 0)를 env 에 넣고 저장된 action 으로 한 스텝
돌리면 next_observation 이 1e-13 까지 일치(같은 env) — 그 스텝의 실제 Δxy 와 위 식의 차이는 스텝당 ≤ 5e-3 m,
500 스텝 누적 끝점 차이 ≤ 0.01 m (그림 픽셀 ~0.12 m). qvel·dt 적분은 끝점이 ~0.15 m 어긋나 보상 역산을 쓴다.

사용: source scripts/aetenv.sh && python -m src.viz.moant_dataset_xy \
      --pkl data/MO-Ant-v2_50000_expert_uniform.pkl --out <scratch>/moant_ds_xy.npz
산출 npz: xy [N, T+1, 2] float32 (스텝 전 위치 T 개 + 마지막 스텝 후), lengths [N] (= T+1), preference [N, 2].
spaghetti_from_npz --coverage_npz 가 읽는다.
"""
import argparse
import os
import pickle

import numpy as np

DT = 0.05            # environment/ant.py: timestep 0.01 (assets/ant.xml) × frame_skip 5
SURVIVE = 1.0        # environment/ant.py survive_reward
CTRL_COEF = 0.5      # environment/ant.py ctrl_cost = 0.5 * ‖a‖²


def step_dxy(raw_rewards, actions, dt=DT):
    """[T, 2] raw_rewards, [T, 8] actions → [T, 2] 스텝별 torso 변위 (보상 역산)."""
    ctrl = CTRL_COEF * np.square(np.clip(actions, -1.0, 1.0)).sum(-1, keepdims=True)
    return (np.asarray(raw_rewards, np.float64) - SURVIVE + ctrl) * dt


def trajectory_xy(traj, dt=DT):
    """D4MORL 궤적 dict → [T+1, 2] xy (원점 출발)."""
    d = step_dxy(traj["raw_rewards"], traj["actions"], dt)
    return np.concatenate([np.zeros((1, 2)), np.cumsum(d, axis=0)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        D = pickle.load(f)
    L = np.array([len(t["raw_rewards"]) + 1 for t in D], dtype=np.int64)
    xy = np.full((len(D), int(L.max()), 2), np.nan, np.float32)
    for i, t in enumerate(D):
        xy[i, :L[i]] = trajectory_xy(t)
    pref = np.stack([np.asarray(t["preference"])[0] for t in D]).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, xy=xy, lengths=L, preference=pref)
    end = xy[np.arange(len(D)), L - 1]
    print(f"Saved {args.out}: {len(D)} trajectories, T+1 = {L.max()}, "
          f"end x [{end[:, 0].min():.1f}, {end[:, 0].max():.1f}] y [{end[:, 1].min():.1f}, {end[:, 1].max():.1f}]")


if __name__ == "__main__":
    main()
