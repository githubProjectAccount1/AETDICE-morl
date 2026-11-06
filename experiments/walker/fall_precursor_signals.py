"""어떤 스텝 신호가 Walker2d 낙상을 가장 잘 예고하는가 (2026-11-06, D6.36).

사용자: "그럼 넘어지는것과 바로 관련되는 리워드 항들이라던가 페널티 항들은 없을까? 일단 가장 직접적인게
posture라서 쓰는건데." / "분석 돌려봐".

측정:
  롤아웃 = cd_fall_support.rollout (학습 최종 평가와 같은 경로·키, 스텝별 raw 관측 기록) — 복사 금지 SSOT.
  관측 obs = [qpos[1:] (8), clip(qvel) (9)] → z = obs[0], θ = obs[1], ż = obs[9], θ̇ = obs[10].
  신호 (클수록 위험하도록 부호를 맞춘다):
    -S0        자세 (src _walker2d_quadratic_safety, 논문 S0)
    -margin    종료 경계까지 여유 (src _walker2d_boundary_margin, S2a 의 clip 전 값)
    |θ̇|        몸통 각속도
    -ż         아래로 떨어지는 속도
    ttb        경계까지 남은 시간 근사 = −(margin_θ / (|θ̇|·dt + ε)) (각도 경계 기준, 작을수록 위험 → 부호 반전)
  라벨 = 그 스텝 뒤 k 스텝 안에 낙상 (k = 10, 25, 50). 낙상 안 한 에피소드의 스텝은 전부 0.
  지표 = ROC AUC (스텝 단위). 낙상 직전 마지막 상태에서 높이·각도 중 어느 경계가 더 가까웠는지도 센다.
사용: source scripts/aetenv.sh && JAX_PLATFORMS=cpu python -m experiments.walker.fall_precursor_signals --runs <run_dir> ...
"""
import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.walker.cd_fall_support import rollout
from src.data.reward_transform import (WALKER2D_ANGLE_BOUND, WALKER2D_HEIGHT_BOUNDS, _WALKER2D_HEIGHT_HALF,
                                       _walker2d_boundary_margin, _walker2d_quadratic_safety)

_DT = 0.008
_KS = (10, 25, 50)
_T = 500


def signals(obs):
    z, th, zd, thd = obs[:, 0], obs[:, 1], obs[:, 9], obs[:, 10]
    lo, hi = WALKER2D_HEIGHT_BOUNDS
    m_h = np.minimum(z - lo, hi - z) / _WALKER2D_HEIGHT_HALF
    m_a = 1.0 - np.abs(th) / WALKER2D_ANGLE_BOUND
    margin = _walker2d_boundary_margin(z, th)
    return {"-S0": -_walker2d_quadratic_safety(z, th), "-margin": -margin, "-margin_h": -m_h, "-margin_angle": -m_a,
            "|θ̇|": np.abs(thd), "-ż": -zd, "ttb_angle": -(m_a / (np.abs(thd) * _DT + 1e-6))}, m_h, m_a


def analyse(run_dir):
    cfg, sv, _, _, lens, same = rollout(run_dir)
    sig, m_h, m_a = signals(sv)
    ends = np.cumsum(lens); starts = ends - lens
    fell = lens < _T
    out = {"run": run_dir, "n_ep": int(len(lens)), "fall_rate": float(fell.mean()), "reproduces_final_eval": same,
           "auc": {}, "last_state_nearest_boundary": {}}
    for k in _KS:
        y = np.zeros(len(sv), bool)
        for s, e, f in zip(starts, ends, fell):
            if f:
                y[max(s, e - k):e] = True
        out["auc"][k] = ({n: float(roc_auc_score(y, v)) for n, v in sig.items()} if 0 < y.sum() < len(y) else None)
    last = ends[fell] - 1
    if len(last):
        out["last_state_nearest_boundary"] = {"height": int((m_h[last] < m_a[last]).sum()),
                                              "angle": int((m_a[last] <= m_h[last]).sum())}
        out["last_state_mean"] = {"margin_h": float(m_h[last].mean()), "margin_angle": float(m_a[last].mean()),
                                  "S0": float(_walker2d_quadratic_safety(sv[last, 0], sv[last, 1]).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = []
    for r in args.runs:
        o = analyse(r); res.append(o)
        print(json.dumps({k: o[k] for k in ("run", "fall_rate", "reproduces_final_eval", "last_state_nearest_boundary")},
                         ensure_ascii=False), flush=True)
        for k, a in o["auc"].items():
            if a:
                print(f"  k={k}: " + "  ".join(f"{n} {v:.3f}" for n, v in sorted(a.items(), key=lambda x: -x[1])), flush=True)
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
