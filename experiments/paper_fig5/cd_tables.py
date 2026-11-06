"""논문 Section 6 / Appendix 의 Cobb-Douglas · safe-CD 표 수치 (2026-11-05, D6.24).

사용자 결정 (docs/CD_redesign_worksheet.md D6.24):
  - 지표 CD_rate, ρ 열 0.7·0.3 (D6.27; 처음엔 0.7·0.1), safety = S0+S1 중앙값 (490, 166), β=1e-6.
  - safe-CD 열 = J 와 낙상률 ("use safe-cd J and fallrate. usually that's what we are targeting
    to see if it improves the fallrate.").
  - 기준선 (BC / ESR-IQL / ESR-DICE) 의 safe-CD 열은 safety env 재평가 (eval/safepv_100_score.npy).

J 정의 = AET 학습 목적 그대로: J = Σ_k plog( E_τ[ F_k(τ) ] ),  F = CobbDouglasUtility(safety=linear)
  → F_0 = T·CD_rate, F_1 = ΣS_posture − 490, F_2 = ΣS_velocity − 166  (src.data.reward_transform, SSOT).
CD_rate 는 cd_rate_score (SSOT). 정규화 = raw/T (config reward_min 0, reward_max T — buffer 와 같음).

사용: source scripts/aetenv.sh && python -m experiments.paper_fig5.cd_tables [--env walker2d|half_cheetah]
"""
import argparse
import glob
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from src.data.reward_transform import CobbDouglasUtility, cd_rate_score, cd_safety_step_ranges
from src.utility import cd_kwargs_from_config, piecewise_log

T = 500
RHOS = (0.7, 0.3)          # D6.27 (2026-11-05): "0.3으로 변경하자. 0.1은 굳이 넣지 말자."
SEEDS = (1, 2, 3)
REEVAL = "safepv_100"          # scripts/reeval_safe_score.sh 의 OUT
SAFE_PREFIX = "safecd_b1e-6_pv_lin_thmed"
ROOTS = {"walker2d": "results_cd_walker_expert", "half_cheetah": "results_cd_halfcheetah_expert"}
PFX = {"walker2d": "", "half_cheetah": "hc_"}


def _one(pattern):
    g = [d for d in glob.glob(pattern) if "crashed" not in d]
    assert len(g) <= 1, (pattern, g)
    return g[0] if g else None


def _load(run_dir, reeval=False):
    """(raw returns [N, C], lengths [N]). reeval=True 면 safety env 재평가 산출물."""
    e = f"{run_dir}/eval/"
    if reeval:
        return (np.load(e + f"{REEVAL}_score.npy").astype(np.float64),
                np.load(e + f"{REEVAL}_episode_lengths.npy").astype(np.float64))
    return (np.load(e + "final_returns.npy").astype(np.float64),
            np.load(e + "final_episode_lengths.npy").astype(np.float64))


def cd_rate(R, L, rho):
    return float(np.mean(cd_rate_score(R[:, 0] / T, R[:, 1] / T, L, T, rho, xp=np)))


def safe_J(R, L, safe_cfg):
    """R: raw [N, 4] (speed, cost, S_posture, S_velocity). AET 목적 J = Σ_k plog(E[F_k])."""
    assert R.shape[1] == 4, R.shape
    F = np.asarray(CobbDouglasUtility(**cd_kwargs_from_config(safe_cfg))(
        jnp.asarray(R / T), keep_dims=True, steps=jnp.asarray(L)))
    assert F.shape == (R.shape[0], 3), F.shape
    return float(np.sum(np.asarray(piecewise_log(jnp.asarray(F.mean(axis=0))))))


POS_RESHAPE = "walker2d_cd_safe_pok"     # D6.32: safe-CD 채점 기준 = pos-violog (스텝별 자세 문턱 위반)
POS_REEVAL = "safepok_100"               # scripts/reeval_safe_score.sh (SCORE_RESHAPE=walker2d_cd_safe_pok) 의 OUT


def pos_violog_metrics(R, L, rho):
    """R: raw [N, 3] (speed, cost, ΣS0ok) — walker2d_cd_safe_pok 레이아웃, L: [N].
    D6.32 (2026-11-06) 사용자: "safe-cd utiility는 이제 이 pos-violog 기준으로 채점해야해. posture violation이 몇번있었는지 등등으로".
      posture 위반 D = (S0 < 0.5 인 산 스텝 수) + (못 산 스텝 수 T − L)
      safe-CD utility = E_τ[ plog(T·CD_rate) − log(1 + D) ]  — AET pos-violog 학습 목적 그대로 (CobbDouglasUtility, SSOT)."""
    assert R.shape[1] == 3, R.shape
    U = CobbDouglasUtility(rho=float(rho), form="rate", horizon=float(T), safety="violog",
                           safety_range=cd_safety_step_ranges(POS_RESHAPE))
    F = np.asarray(U(jnp.asarray(R / T), keep_dims=True, steps=jnp.asarray(L)), np.float64)
    D = T - R[:, 2]                                   # 위반 스텝 수 (못 산 스텝 포함)
    return dict(J=float(F.sum(-1).mean()), viol=float(D.mean()), viol0=float(np.mean(D == 0)),
                fall=float(np.mean(L < T)), cd=cd_rate(R, L, rho))


H_REEVAL = "safehm_100"                 # D6.39: scripts/reeval_safe_score.sh (SCORE_RESHAPE=walker2d_cd_safe_hm) 의 OUT
H_RESHAPE = "walker2d_cd_safe_h"


def height_metrics(R, L, rho, r_ths):
    """R: raw [N, 4] (speed, cost, Σm^h, Σ1[m^h ≥ 0.25]) — walker2d_cd_safe_hm 레이아웃, L: [N].
    D6.39 (2026-11-06) 사용자: "safety cd metric도 이제 바꿔야해. 높이로" → "둘 다 보고", 지표 문턱은 학습 결과 보고 결정.
      높이 F-log utility(r_th) = E_τ[ plog(T·CD_rate) + plog(Σm^h − r_th) ]  — 높이 F-log 학습 목적 그대로 (CobbDouglasUtility)
      높이 위반 D = (m^h < 0.25 인 산 스텝) + (T − L), 위반 0 비율, 낙상률, CD_rate."""
    assert R.shape[1] == 4 and R.shape[0] == L.shape[0], (R.shape, L.shape)
    out = dict(cd=cd_rate(R, L, rho), fall=float(np.mean(L < T)), viol=float(np.mean(T - R[:, 3])),
               viol0=float(np.mean(T - R[:, 3] == 0)))
    for th in r_ths:
        U = CobbDouglasUtility(rho=float(rho), form="rate", horizon=float(T), safety="inlog", threshold=float(th))
        F = np.asarray(U(jnp.asarray(R[:, :3] / T), keep_dims=True, steps=jnp.asarray(L)), np.float64)
        out[f"J_h{th:g}"] = float(F.sum(-1).mean())
    return out


def _agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    return (a.mean(), a.std(ddof=0), len(a)) if len(a) else (np.nan, np.nan, 0)


def method_dirs(env, method, rho, iql_best):
    root, p = ROOTS[env], PFX[env]
    if method == "BC":
        return [_one(f"{root}/*/*_{p}{'cdexp_' if env == 'walker2d' else ''}bcracc_nofilter_original_s{s}")
                or _one(f"{root}/*/*_{p}cdexp_bcracc_nofilter_s{s}") for s in SEEDS]
    if method == "ESR-IQL":
        tp = iql_best.get((env, rho))
        return [_one(f"{root}/*/*_{tp}_original_s{s}") for s in SEEDS] if tp else []
    if method == "ESR-DICE":
        return [_one(f"{root}/*/G-linear*_F-cobbdouglas_rho{rho}_cdrate_beta1e-6_*_{p}cdexp_mog_b1e-6_rho{rho}_original_s{s}")
                for s in SEEDS]
    if method == "safe-CD AET":
        return [_one(f"{root}/*/*_{SAFE_PREFIX}_rho{rho}_original_s{s}") for s in SEEDS]
    raise KeyError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="walker2d", choices=list(ROOTS))
    ap.add_argument("--iql_best", nargs="*", default=[],
                    help="ρ=TAG_PREFIX 쌍, 예: 0.7=esriql_tau0.9_T10_rho0.7 (sweep 결과로 고른 설정)")
    args = ap.parse_args()
    iql_best = {(args.env, float(kv.split("=")[0])): kv.split("=")[1] for kv in args.iql_best}
    methods = ["BC", "ESR-IQL", "ESR-DICE"] + (["safe-CD AET"] if args.env == "walker2d" else [])
    out = {}
    for rho in RHOS:
        safe_ref = _one(f"{ROOTS['walker2d']}/*/*_{SAFE_PREFIX}_rho{rho}_original_s1")
        safe_cfg = SimpleNamespace(**json.load(open(f"{safe_ref}/config.json"))) if safe_ref else None
        for m in methods:
            # 아직 학습 중인 런 (final_returns 없음) 은 빼고 n 으로 드러낸다.
            dirs = [d for d in method_dirs(args.env, m, rho, iql_best)
                    if d and glob.glob(f"{d}/eval/final_returns.npy")]
            cd, fall, J, fallS = [], [], [], []
            for d in dirs:
                R, L = _load(d)
                cd.append(cd_rate(R, L, rho)); fall.append(float(np.mean(L < T)))
                if args.env != "walker2d" or safe_cfg is None:
                    continue
                if m == "safe-CD AET":
                    Rs, Ls = R, L
                else:
                    try:
                        Rs, Ls = _load(d, reeval=True)
                    except FileNotFoundError:
                        continue          # 재평가 전 — 표에서 빈칸으로 둔다 (조용한 대체값 없음)
                J.append(safe_J(Rs, Ls, safe_cfg)); fallS.append(float(np.mean(Ls < T)))
            out[(m, rho)] = dict(cd=_agg(cd), fall=_agg(fall), J=_agg(J), fallS=_agg(fallS))
            c, f, j, fs = out[(m, rho)]["cd"], out[(m, rho)]["fall"], out[(m, rho)]["J"], out[(m, rho)]["fallS"]
            print(f"{args.env} ρ{rho} {m:12s} CD_rate {c[0]:.3f}±{c[1]:.3f} (n={c[2]}) fall {f[0]:.2f} | "
                  f"safe J {j[0]:.2f}±{j[1]:.2f} (n={j[2]}) fall {fs[0]:.2f}")
    return out


if __name__ == "__main__":
    main()
