"""Table 4 (MO-Ant Expert) 스타일 리포트 — 여러 ckpt 의 eval 결과를 한 표로 출력.

`src/eval/rollout.py` 는 ckpt 하나의 점수를 `score_X: val` 줄로 나열한다. 이 스크립트는
여러 ckpt(방법 × seed)를 모아 방법별 평균 획득 누적 보상을 선호 가중치 w 별로 표로 찍고,
맨 아래에 **기준 알고리즘(FairDICE) 대비 제안 알고리즘(AETDICE)의 누적 보상 증가율(%)** 행을
추가한다.

    증가율(%) = (제안 평균 − 기준 평균) / 기준 평균 × 100     (w 열마다 하나씩)
    평균 증가율(%) = 세 w 열 증가율의 산술평균

채점은 `evaluate.py` 와 동일: F = exp(x / exp_f_scale)(config, 기본 50), per-pref 열 =
linear-exp 가중합. 기본은 각 run_dir 의 캐시된 100-ep 결과(`eval/final_returns_normalized.npy`)
를 재채점하고, `--rerun` 시 rollout 을 새로 돌린다.

사용:
    python -m src.eval.report                       # 기본 프리셋(AET-DICE + FairDICE, seed1-5)
    python -m src.eval.report --rerun --num_episodes 100   # ckpt 새로 롤아웃 후 표
    python -m src.eval.report --paper               # 논문 Table 4 대조도 함께
    python -m src.eval.report --method "AET-DICE=dirA,dirB" --method "FairDICE=dirC,dirD"
"""
import argparse
import json
import os
import subprocess
import sys
import unicodedata

import jax.numpy as jnp
import numpy as np

from src.utility import Utility

_EXP_SCALE_DEFAULT = 50.0            # Table 4 = exp(x / 50); config 에 exp_f_scale 없을 때
_NORM_NPY = "final_returns_normalized.npy"   # 학습 최종 100-ep eval 산출물 (캐시 경로)
_W_COLS = ["w[1,0]", "w[0.5,0.5]", "w[0,1]"]  # 표 열 = 선호 가중치별 획득 누적 보상 (G,F 제외)
_ALL_COLS = ["G,F"] + _W_COLS        # 내부 계산은 G,F 도 포함 (논문 대조용)

# 2-obj d4morl 정준 선호 (evaluate.py 의 LINEAR_BASELINE_PREFS 와 동일)
_TABLE4_PREFS = [
    ("w[1,0]",     np.array([1.0, 0.0], dtype=np.float32)),
    ("w[0.5,0.5]", np.array([0.5, 0.5], dtype=np.float32)),
    ("w[0,1]",     np.array([0.0, 1.0], dtype=np.float32)),
]

# 논문 Table 4 (MO-Ant Expert) 참조값. w[0.5,0.5] 는 논문 미기재 → None.
_PAPER = {
    "AET-DICE": {"G,F": 13.686, "w[1,0]": 817, "w[0.5,0.5]": None, "w[0,1]": 1083},
    "FairDICE": {"G,F": 13.041, "w[1,0]": 617, "w[0.5,0.5]": None, "w[0,1]": 748},
}

# 기본 프리셋: MO-Ant expert, 두 방법 × seed 1-5.
_ANT_BASE = "results_Transformed/ant_expert/rsf1.0to1.0_musf1e4"
_PRESET = [
    ("AET-DICE", [f"{_ANT_BASE}/G-plog_F-exp_F_beta1e-6_seed{s}_tfd4morl_ant_expert" for s in range(1, 6)]),
    ("FairDICE", [f"{_ANT_BASE}/fairdice_G-plog_F-linear_beta1e-5_seed{s}_tfd4morl_ant_expert" for s in range(1, 6)]),
]


def _dwidth(s):
    """터미널 표시 폭 (한글=2, ASCII=1) — 한글 라벨 정렬용."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _ljust(s, w):
    return s + " " * max(0, w - _dwidth(s))


def _rjust(s, w):
    return " " * max(0, w - _dwidth(s)) + s


def _load_exp_scale(run_dir):
    """run 의 config.json 에서 exp_f_scale 읽기 (없으면 기본 50)."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    return float(cfg.get("exp_f_scale") or _EXP_SCALE_DEFAULT)


def _score_columns(returns_norm, exp_scale):
    """evaluate.py 와 동일하게 Table-4 열 계산. returns_norm: [episodes, 2]."""
    assert returns_norm.ndim == 2 and returns_norm.shape[1] == 2, \
        f"Table 4 는 2-obj 전용 — shape={returns_norm.shape}"
    f_util = Utility(type="exponential_F", exp_scale=exp_scale)          # convex F
    u0 = np.asarray(f_util(jnp.zeros((1, 2), dtype=jnp.float32), keep_dims=True)).squeeze(0)  # F(0)
    # mean_Ftilde = mean_over_ep[F(R)] - F(0)  (per objective, [2])
    mean_ft = np.asarray(np.mean(np.asarray(f_util(jnp.asarray(returns_norm), keep_dims=True)), axis=0)) - u0
    cols = {"G,F": float(Utility(type="log")(jnp.asarray(mean_ft), keep_dims=False))}   # concave-G headline
    for label, w in _TABLE4_PREFS:                                       # 획득 누적 보상 (linear-exp per pref)
        cols[label] = float(np.sum(w * mean_ft))
    return cols


def _resolve_npy(run_dir, num_episodes, npy_name):
    """읽을 eval npy 결정. 명시값(--npy_name) 우선 → 롤아웃 출력(returns_N) → 학습 최종(final)."""
    eval_dir = os.path.join(run_dir, "eval")
    candidates = [npy_name] if npy_name else [f"returns_{num_episodes}_normalized.npy", _NORM_NPY]
    for c in candidates:
        p = os.path.join(eval_dir, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"eval 결과 없음: {eval_dir} 에 {candidates} 중 하나도 없음 (--rerun 으로 새로 평가하세요)")


def _eval_run(run_dir, num_episodes, npy_name, rerun):
    """run_dir 하나 → Table-4 열 dict. rerun=True 면 rollout 을 새로 실행."""
    if rerun:
        subprocess.run(
            [sys.executable, "-m", "src.eval.rollout",
             "--run_dir", run_dir, "--num_episodes", str(num_episodes)],
            check=True,
        )
        npy_name = f"returns_{num_episodes}_normalized.npy"              # rollout 이 저장하는 이름
    npy = _resolve_npy(run_dir, num_episodes, npy_name)
    return _score_columns(np.load(npy), _load_exp_scale(run_dir))


def _aggregate(cols_list):
    """seed별 열 dict 리스트 → {열: (mean, population-std)}."""
    return {k: (float(np.mean([c[k] for c in cols_list])),
                float(np.std([c[k] for c in cols_list]))) for k in _ALL_COLS}


def _increase_rates(results, proposed, baseline):
    """w 열마다 (제안−기준)/기준 × 100 + 세 열 평균.

    표에 정수로 표시되는 평균값과 손검산이 일치하도록, 표시 자리수(정수)로 반올림한 평균을
    분자/분모에 쓴다 (무결성 검사에서 "표에 보이는 값으로 검산"과 동일한 결과)."""
    prop, base = results[proposed], results[baseline]
    rates = {c: (round(prop[c][0]) - round(base[c][0])) / round(base[c][0]) * 100.0 for c in _W_COLS}
    return rates, sum(rates.values()) / len(rates)


def _fmt_cell(mean, std, prec=0):
    """mean±std. std==0(단일 seed)이면 단일값."""
    return f"{mean:.{prec}f}" if std == 0.0 else f"{mean:.{prec}f}±{std:.{prec}f}"


def _print_main_table(results, n_seeds, proposed, baseline):
    """행=방법(+증가율), 열=선호별 획득 누적 보상. 맨 아래 누적 보상 증가율(%) 행."""
    tags = {proposed: " (제안)", baseline: " (기준)"}
    label_w = max(_dwidth(m + tags.get(m, "")) for m in results) + 2
    label_w = max(label_w, _dwidth("누적보상 증가율 (%)") + 2)
    col_w, avg_w = 18, 12

    header = _ljust("METHOD", label_w) + "".join(_rjust(c, col_w) for c in _W_COLS) + _rjust("평균", avg_w)
    print(header)
    print("-" * _dwidth(header))
    for method, agg in results.items():                                 # 방법별 획득 누적 보상 (mean±std)
        row = _ljust(method + tags.get(method, ""), label_w)
        row += "".join(_rjust(_fmt_cell(*agg[c]), col_w) for c in _W_COLS)
        row += _rjust("—", avg_w)
        print(row)

    if proposed in results and baseline in results:                     # 누적 보상 증가율 행
        rates, avg = _increase_rates(results, proposed, baseline)
        print("-" * _dwidth(header))
        row = _ljust("누적보상 증가율 (%)", label_w)
        row += "".join(_rjust(f"{rates[c]:+.2f}", col_w) for c in _W_COLS)
        row += _rjust(f"{avg:+.2f}", avg_w)
        print(row)
        print(f"\n(각 방법 {n_seeds} seed · 100-ep · population std)")
        print(f"증가율(%) = (제안[{proposed}] − 기준[{baseline}]) / 기준 × 100 · 평균 = 세 w 열 증가율의 평균")
    else:
        print(f"\n(각 방법 {n_seeds} seed · 증가율 생략: 제안/기준 방법이 결과에 없음)")


def _print_paper_compare(results):
    """재현값 vs 논문 Table 4 side-by-side (논문값 있는 열만)."""
    print("\n" + "=" * 60)
    print("재현 vs 논문 Table 4 (무결성/재현 확인)")
    print("=" * 60)
    for method, agg in results.items():
        paper = _PAPER.get(method)
        if paper is None:
            continue
        print(f"\n[{method}]")
        for c in _ALL_COLS:
            pv = paper.get(c)
            if pv is None:
                continue
            mean, std = agg[c]
            is_gf = (c == "G,F")
            ours = _fmt_cell(mean, std, prec=3 if is_gf else 0)
            pstr = f"{pv:.3f}" if is_gf else f"{pv:.0f}"
            dstr = f"{mean - pv:+.3f}" if is_gf else f"{mean - pv:+.0f}"
            print(f"  {_ljust(c, 12)} 재현 {ours:>14}   논문 {pstr:>8}   Δ {dstr:>8}")


def _parse_methods(method_args):
    """--method 'Label=dir1,dir2' 인자들을 [(label, [dirs])] 로. 없으면 기본 프리셋."""
    if not method_args:
        return _PRESET
    parsed = []
    for spec in method_args:
        label, _, dirs = spec.partition("=")
        assert dirs, f"--method 형식은 'Label=dir1,dir2' : {spec!r}"
        parsed.append((label.strip(), [d.strip() for d in dirs.split(",") if d.strip()]))
    return parsed


def main():
    p = argparse.ArgumentParser(description="Table 4 스타일 eval 리포트 (+ 누적 보상 증가율)")
    p.add_argument("--method", action="append", default=None,
                   help="'Label=run_dir1,run_dir2,...' (반복 가능). 생략 시 기본 프리셋(AET-DICE+FairDICE seed1-5).")
    p.add_argument("--proposed", type=str, default="AET-DICE", help="제안 알고리즘 라벨 (증가율 분자)")
    p.add_argument("--baseline", type=str, default="FairDICE", help="기준 알고리즘 라벨 (증가율 분모)")
    p.add_argument("--rerun", action="store_true", help="캐시 대신 rollout 을 새로 실행")
    p.add_argument("--num_episodes", type=int, default=100, help="--rerun 시 에피소드 수")
    p.add_argument("--npy_name", type=str, default=None,
                   help="캐시 npy 파일명 (생략 시 자동: returns_<N>_normalized.npy 있으면 그걸, 없으면 final_returns_normalized.npy)")
    p.add_argument("--paper", action="store_true", help="논문 Table 4 대조 블록도 함께 출력")
    args = p.parse_args()

    methods = _parse_methods(args.method)
    results = {}
    n_seeds = None
    for label, run_dirs in methods:
        cols_list = [_eval_run(d, args.num_episodes, args.npy_name, args.rerun) for d in run_dirs]
        results[label] = _aggregate(cols_list)
        n_seeds = len(cols_list) if n_seeds is None else n_seeds

    print()
    _print_main_table(results, n_seeds, args.proposed, args.baseline)
    if args.paper:
        _print_paper_compare(results)


if __name__ == "__main__":
    main()
