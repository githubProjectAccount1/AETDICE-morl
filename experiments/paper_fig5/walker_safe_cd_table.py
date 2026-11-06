"""Tables 12 and 13 of the paper (MO-Walker2d Cobb-Douglas and safe-CD).

For every method and rho in {0.7, 0.3}, over seeds 1-3 (mean +- population std):
  CD utility       cd_rate on eval/final_returns.npy (the run's own final evaluation, 100 episodes)
  safe-CD utility  E[plog(T * CD_rate) + plog(sum m^h - R_th)] with R_th = 279.4 (p75 of the dataset
                   safety returns), on the walker2d_cd_safe_hm re-evaluation (eval/safehm_100_*.npy)
  fall rate        fraction of re-evaluation episodes shorter than T = 500
  satisfaction     fraction of re-evaluation episodes with sum m^h >= R_th

The re-evaluation files are produced by
  SCORE_RESHAPE=walker2d_cd_safe_hm OUT=safehm_100 bash scripts/reeval_safe_score.sh <run_dirs.txt>
for every run (baselines and AET-CD(safe) alike).

Usage: python -m experiments.paper_fig5.walker_safe_cd_table [--root results_cd_walker_expert]
"""
import argparse

import numpy as np

from experiments.paper_fig5.cd_tables import T, _one, cd_rate, height_metrics

RHOS = (0.7, 0.3)
SEEDS = (1, 2, 3)
R_TH_EVAL = 279.4
REEVAL = "safehm_100"


def method_globs(root, iql_tags, safe_prefix):
    """method -> f(rho, seed) -> glob pattern of the run dir (TAG_PREFIXes of README.md)."""
    return {
        "BC": lambda r, s: f"{root}/*/*_cdexp_bcracc_nofilter*_s{s}",
        "ESR-IQL": lambda r, s: f"{root}/*/*_{iql_tags[r]}_original_s{s}",
        "AET-CD": lambda r, s: f"{root}/*/*_cdexp_mog_b1e-6_rho{r}_original_s{s}",
        "AET-CD(safe)": lambda r, s: f"{root}/*/*_{safe_prefix}_rho{r}_original_s{s}",
    }


def run_metrics(run_dir, rho):
    e = f"{run_dir}/eval/"
    R = np.load(e + "final_returns.npy").astype(np.float64)
    L = np.load(e + "final_episode_lengths.npy").astype(np.float64)
    Rh = np.load(e + f"{REEVAL}_score.npy").astype(np.float64)
    Lh = np.load(e + f"{REEVAL}_episode_lengths.npy").astype(np.float64)
    h = height_metrics(Rh, Lh, rho, [R_TH_EVAL])
    return dict(cd=cd_rate(R, L, rho), safe=h[f"J_h{R_TH_EVAL:g}"], fall=100 * h["fall"],
                sat=100 * float(np.mean(Rh[:, 2] >= R_TH_EVAL)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_cd_walker_expert")
    ap.add_argument("--iql_rho0.7", dest="iql07", default="esriql_tau0.9_T18_rho0.7")
    ap.add_argument("--iql_rho0.3", dest="iql03", default="esriql_tau0.9_T10_rho0.3")
    ap.add_argument("--safe_prefix", default="safecd_b1e-3_h_inlog_th1000")
    args = ap.parse_args()
    globs = method_globs(args.root, {0.7: args.iql07, 0.3: args.iql03}, args.safe_prefix)
    fmt = lambda a, d: f"{np.mean(a):.{d}f} ± {np.std(a):.{d}f}"
    print(f"T={T}, safe-CD metric threshold R_th={R_TH_EVAL}")
    for name, g in globs.items():
        for rho in RHOS:
            rows = []
            for s in SEEDS:
                d = _one(g(rho, s))
                assert d is not None, f"missing run: {g(rho, s)}"
                rows.append(run_metrics(d, rho))
            col = lambda k: [x[k] for x in rows]
            print(f"{name:13s} rho={rho}  CD {fmt(col('cd'), 3)} | safe-CD {fmt(col('safe'), 1)} | "
                  f"fall {fmt(col('fall'), 1)}% | satisfaction {fmt(col('sat'), 1)}%")


if __name__ == "__main__":
    main()
