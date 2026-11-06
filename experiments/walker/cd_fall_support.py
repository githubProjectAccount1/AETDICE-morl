"""Walker CD 정책이 넘어지기 직전 상태는 데이터 support 밖인가, 안이지만 w 가 적은 곳인가 (2026-10-27).

docs/CD_redesign_worksheet.md §10 D1 (Decision 10.1, 사용자 2026-10-27 "10.1 의 진단부터").
§7 가설: w 가 에피소드 초반에 몰려 WBC 가 중후반 상태를 거의 배우지 않고, 정책이 그 구간에서 넘어진다
("in-support, under-weighted"). 대안: 정책이 데이터 support 밖으로 나가서 넘어진다 ("OOD").

측정 (임계값은 전부 데이터 자신에서 나온다):
  특징 공간 = [정규화 상태 (buffer 와 같은 state_mean/std), rate-racc R_t/max(t,1) 를 racc_mean/std 로 표준화,
              t 를 참조 표본의 mean/std 로 표준화] — 정책이 실제로 보는 조건 입력과 같은 양이다.
  참조 집합  = 데이터 궤적 90% 에서 뽑은 transition _N_REF 개.
  OOD 기준  = 나머지 10% 궤적(보류)의 transition 이 참조 집합까지의 1-NN 거리의 p99.
              같은 궤적의 이웃 스텝이 참조에 섞이면 기준이 과소평가되므로 궤적 단위로 나눈다.
  롤아웃    = 학습 최종 평가와 같은 경로·키 (evaluate, PRNGKey(0), 100 ep, reward_reshape 순차 경로).
              에피소드 길이가 저장된 final_episode_lengths.npy 와 같은지 확인한다 (재현 검사).
  w 부착    = 각 방문 상태의 최근접 참조 transition 에 그 런 critic 의 자기 시점 w (w_reweight, head='own').
              데이터 평균 w 대비 비율로 낸다. BC 런은 critic 이 없어 w 를 붙이지 않는다.
사용: source scripts/aetenv.sh && JAX_PLATFORMS=cpu python -m experiments.walker.cd_fall_support --run <name>
      (이름 = _RUNS 키) ; 모두 끝나면 --summary
"""
import argparse
import glob
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.data.reward_transform import racc_rate
from src.ops.w_reweight import load_run, traj_starts, transition_w

_RUNS = {
    "aet_b1e-4": "results_cd_walker_diffusion/*/*rho0.5_cdrate_beta1e-4_seed1_*",
    "aet_b1e-5": "results_cd_walker_sweep/*/*amp1_b1e-5_diffusion_s1",
    "aet_b1e-6": "results_cd_walker_sweep/*/*amp1_b1e-6_diffusion_s1",
    "bc_top5": "results_cd_walker_bc/*/*cdbc_top5_rho0.5_s1",
}
_BUFFER_RUN = "aet_b1e-4"      # BC 런은 rate 버퍼가 없다 — 같은 데이터·같은 특징 공간을 이 런의 버퍼로 만든다
_N_REF = 1_000_000
_N_HOLD = 20_000
_HOLD_FRAC = 0.1
_SPLIT_SEED = 0
_OOD_Q = 99.0
_PRE_FALL = 20                 # 낙상 직전 창 (스텝)
_T_BIN = 25                    # §7 진단과 같은 시점 구간 폭
_N_EP = 100
_KNN_JOBS = 32
_OUT_DIR = "results_viz/cd_fall_support"


def _one(pattern):
    hits = sorted(glob.glob(pattern))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def reference_split(d, n_ref, n_hold, seed):
    """궤적 단위 90/10 분할 → (참조 transition 인덱스, 보류 transition 인덱스)."""
    starts, lengths = traj_starts(np.asarray(d.traj_ids))
    rng = np.random.default_rng(seed)
    hold_traj = rng.random(len(starts)) < _HOLD_FRAC
    tid = np.repeat(np.arange(len(starts)), lengths)
    ref_pool, hold_pool = np.flatnonzero(~hold_traj[tid]), np.flatnonzero(hold_traj[tid])
    ref = np.sort(rng.choice(ref_pool, min(n_ref, len(ref_pool)), replace=False))
    hold = np.sort(rng.choice(hold_pool, min(n_hold, len(hold_pool)), replace=False))
    return ref, hold


class FeatureSpace:
    """정책 조건 입력과 같은 양으로 만든 공통 특징 공간 — 데이터와 롤아웃이 같은 식을 쓴다."""

    def __init__(self, cfg, t_ref):
        self.racc_mean = np.asarray(cfg.racc_mean, np.float32)
        self.racc_std = np.asarray(cfg.racc_std, np.float32)
        self.state_mean = np.asarray(cfg.state_mean, np.float32)
        self.state_std = np.asarray(cfg.state_std, np.float32)
        self.t_mean, self.t_std = float(t_ref.mean()), float(t_ref.std())

    def from_buffer(self, d, idx):
        # buffer 는 상태를 이미 정규화해 저장하고, rate 버퍼의 raccs 는 racc_rate(R_t, t) 다.
        s = np.asarray(d.states[idx], np.float32)
        r = (np.asarray(d.raccs[idx], np.float32) - self.racc_mean) / self.racc_std
        t = (np.asarray(d.timesteps[idx], np.float32)[:, None] - self.t_mean) / self.t_std
        return np.concatenate([s, r, t], 1)

    def from_rollout(self, raw_states, racc_raw, t):
        s = (raw_states - self.state_mean) / self.state_std
        r = (racc_rate(racc_raw.astype(np.float32), t) - self.racc_mean) / self.racc_std
        tt = (t.astype(np.float32)[:, None] - self.t_mean) / self.t_std
        return np.concatenate([s, r, tt], 1).astype(np.float32)


def rollout(run_dir):
    """학습 최종 평가와 같은 롤아웃 + 스텝별 (raw 상태, racc, t)."""
    from src.eval.evaluate import evaluate
    from src.eval.rollout import load_config, load_model_cpu, LEARNER_MODULES
    from src.train.setup import build_env, build_utility_F
    cfg = load_config(run_dir)
    build_utility_F(cfg)
    env = build_env(cfg)
    learner = LEARNER_MODULES[getattr(cfg, "learner_type", "aet")]
    ts = load_model_cpu(os.path.abspath(os.path.join(run_dir, "model")), cfg, learner.init_train_state)
    policy = learner.get_model(ts.policy_state)[0]
    info = evaluate(cfg, policy, env, None, num_episodes=_N_EP, max_steps=cfg.max_seq_len,
                    reward_reshape=cfg.reward_reshape, return_racc_traj=True,
                    key=jax.random.PRNGKey(0), eval_batch=getattr(cfg, "eval_batch", 0))
    rv, sv, lens = info["racc_visits"], info["state_visits"], info["episode_lengths"]
    assert rv.shape[0] == sv.shape[0] == int(lens.sum()), (rv.shape, sv.shape, lens.sum())
    saved = np.load(os.path.join(run_dir, "eval", "final_episode_lengths.npy"))
    return cfg, sv, rv[:, 1:], rv[:, 0].astype(np.int64), lens, bool(np.array_equal(saved, lens))


def analyse(name):
    run_dir = _one(_RUNS[name])
    base = load_run(_one(_RUNS[_BUFFER_RUN]) if name.startswith("bc") else run_dir)
    d = base["buffer"].data
    ref, hold = reference_split(d, _N_REF, _N_HOLD, _SPLIT_SEED)
    fs = FeatureSpace(base["config"], np.asarray(d.timesteps[ref], np.float32))
    x_ref, x_hold = fs.from_buffer(d, ref), fs.from_buffer(d, hold)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=_KNN_JOBS).fit(x_ref)
    d_hold = nn.kneighbors(x_hold)[0][:, 0]
    thr = float(np.percentile(d_hold, _OOD_Q))

    cfg, sv, racc_raw, t, lens, reproduced = rollout(run_dir)
    x_pol = fs.from_rollout(sv, racc_raw, t)
    dist, nn_idx = nn.kneighbors(x_pol)
    dist, nn_idx = dist[:, 0], nn_idx[:, 0]
    ood = dist > thr

    ep = np.repeat(np.arange(len(lens)), lens)
    fell = lens < cfg.max_seq_len
    steps_left = np.repeat(lens, lens) - 1 - t                    # 남은 스텝 (마지막 스텝 = 0)
    pre_fall = fell[ep] & (steps_left < _PRE_FALL)
    in_completed = ~fell[ep]
    res = dict(run=os.path.basename(run_dir), reproduced_eval=reproduced, n_ep=int(len(lens)),
               fall_rate=float(fell.mean()), fall_step_median=float(np.median(lens[fell])) if fell.any() else None,
               ood_threshold_p99_holdout=thr, holdout_dist_median=float(np.median(d_hold)),
               ood_frac_all=float(ood.mean()), ood_frac_pre_fall=float(ood[pre_fall].mean()) if pre_fall.any() else None,
               ood_frac_completed=float(ood[in_completed].mean()) if in_completed.any() else None,
               ood_frac_fell_before_window=float(ood[fell[ep] & ~pre_fall].mean()) if (fell[ep] & ~pre_fall).any() else None,
               dist_median_ratio_pre_fall=float(np.median(dist[pre_fall]) / np.median(d_hold)) if pre_fall.any() else None,
               dist_median_ratio_completed=float(np.median(dist[in_completed]) / np.median(d_hold)) if in_completed.any() else None,
               eps_with_ood_in_pre_fall=float(np.mean([ood[pre_fall & (ep == e)].any() for e in np.flatnonzero(fell)]))
               if fell.any() else None)
    if not name.startswith("bc"):
        w = transition_w(base, ref, head="own")
        assert w.shape == ref.shape, (w.shape, ref.shape)
        w_mean = float(w.mean())
        t_ref = np.asarray(d.timesteps[ref])
        n_bins = -(-cfg.max_seq_len // _T_BIN)
        b = np.minimum(t_ref // _T_BIN, n_bins - 1)
        bin_ratio = (np.bincount(b, weights=w, minlength=n_bins) / w.sum()) / (np.bincount(b, minlength=n_bins) / len(b))
        w_nn, bin_nn = w[nn_idx] / w_mean, bin_ratio[np.minimum(t_ref[nn_idx] // _T_BIN, n_bins - 1)]
        for k, m in (("pre_fall", pre_fall), ("completed", in_completed), ("all", np.ones_like(ood))):
            if m.any():
                res[f"nn_w_ratio_mean_{k}"] = float(w_nn[m].mean())
                res[f"nn_w_pos_frac_{k}"] = float((w_nn[m] > 0).mean())
                res[f"nn_tbin_wmass_ratio_{k}"] = float(bin_nn[m].mean())
        res["data_w_pos_frac"] = float((w > 0).mean())
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(os.path.join(_OUT_DIR, f"{name}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez(os.path.join(_OUT_DIR, f"{name}.npz"), dist=dist, ood=ood, ep=ep, t=t, lens=lens, thr=thr)
    for k, v in res.items():
        print(f"  {k:32s} {v}")


def summary():
    rows = {n: json.load(open(os.path.join(_OUT_DIR, f"{n}.json"))) for n in _RUNS
            if os.path.exists(os.path.join(_OUT_DIR, f"{n}.json"))}
    with open(os.path.join(_OUT_DIR, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    keys = ("fall_rate", "fall_step_median", "ood_frac_all", "ood_frac_completed", "ood_frac_fell_before_window",
            "ood_frac_pre_fall", "eps_with_ood_in_pre_fall", "dist_median_ratio_completed", "dist_median_ratio_pre_fall",
            "nn_w_ratio_mean_completed", "nn_w_ratio_mean_pre_fall", "nn_w_pos_frac_completed", "nn_w_pos_frac_pre_fall",
            "nn_tbin_wmass_ratio_completed", "nn_tbin_wmass_ratio_pre_fall", "data_w_pos_frac", "reproduced_eval")
    print("metric".ljust(32) + "".join(n.rjust(12) for n in rows))
    for k in keys:
        print(k.ljust(32) + "".join(
            (f"{rows[n][k]:12.3f}" if isinstance(rows[n].get(k), float) else str(rows[n].get(k, "-")).rjust(12))
            for n in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(_RUNS))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        assert args.run, "--run 또는 --summary"
        analyse(args.run)


if __name__ == "__main__":
    main()
