"""AETDICE 체크포인트의 w = d^π/d^D 를 데이터셋 transition 에 대해 재현한다 (학습 無).

왜 공용 모듈인가 (SSOT): w 재현 공식이 `w_spread_diagnostic.py` 에 있었고 corner-질량
진단이 두 번째 사본을 만들 차례였다. w 공식은 AET_learner 와 한 글자도 달라서는 안 되는
load-bearing 코드라 한 곳에만 둔다.

공식 (AET_learner 와 동일, CHI):
    e     = (r·mu) + ν(s', racc') - ν(s, racc)        # [B, H]
    w     = relu( f'^{-1}(e / β) )                     # w_max 로 clip (기본 inf = OFF)
    w_b   = mean_h w[b, h]                             # WBC 가 (b,h) 전부 쓰므로 head 평균
"""
import os

import numpy as np
import jax.numpy as jnp

from src.divergence import FDivergence, f_derivative_inverse
from src.eval.rollout import LEARNER_MODULES, load_config, load_model_cpu
from src.train.setup import build_buffer, build_cd_filter, build_utility_F

_CHUNK = 4096          # ν 출력이 [B, H+1] 이라 B 를 키우면 메모리가 H 배로 늘어난다.


def load_run(run_dir, model_subdir="model"):
    """(config, buffer, nu, mu, beta, fdiv, w_max) — 학습 없이 체크포인트만 복원한다."""
    config = load_config(run_dir)
    build_utility_F(config)
    buffer = build_buffer(config, build_cd_filter(config))     # config 에 racc 통계를 채운다
    learner = LEARNER_MODULES[getattr(config, "learner_type", "aet")]
    path = os.path.abspath(os.path.join(run_dir, model_subdir))
    if not os.path.exists(path):
        raise SystemExit(f"체크포인트 없음: {path}")
    ts = load_model_cpu(path, config, learner.init_train_state)
    nu = learner.get_model(ts.nu_state)[0]
    mu = np.asarray(learner.get_model(ts.mu_state)[0]())
    policy = learner.get_model(ts.policy_state)[0]
    return dict(config=config, buffer=buffer, nu=nu, mu=mu, policy=policy, learner=learner,
                beta=float(config.beta),
                # CLI 는 enum NAME("CHI")을 넘긴다 (value "Chi" 아님).
                fdiv=FDivergence[config.divergence],
                w_max=float(getattr(config, "w_max", float("inf"))),
                model_path=path)


def transition_w(run, idx, chunk=_CHUNK, verbose=False, betas=None, return_parts=False,
                 head="mean"):
    """per-transition w (head 평균). 반환 [len(idx)] (betas=None) 또는 [len(betas), len(idx)].

    betas 를 주면 **같은 한 번의 ν 순전파로 여러 β 의 w 를 동시에** 낸다. 왜 필요한가
    (2026-09-17): CHI 는 w = relu(1 + e/β) 이고 실측 e 는 mean −3.3, σ 11.1 인데 β=1e-6
    으로 학습해 왔다 → 대다수 transition 이 w=0, 소수가 1e6 규모 = ESS 0.017%. β 가
    어느 규모여야 w 가 연속 가중으로 살아나는지 **학습 전에** 이 스캔으로 정한다.
    (주의: ν 는 원래 β 로 학습된 것이라, 다른 β 의 w 는 가중 퇴화 진단이지 그 β 로
     학습한 결과의 예측은 아니다.)
    """
    d = run["buffer"].data
    nu, mu, fdiv, w_max = run["nu"], run["mu"], run["fdiv"], run["w_max"]
    bs = np.atleast_1d(np.asarray(betas if betas is not None else run["beta"], np.float64))
    out = np.empty((len(bs), len(idx), 2) if head == "both" else (len(bs), len(idx)), np.float64)
    # return_parts: e 의 두 성분을 따로 돌려준다 — 신호가 보상항에서 죽는지 ν-telescope
    # 항에서 죽는지 구분하려면 합쳐진 e 만으로는 알 수 없다 (2026-09-19 사슬 진단).
    parts = {k: np.empty(len(idx), np.float64) for k in ("r_mu", "nu_diff", "e_mean", "e_std")} \
        if return_parts else None
    # head="mean": WBC(original)이 action 을 H head 전부에 복제해 채점하므로 head 평균.
    # head="own" : 각 transition 을 **자기 timestep head** t_b 에서만 본다. 필터 기준으로는
    #   이쪽이 맞다 — 평균은 "500 head 중 하나만 양수여도 w>0" 이라 지나치게 관대해서
    #   궤적 단위 zero-w 필터가 공집합이 된다 (2026-09-19 실측 50,000/50,000 keep).
    # "min"    : head 전부가 w>0 이어야 양수 — 가장 엄격한 끝.
    # "fracpos": 그 transition 에서 w>0 인 head 의 **비율** (0~1). 엄격도 진단용.
    # "both" : own 과 mean 을 한 번의 ν 순전파로 동시에 낸다 (열 2 개로 반환).
    #          후보 필터를 전수로 비교할 때 ν forward 를 두 번 돌리지 않기 위해서다.
    assert head in ("mean", "own", "min", "fracpos", "both"), head
    ts_all = np.asarray(d.timesteps) if head in ("own", "both") else None
    for i in range(0, len(idx), chunk):
        sl = idx[i:i + chunk]
        nu_t = np.asarray(nu(jnp.asarray(d.states[sl]), jnp.asarray(d.raccs[sl]),
                             timesteps=None))[:, :-1]
        nu_n = np.asarray(nu(jnp.asarray(d.next_states[sl]), jnp.asarray(d.next_raccs[sl]),
                             timesteps=None))[:, 1:]
        r_mu = (np.asarray(d.rewards[sl], np.float64) @ mu).reshape(-1, 1)
        nu_d = nu_n - nu_t
        e = r_mu + nu_d
        if parts is not None:
            j = slice(i, i + len(sl))
            parts["r_mu"][j]    = r_mu[:, 0]
            parts["nu_diff"][j] = nu_d.mean(axis=1)
            parts["e_mean"][j]  = e.mean(axis=1)
            parts["e_std"][j]   = e.std(axis=1)      # head 간 분산 (시간 정렬 민감도)
        for bi, b in enumerate(bs):
            w = np.asarray(jnp.maximum(0.0, f_derivative_inverse(jnp.asarray(e) / b, fdiv,
                                                                 w_max=w_max)))
            if head == "both":
                t_b = np.minimum(ts_all[sl], w.shape[1] - 1)
                out[bi, i:i + len(sl), 0] = w[np.arange(len(sl)), t_b]
                out[bi, i:i + len(sl), 1] = w.mean(axis=1)
            elif head == "mean":
                out[bi, i:i + len(sl)] = w.mean(axis=1)
            elif head == "min":
                out[bi, i:i + len(sl)] = w.min(axis=1)
            elif head == "fracpos":
                out[bi, i:i + len(sl)] = (w > 0).mean(axis=1)
            else:
                t_b = np.minimum(ts_all[sl], w.shape[1] - 1)
                out[bi, i:i + len(sl)] = w[np.arange(len(sl)), t_b]
        if verbose and (i // chunk) % 40 == 0:
            print(f"    w {i + len(sl):,}/{len(idx):,}", flush=True)
    w_out = out[0] if betas is None else out
    return (w_out, parts) if return_parts else w_out


def traj_starts(traj_ids):
    """궤적 경계. traj_ids 는 단조 비감소이므로 diff 로 시작 위치를 얻는다."""
    t = np.asarray(traj_ids)
    starts = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    lengths = np.diff(np.r_[starts, len(t)])
    return starts, lengths


def sample_per_traj(traj_ids, per_traj, seed=0):
    """궤적마다 **균등 간격** per_traj 개 transition 인덱스. WBC 가 모든 transition 을
    쓰므로 균등 표본의 평균은 그 궤적 평균 w 의 불편추정이다."""
    starts, lengths = traj_starts(traj_ids)
    idx, owner = [], []
    for j, (s, L) in enumerate(zip(starts, lengths)):
        k = min(per_traj, L)
        pos = np.linspace(0, L - 1, k).round().astype(np.int64)
        idx.append(s + pos)
        owner.append(np.full(k, j, np.int64))
    return np.concatenate(idx), np.concatenate(owner), lengths


def traj_returns_normalized(run):
    """궤적별 **정규화 누적보상** [num_traj, D].

    F̃(x)=F(x)−F(0) 이 저장돼 있으므로 마지막 next_racc 를 역변환한다. exponential_F 는
    F̃(x)=exp(x/S)−1 → x = S·ln(1+F̃). linear 는 항등.
    """
    d = run["buffer"].data
    starts, lengths = traj_starts(d.traj_ids)
    last = starts + lengths - 1
    F_end = np.asarray(d.next_raccs[last], np.float64)
    ftype = getattr(run["config"], "utility_F_type", "linear")
    if ftype == "exponential_F":
        S = float(getattr(run["config"], "exp_f_scale", 50.0))
        return S * np.log1p(np.maximum(F_end, -1.0 + 1e-12))
    if ftype == "linear":
        return F_end
    raise SystemExit(f"역변환 미구현 F 타입: {ftype}")
