import errno
import os
import shutil
import pickle
import queue
import threading
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from src.utility import Utility
from src.utils import min_max_normalization
# Responsibility split:
#   load_d4morl_dataset() — ONLY per-objective min/max normalization of per-step rewards.
#   Buffer.__init__()     — builds raw per-trajectory raccs from normalized rewards, then
#                           applies the SHIFTED utility  F̃(x) := F(x) − F(0)  so that F̃(0)=0.
#                           Stored raccs live in F̃-space; per-step scalarized reward is the
#                           plain telescoping difference F̃(R_{t+1}) − F̃(R_t) with NO offset.
#                           Trajectory sum of scalarized rewards = F̃(R_final) = F(R_final) − F(0),
#                           which is the outer objective the policy is trained against (paired with
#                           a matching G(F̃(·)) score in evaluate.py).


Batch = namedtuple(
    "Batch",
    [
        "states",
        "next_states",
        "actions",
        "rewards",
        "init_states",
        "raccs",
        "next_raccs",
        "timesteps",      # per-transition timestep t within its trajectory (WBC head alignment)
        # ★ 2026-09-21: 궤적 단위 성분 배정(--traj_mixture)이 "같은 궤적은 같은 성분" 을
        #   강제하려면 transition 이 어느 궤적 소속인지 알아야 한다. 저장부에는 원래 있었고
        #   (_StoredArrays.traj_ids) sample 에서 init_states 로 치환되며 버려지던 것을 같이 싣는다.
        "traj_ids",
    ],
)

# Internal per-step storage. Same shape as Batch except `init_states` is replaced by
# `traj_ids` (int32 [total_transitions]) — we store one init_state PER TRAJECTORY in
# Buffer.traj_init_states and look it up at sample time, avoiding N-copy tiling
# (saves ~state_dim × total_transitions × 4 bytes of GPU memory).
_StoredArrays = namedtuple(
    "_StoredArrays",
    [
        "states",
        "next_states",
        "actions",
        "rewards",
        "traj_ids",
        "raccs",
        "next_raccs",
        "timesteps",      # per-transition timestep t within its trajectory (WBC head alignment)
    ],
)


def load_d4morl_dataset(path, env_name, reward_scaling_factor, reward_reshape: str = "none",
                         max_seq_len=None, cd_filter=None, traj_filter=None,
                         max_trajectories=None, reshape_fn=None, reward_norm_scale=None):
    """
    Load d4morl pkl dataset and normalize per-step rewards per-objective:
        normalized[:, i] = scaling_factor[i] * (raw[:, i] - reward_min[i]) / (reward_max[i] - reward_min[i])

    Responsibility is limited to normalization. Per-trajectory accumulation into raccs
    and the utility_F transform are done inside Buffer.__init__.

    `reward_reshape` selects an optional per-trajectory reshape applied BEFORE
    per-objective min/max + normalization (so reward_min/max reflect the reshaped
    space, not the original 2-obj space). "none" preserves the dataset's reward
    layout. "walker2dcd" reshapes MO-Walker2d's 2-obj (speed, energy) into
    (forward_velocity, ‖a‖² cost) using raw_rewards — see
    `reward_transform.reshape_walker2dcd_trajectory`.

    `max_seq_len` truncates each trajectory to its first `max_seq_len` transitions
    BEFORE reshape and BEFORE per-objective min/max are computed — this matches
    the eval-time horizon cap (main.py:--max_seq_len) and ensures
    reward_min/max reflect the actually-trained-on transitions. None disables
    truncation (full trajectory length is used, matching pre-truncation behavior).

    `cd_filter` — optional dict {"rho": float, "k_pct": float (0,1]} that filters
    the dataset to the top-K% trajectories by CD score (using `rho` for the
    elasticity). reward_min/max are computed on the FULL dataset BEFORE filtering
    so eval-time normalization stays comparable to the unfiltered AETDICE pipeline;
    only the kept trajectories' (state, action, reward) transitions enter the buffer.
    Used by the filtered-BC baseline — see BC_learner.py.
    """
    from src.data.reward_transform import (RESHAPE_TRAJECTORY, CD_RESHAPES,
                                           cd_reward_bounds)

    with open(path, 'rb') as f:
        trajectories = pickle.load(f)

    reward_scaling_factor_arr = np.asarray(reward_scaling_factor, dtype=np.float32)

    # Truncate first — applied BEFORE reshape so reshaped per-step quantities
    # (e.g. ‖a‖² for walker2dcd) only see the kept-transitions actions. Buffer doesn't
    # care about absolute trajectory length; it just sees a per-step transition
    # stream + traj_ids for racc accumulation.
    if max_seq_len is not None:
        truncated = []
        for t in trajectories:
            T = t['observations'].shape[0]
            if T <= max_seq_len:
                truncated.append(t)
                continue
            new_t = dict(t)
            for k, v in t.items():
                if hasattr(v, 'shape') and len(v.shape) >= 1 and v.shape[0] == T:
                    new_t[k] = v[:max_seq_len]
            truncated.append(new_t)
        trajectories = truncated

    # Subsample the number of trajectories. High-dim reshapes (antpref100,
    # reward_dim=100) can't fit the full dataset's [N, reward_dim] racc arrays on
    # device (100× the 2-obj size → OOM); a small n also matches Exp B's small-n
    # regime. Random subset with a fixed seed for reproducibility; done BEFORE
    # reshape/accumulation so traj_id enumeration stays consistent.
    if max_trajectories is not None and len(trajectories) > max_trajectories:
        rng = np.random.default_rng(0)
        keep = sorted(rng.choice(len(trajectories), size=int(max_trajectories), replace=False))
        trajectories = [trajectories[i] for i in keep]

    # If a reshape is requested, apply it to every trajectory's `raw_rewards` and
    # MUTATE the in-memory traj dict so the rest of this function sees the new
    # reward layout (the trajectories list is local to this fn — no caller side-effect).
    # An explicitly injected `reshape_fn` (parameterized reward) takes precedence over the
    # name-based RESHAPE_TRAJECTORY dispatch — keeps parameterized reshapes out of the fixed
    # name registry (open/closed extension point).
    if reshape_fn is None:
        reshape_fn = RESHAPE_TRAJECTORY.get(reward_reshape) if reward_reshape != "none" else None
        if reward_reshape != "none" and reshape_fn is None:
            raise ValueError(
                f"Unknown reward_reshape={reward_reshape!r}. "
                f"Supported: 'none' | {sorted(RESHAPE_TRAJECTORY.keys())}"
            )
    if reshape_fn is not None:
        for t in trajectories:
            # Pass observations through so reshapes that need state info (e.g.
            # walker2d_cd_safe, which reads height/angle from obs) work. Other
            # reshapes accept it as an unused kwarg.
            t['raw_rewards'] = reshape_fn(
                t['actions'], t['raw_rewards'], t.get('next_observations'),
            )

    # Per-objective bounds computed over ALL per-step raw rewards in the dataset
    # (post-reshape if applicable; pre-cd_filter so the eval-side normalization
    # matches the unfiltered AETDICE pipeline).
    all_raw_rewards = np.concatenate(
        [t['raw_rewards'].astype(np.float32) for t in trajectories], axis=0
    )  # [total_transitions, reward_dim]
    if reward_reshape in CD_RESHAPES:
        # CD 는 shift 를 넣으면 순위가 뒤집힌다 → scale-only (0, T). cd_reward_bounds 참조.
        # No silent fallback: T 없이는 CD 정규화가 정의되지 않는다.
        if max_seq_len is None:
            raise ValueError(
                f"reward_reshape={reward_reshape!r} (CD) 는 max_seq_len(T)이 필요하다 — "
                "CD 정규화 상수가 (0, T) 이기 때문. --max_seq_len 을 주라.")
        reward_min_per_obj, reward_max_per_obj = cd_reward_bounds(
            all_raw_rewards.shape[-1], max_seq_len)
        print(f"  [CD] scale-only 정규화: reward_min={reward_min_per_obj}, "
              f"reward_max={reward_max_per_obj} (데이터셋 min/max 미사용)")
    elif reward_norm_scale is not None:
        # PEDA(D4MORL) 공식 코드의 실제 학습 설정 = raw / 1000 (shift 없음, env·tier 공통 상수) — 2026-10-27 ablation.
        # 같은 min–max 공식에 (0, scale) 을 넣으면 정확히 raw/scale 이 된다 (eval 도 config 의 이 상수를 쓴다).
        reward_min_per_obj = np.zeros(all_raw_rewards.shape[-1], np.float32)
        reward_max_per_obj = np.full(all_raw_rewards.shape[-1], reward_norm_scale, np.float32)
        print(f"  [norm] scale-only raw/{reward_norm_scale:g} (PEDA 코드 설정, 데이터셋 min/max 미사용)")
    else:
        reward_min_per_obj = all_raw_rewards.min(axis=0).astype(np.float32)
        reward_max_per_obj = all_raw_rewards.max(axis=0).astype(np.float32)
    def normalize_per_obj(rewards_per_step):
        # canonical min-max (utils) — scorer(reward_transform)들과 동일 공식·eps 공유 (SSOT)
        return min_max_normalization(rewards_per_step,
                                     reward_min_per_obj[None, :], reward_max_per_obj[None, :])

    # Optional top-K% CD filter — applied AFTER full-dataset reward_min/max are
    # computed, so the kept trajectories' returns live in the same normalization
    # space the AETDICE pipeline uses. Drops everything not in the top-K% by CD.
    if cd_filter is not None:
        from src.data.reward_transform import filter_top_k
        _kind = str(cd_filter.get("kind", "cd"))
        keep_idx = filter_top_k(
            trajectories_reshaped=[t['raw_rewards'] for t in trajectories],
            reward_min_per_obj=reward_min_per_obj,
            reward_max_per_obj=reward_max_per_obj,
            kind=_kind,
            rho=float(cd_filter["rho"]),
            k_pct=float(cd_filter["k_pct"]),
            reward_scaling_factor=reward_scaling_factor_arr,
        )
        _tag = (f"traj_filter kind={_kind} k_pct={cd_filter['k_pct']}"
                + (f" rho={cd_filter['rho']}" if _kind == "cd" else ""))
        keep_set = set(int(i) for i in keep_idx)
        kept = [trajectories[i] for i in sorted(keep_set)]
        print(f"  {_tag}: {len(trajectories)} → {len(kept)} trajectories")
        trajectories = kept

    # Optional trajectory-id filter (w-based coverage ablation, Exp3-03). Keeps only
    # trajectories whose ORIGINAL enumeration index ∈ keep_ids. keep_ids are computed
    # offline from a trained model's importance ratio w=d^π/d^D (see
    # docs/rebuttal/experiments/03_coverage/dataset_filtering.ipynb) and saved as .npy.
    # Applied AFTER full-dataset reward_min/max (like cd_filter) so eval normalization
    # matches the unfiltered pipeline. Mutually exclusive with cd_filter in practice.
    traj_z_arr = None
    if traj_filter is not None:
        keep_ids = traj_filter.get("keep_ids")
        if keep_ids is None:
            keep_ids = np.load(traj_filter["keep_ids_path"])
        keep_set = set(int(i) for i in np.asarray(keep_ids).ravel())
        kept = [trajectories[i] for i in range(len(trajectories)) if i in keep_set]
        if len(kept) == 0:
            raise ValueError(
                f"traj_filter kept 0 trajectories (keep_ids range vs "
                f"{len(trajectories)} trajs) — check the keep-list was built on this dataset."
            )
        print(f"  traj_filter: {len(trajectories)} → {len(kept)} trajectories "
              f"(keep_ids n={len(keep_set)})")
        # ── 에피소드 단위 잠재 z (관측 라벨) ──────────────────────────────────────
        # 왜 (2026-09-18): 커밋 장치가 필요하다. racc 는 개시가 구조적으로 불가능하고
        # (racc₀=0) 정책이 쓰지도 않는다(실험 G). 혼합 성분은 head(t) 사이 정체성이 없다
        # (실험 E). z 는 **전 구간 상수**라 정체성 문제가 없고, 학습 시 **관측값**으로 주면
        # posterior collapse 가 원리적으로 불가능하다. 전문가 2 개(13.4680)를 단일 망으로
        # 실현하는 것이 목표다.
        # ★ 원본 궤적 id 는 필터 직후 재열거되며 사라지므로 **여기서** 라벨을 붙여야 한다.
        kept_ids = [i for i in range(len(trajectories)) if i in keep_set]
        z_path = traj_filter.get("z_keeplist")
        if z_path:
            z_set = set(int(i) for i in np.asarray(np.load(z_path)).ravel())
            traj_z_arr = np.array([1.0 if i in z_set else 0.0 for i in kept_ids], np.float32)
            print(f"    z_keeplist: z=1 인 궤적 {int(traj_z_arr.sum())}/{len(traj_z_arr)}")
        trajectories = kept

    all_states = []
    all_next_states = []
    all_actions = []
    all_rewards = []
    all_traj_init_states = []  # one [state_dim] per trajectory (not tiled)
    all_traj_ids = []          # one int32 array [T] per trajectory, constant = traj_idx

    for traj_idx, traj in enumerate(trajectories):
        obs = traj['observations']
        next_obs = traj['next_observations']
        actions = traj['actions']
        raw_r = traj['raw_rewards'].astype(np.float32)

        rewards = reward_scaling_factor_arr[None, :] * normalize_per_obj(raw_r)

        T = obs.shape[0]

        all_states.append(obs.astype(np.float32))
        all_next_states.append(next_obs.astype(np.float32))
        all_actions.append(actions.astype(np.float32))
        all_rewards.append(rewards.astype(np.float32))
        all_traj_init_states.append(obs[0].astype(np.float32))
        all_traj_ids.append(np.full(T, traj_idx, dtype=np.int32))

    return {
        'states': np.concatenate(all_states, axis=0),
        'next_states': np.concatenate(all_next_states, axis=0),
        'actions': np.concatenate(all_actions, axis=0),
        'rewards': np.concatenate(all_rewards, axis=0),  # normalized per-step, pre-F.
        # Per-trajectory init states (compressed). Materialized at sample time via traj_ids lookup.
        'traj_init_states': np.stack(all_traj_init_states, axis=0),   # [num_traj, state_dim]
        'traj_ids':         np.concatenate(all_traj_ids, axis=0),     # [total_transitions]
        # 에피소드 단위 잠재 z (관측 라벨). traj_filter 에 z_keeplist 가 없으면 None.
        'traj_z':           traj_z_arr,                                # [num_traj] or None
        'reward_min': reward_min_per_obj,
        'reward_max': reward_max_per_obj,
        'reward_scaling_factor': reward_scaling_factor_arr,
    }


def inject_buffer_stats(config, buffer):
    """buffer 파생 통계를 config 에 주입 (네트워크·eval 이 소비). 주입 지점을 여기 하나로 두어
    새 stat 을 추가해도 소비처가 자동으로 함께 받는다 (SSOT)."""
    config.state_mean = buffer.state_mean
    config.state_std = buffer.state_std
    config.max_racc = buffer.max_racc
    config.min_racc = buffer.min_racc
    config.racc_mean = buffer.racc_mean
    config.racc_std = buffer.racc_std
    # F̃-output dim = 네트워크 racc 입력 크기. pointwise F 는 reward_dim 과 같지만 CobbDouglas 는 항상 2.
    config.racc_dim = int(np.asarray(buffer.racc_mean).shape[0])
    config.reward_min = buffer.reward_min
    config.reward_max = buffer.reward_max


def _utility_f_param_token(config):
    """utility_F 의 **인스턴스 파라미터**를 캐시 키에 넣는다.

    ★ 캐시는 raccs/rewards 를 **F̃ 변환된 상태로** 저장한다. 따라서 같은 f_type 이라도
      파라미터가 다르면 내용이 다르다. 예전 키는 `F{utility_F_type}` 까지만 봐서
      cobbdouglas 의 ρ 를 구분하지 못했다 → ρ=0.9 로 띄운 런이 ρ=0.1 로 만들어진
      캐시를 **HIT 해서 ρ=0.1 의 학습 신호로 훈련**됐다 (2026-09-16 발견, 오프라인 CD 19 런
      전부 오염). 로그에는 "Buffer cache HIT" 만 찍혀 조용히 지나갔다.
    cobbdouglas 외 F 는 파라미터가 없거나(linear) 이름에 이미 반영돼 토큰을 붙이지 않는다
    — 기존 캐시 이름을 보존하기 위해서다.
    """
    f_type = getattr(config, "utility_F_type", None)
    if f_type == "exponential_F":
        # F̃ = exp(x/S) − 1 이라 S 가 다르면 racc/rewards 내용이 통째로 다르다.
        # 기본 S=50 일 때만 토큰을 비워 기존 캐시를 보존한다 (cobbdouglas ρ 와 같은 규칙).
        scale = float(getattr(config, "exp_f_scale", 50.0))
        return "" if scale == 50.0 else f"_S{scale:g}"
    if f_type == "sigmoid":
        # τ·s 가 F̃ 를 통째로 바꾼다 (2026-10-25).
        tau = "-".join(f"{float(t):g}" for t in config.f_tau)
        return f"_tau{tau}_s{float(config.f_sigmoid_scale):g}"
    if f_type != "cobbdouglas":
        return ""
    rho = float(getattr(config, "cobbdouglas_rho", 0.5))
    amp = float(getattr(config, "cobbdouglas_amp", 1.0))
    tok = f"_rho{rho:g}"
    if amp != 1.0:
        tok += f"_amp{amp:g}"
    # rate 형태(exp A)는 rewards 내용이 다르다 — total(기본)은 토큰 없이 기존 캐시 이름을 보존한다.
    if (getattr(config, "cobbdouglas_form", None) or "total") != "total":
        tok += f"_cd{config.cobbdouglas_form}"
    # ★ 2026-11-03: 문턱·safety 결합 방식·κ 도 F̃ (저장되는 telescoped 보상) 를 바꾼다. 빠져 있어서
    #   product 런이 만든 캐시를 linear 런이 그대로 읽었다 (smoke 에서 발견). 문턱 누락은 그 전부터 있던 버그 —
    #   문턱만 다른 옛 safe-CD 런끼리 캐시를 공유했을 수 있다. 기본값이면 토큰을 비워 기존 캐시 이름을 보존한다.
    th = getattr(config, "reward_threshold", None)
    if th is not None:
        from src.run_naming import fmt_threshold
        tok += f"_th{fmt_threshold(th)}"
    safety = getattr(config, "cobbdouglas_safety", None) or "plog"
    if safety != "plog":
        tok += f"_sf{safety}"
    kappa = float(getattr(config, "cobbdouglas_kappa", None) or 1.0)
    if kappa != 1.0:
        tok += f"_k{kappa:g}"
    return tok


def bufcache_key(config, reshape_slot):
    """processed-buffer 캐시 디렉토리 경로. reshape_slot 만 경로별로 다르다(=config.reward_reshape).
    rsf-hash 등 나머지 캐시 결정자는 여기 한 곳에서만 정의 (SSOT: 한쪽에만 새
    결정자를 더하면 stale HIT/충돌). rsf token 은 문자열 hash() 가 프로세스마다 달라 결정적 인코딩 사용."""
    stem = os.path.splitext(os.path.basename(config.dataset_path))[0]
    rsf = np.asarray(config.reward_scaling_factor).ravel()
    rsf_h = (f"u{rsf.size}x{rsf[0]:g}" if np.all(rsf == rsf[0])
             else str(int(np.abs(np.round(rsf, 4)).sum() * 1000)))
    key = (f"{stem}_{reshape_slot}_ms{config.max_seq_len}"
           f"_F{config.utility_F_type}_mt{getattr(config, 'max_trajectories', None)}_rsf{rsf_h}"
           f"{_utility_f_param_token(config)}"
           # ★ 조건부 racc 인코딩은 **저장 내용을 바꾼다** — 키에 없으면 ftilde 캐시를
           #   rate 런이 그대로 읽어 조용히 틀린 입력으로 학습한다.
           f"{'' if getattr(config, 'racc_encoding', 'ftilde') == 'ftilde' else '_raccrate'}"
           # ★ 보상 정규화 상수도 저장 내용을 바꾼다 (min–max vs raw/scale).
           f"{'' if getattr(config, 'reward_norm_scale', None) is None else f'_rns{config.reward_norm_scale:g}'}")
    return os.path.join(os.path.dirname(config.dataset_path), "_bufcache", key)


class Buffer:
    def __init__(self, config, data, utility_F: Utility, cache_path=None):
        # utility_F is the INNER (per-step / trajectory-level) utility. The SHIFTED form
        # F̃(x) := F(x) - F(0) is what's actually stored (so F̃(0) = 0 and the initial-state
        # augmentation is zero). F̃ is absorbed into the augmented reward via the plain
        # telescoping F̃(R+r) - F̃(R); no first-step offset is needed. The OUTER utility G
        # does not appear here — it lives in the learner's conjugate_G and in eval scoring.
        self.utility_F = utility_F
        self.rng = np.random.default_rng(getattr(config, "seed", 0))
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._prefetch_stop = None
        self._prefetch_num_steps = None
        self._prefetch_batch_size = None

        # Processed-buffer cache. reshape + per-traj racc accumulation + F̃-transform +
        # stats are expensive (a 100-dim antpref100 dataset ≈ 30 min / 100 GB RAM to
        # build). They depend only on (dataset, reshape, max_seq_len, utility_F, rsf,
        # max_traj) — all fixed across an HP sweep — so ONE cache serves every run.
        # On a cache hit the big arrays are mmap'd (RAM stays low; sample() reads only
        # the gathered batch), and `data` is not touched.
        if cache_path is not None and os.path.exists(cache_path):
            self._load_cache(cache_path)
            self._apply_transition_filter(getattr(config, "transition_filter_path", None))
            return

        # Per-objective reward-normalization bookkeeping (exposed so eval mirrors
        # the same (r - reward_min) / (reward_max - reward_min) normalization).
        self.reward_min = data["reward_min"]
        self.reward_max = data["reward_max"]
        self.reward_scaling_factor = data["reward_scaling_factor"]

        states_np        = data["states"].astype(np.float32)
        next_states_np   = data["next_states"].astype(np.float32)
        traj_init_states_np = data["traj_init_states"].astype(np.float32)  # [num_traj, state_dim]
        traj_ids_np      = data["traj_ids"].astype(np.int32)                # [total_transitions]
        rewards_np       = data["rewards"].astype(np.float32)               # normalized per-step, pre-F.
        actions_raw      = data["actions"]

        self.size = states_np.shape[0]

        # Build raw per-trajectory raccs from normalized per-step rewards. traj_ids is
        # monotonically non-decreasing (from enumerate() in load_d4morl_dataset), so a
        # single pass with a reset at each traj boundary is correct and cheap (runs once).
        # Per-trajectory prefix sums, VECTORIZED (the old Python loop over N transitions
        # was ~O(minutes) at 2-D and pathological — tens of minutes — at reward_dim=100).
        # racc[i] = sum of rewards BEFORE step i within its trajectory (exclusive prefix).
        # Compute the global inclusive cumsum, then subtract the cumsum baseline captured
        # at each trajectory's start (traj_ids is monotonically non-decreasing).
        # CRITICAL: cumsum + baseline in float64. The global cumsum reaches ~15M at the
        # full d4morl scale (25M transitions × ~0.6/step); in float32 (ULP ~2 at 15M) the
        # `inc_all[i] - baseline` subtraction is catastrophic cancellation — the per-traj
        # racc (0..~400) is corrupted by up to ±200 and its min goes negative. That inflated
        # racc feeds F̃=exp(racc/50), so a ~+100 error becomes a ~7× blow-up of the training
        # signal (regression: Table-4 AET-DICE 13.69 → 13.16). float64 keeps full precision;
        # cast back to float32 only after the subtraction. The old per-traj-reset loop never
        # hit this (each trajectory's running sum stayed ≤~400).
        reward_dim = rewards_np.shape[1]
        inc_all = np.cumsum(rewards_np.astype(np.float64), axis=0)       # [N, D] global inclusive (f64)
        new_traj = np.ones(self.size, dtype=bool)
        if self.size > 1:
            new_traj[1:] = traj_ids_np[1:] != traj_ids_np[:-1]          # True at each traj start
        start_pos = np.flatnonzero(new_traj)                            # [num_traj] start indices
        baseline = np.zeros((start_pos.shape[0], reward_dim), dtype=np.float64)
        baseline[1:] = inc_all[start_pos[1:] - 1]                       # inclusive cumsum just before each start
        traj_idx = np.cumsum(new_traj) - 1                             # [N] 0-based traj index per transition
        per_traj_inc = inc_all - baseline[traj_idx]                    # [N, D] per-traj inclusive (f64)
        raw_next_raccs_np = per_traj_inc.astype(np.float32)            # racc AFTER step (inclusive)
        raw_raccs_np      = (per_traj_inc - rewards_np).astype(np.float32)  # racc BEFORE step (exclusive)

        # Per-transition timestep t within its trajectory (0-based). Needed for WBC head
        # alignment: each transition's action must be cloned only at ITS OWN timestep head,
        # not at all H heads (all-H cloning decorrelates racc/time from the action → the
        # policy collapses to π(a|s) and ignores racc). Vectorized: t = global_row − start.
        timesteps_np = (np.arange(self.size, dtype=np.int32)
                        - start_pos[traj_idx].astype(np.int32))       # [N] 0-based t within traj

        self.state_mean = np.mean(states_np, axis=0)
        self.state_std  = np.std(states_np,  axis=0) + 1e-8

        # Normalize states / per-trajectory init states on host.
        states_np      = (states_np      - self.state_mean) / self.state_std
        next_states_np = (next_states_np - self.state_mean) / self.state_std
        traj_init_states_np = (traj_init_states_np - self.state_mean) / self.state_std

        if not config.continuous_action:
            actions_np = actions_raw.astype(np.int32)
        else:
            action_scale = np.asarray(
                getattr(config, "action_scale", np.ones(actions_raw.shape[-1], dtype=np.float32)),
                dtype=np.float32,
            )
            actions_np = (actions_raw.astype(np.float32) / action_scale)

        # u0 = F(0). Subtracted from raccs/next_raccs so that the stored (F̃) values are 0
        # at trajectory start. Kept as an attribute for debug/inspection; not consumed elsewhere.
        self.u0 = np.asarray(self._call_F(
            jnp.zeros((1, config.reward_dim), dtype=jnp.float32), jnp.zeros((1,), jnp.int32)
        ).squeeze(0))

        # Apply F̃ = F - F(0) to raccs/next_raccs and form the telescoped per-step reward.
        # Chunked device round-trips: the full [N, reward_dim] arrays would need
        # 2·N·reward_dim·4 bytes on device at once — fine at reward_dim=2 but ~20 GB
        # (OOM) at reward_dim=100 (antpref100). Process _F_CHUNK rows per round-trip so
        # only ~1 GB is resident at a time; the host storage below still holds all N.
        _F_CHUNK = 2_000_000
        _n = raw_raccs_np.shape[0]
        _rp, _np2, _rw = [], [], []
        for _s in range(0, _n, _F_CHUNK):
            _e = min(_s + _F_CHUNK, _n)
            _a, _b, _c = self._f_transform(
                jnp.asarray(raw_raccs_np[_s:_e]), jnp.asarray(raw_next_raccs_np[_s:_e]),
                jnp.asarray(timesteps_np[_s:_e]), jnp.asarray(timesteps_np[_s:_e] + 1))
            _rp.append(np.asarray(_a)); _np2.append(np.asarray(_b)); _rw.append(np.asarray(_c))
        raccs_Ftilde_np      = np.concatenate(_rp,  axis=0)
        next_raccs_Ftilde_np = np.concatenate(_np2, axis=0)
        rewards_np           = np.concatenate(_rw,  axis=0)

        # ── 조건부 racc 인코딩 (reward_transform.RACC_ENCODINGS 참조) ──────────────
        # 'rate' 면 **조건부 입력만** R_t/(t+1) 로 바꾼다. rewards_np(텔레스코핑 F̃ 차)는
        # 목적함수라 건드리지 않는다 — 두 배열이 분리돼 있어서 가능한 최소 침습이다.
        from src.data.reward_transform import RACC_ENCODINGS, racc_rate
        _racc_enc = getattr(config, "racc_encoding", "ftilde")
        if _racc_enc not in RACC_ENCODINGS:
            raise ValueError(f"racc_encoding={_racc_enc!r}; {RACC_ENCODINGS} 중 하나여야 한다")
        if _racc_enc == "rate":
            raccs_Ftilde_np      = racc_rate(raw_raccs_np, timesteps_np)
            next_raccs_Ftilde_np = racc_rate(raw_next_raccs_np, timesteps_np + 1)
            print(f"  [racc] encoding='rate' — 조건부 입력을 R/(t+1) 로 교체 "
                  f"(per-step 보상은 F̃ 텔레스코핑 그대로)")

        # ── 잠재 z 를 조건부 입력에 덧붙인다 ────────────────────────────────────
        # racc 벡터 뒤에 z 열을 붙이는 **최소 변경**이다: racc_dim 이 자동으로 +1 되고
        # (config.racc_dim 은 racc_mean.shape 에서 파생), RaccEncoder·정책·critic 배선이
        # 그대로 흐른다. z 는 라벨이라 racc 인코딩 변환을 거치지 않고 그대로 붙는다
        # (뒤의 z-score 는 racc 통계와 같이 계산되어 ±1 규모가 된다).
        _traj_z = data.get("traj_z") if isinstance(data, dict) else None
        if _traj_z is not None:
            _z_col = np.asarray(_traj_z, np.float32)[traj_ids_np].reshape(-1, 1)
            raccs_Ftilde_np      = np.concatenate([raccs_Ftilde_np, _z_col], axis=1)
            next_raccs_Ftilde_np = np.concatenate([next_raccs_Ftilde_np, _z_col], axis=1)
            print(f"  [z] 조건부 입력에 z 열 추가 → racc_dim {raccs_Ftilde_np.shape[1]} "
                  f"(z=1 비율 {float(_z_col.mean()):.3f})")

        # Stats on the conditioning raccs — RaccEncoder (in policy/critic) normalizes its
        # racc input by (racc_mean, racc_std), so these must match what the network sees.
        self.max_racc   = np.max(raccs_Ftilde_np,  axis=0)
        self.min_racc   = np.min(raccs_Ftilde_np,  axis=0)
        self.racc_mean  = np.mean(raccs_Ftilde_np, axis=0)
        self.racc_std   = np.std(raccs_Ftilde_np,  axis=0) + 1e-8

        # All storage lives on host now. `sample()` gathers + device_puts the batch.
        self.traj_init_states = traj_init_states_np  # [num_traj, state_dim] on host
        self.data = _StoredArrays(
            states_np,
            next_states_np,
            actions_np,
            rewards_np,
            traj_ids_np,
            raccs_Ftilde_np,
            next_raccs_Ftilde_np,
            timesteps_np,
        )

        # rng + prefetch state already initialized at the top of __init__.
        if cache_path is not None:
            self._save_cache(cache_path)
        # ★ 캐시 저장 **이후**에 적용한다 — 캐시는 필터 없는 전체 버퍼를 담아 마스크마다
        #   재빌드하지 않는다 (그래서 bufcache_key 에 마스크를 넣지 않는다).
        self._apply_transition_filter(getattr(config, "transition_filter_path", None))

    def _apply_transition_filter(self, path):
        """IDRL(arXiv:2504.13368) 식 **transition 단위** 필터를 적용한다.

        원논문 Algorithm 1: D_{k+1} = {w*_k(s,a) > 0 인 support}. 우리 w = d^π/d^D 가
        정확히 그 state-action visitation ratio 다 (논문 ablation: w(a|s) 로 필터하면
        90.8 → 56.8 로 무너진다 — 반드시 w(s,a) 여야 한다).

        racc 계산 **이후**에 적용해야 한다. racc 는 전체 궤적의 exclusive prefix-sum 으로
        빌드 때 계산돼 각 transition 에 **라벨로** 붙으므로, 여기서 부분집합을 취해도 남은
        transition 의 racc 는 그대로 유효하다. (궤적 단위 traj_filter 와는 granularity 가
        달라 별개 메커니즘이다 — 궤적 단위는 500 스텝 중 하나만 w>0 이어도 살아남아
        MO-Ant 에서 0 개가 삭제되는 공집합 필터였다.)
        """
        if not path:
            return
        keep = np.load(path)
        assert keep.dtype == np.bool_ and keep.shape == (self.size,), \
            f"transition 마스크 불일치: {keep.shape}/{keep.dtype} vs size={self.size}"
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            raise SystemExit(f"transition_filter 가 0 개를 남겼다: {path}")
        self.data = _StoredArrays(*(np.asarray(getattr(self.data, k))[idx]
                                    for k in _StoredArrays._fields))
        self.size = int(idx.size)
        print(f"[buffer] transition_filter: {len(keep):,} → {self.size:,} "
              f"({100 * self.size / len(keep):.2f}%) <- {path}", flush=True)

    # ── processed-buffer cache (dir of .npy; big arrays mmap'd on load) ───────────
    _CACHE_BIG = ("states", "next_states", "actions", "rewards", "traj_ids", "raccs", "next_raccs",
                  "timesteps")
    _CACHE_SMALL = ("traj_init_states", "reward_min", "reward_max", "reward_scaling_factor",
                    "state_mean", "state_std", "u0", "max_racc", "min_racc", "racc_mean", "racc_std")

    def _save_cache(self, path):
        # per-process tmp: runs that differ only in seed build the same cache concurrently
        # (a shared ".tmp" once made 5 of 15 parallel runs crash mid-save).
        tmp = f"{path}.tmp.{os.getpid()}"
        os.makedirs(tmp, exist_ok=False)
        d = self.data
        big = dict(states=d.states, next_states=d.next_states, actions=d.actions,
                   rewards=d.rewards, traj_ids=d.traj_ids, raccs=d.raccs, next_raccs=d.next_raccs,
                   timesteps=d.timesteps)
        for k, v in big.items():
            np.save(os.path.join(tmp, k + ".npy"), v)
        small = dict(traj_init_states=self.traj_init_states,
                     reward_min=np.asarray(self.reward_min), reward_max=np.asarray(self.reward_max),
                     reward_scaling_factor=np.asarray(self.reward_scaling_factor),
                     state_mean=self.state_mean, state_std=self.state_std, u0=self.u0,
                     max_racc=self.max_racc, min_racc=self.min_racc,
                     racc_mean=self.racc_mean, racc_std=self.racc_std)
        for k, v in small.items():
            np.save(os.path.join(tmp, k + ".npy"), np.asarray(v))
        np.save(os.path.join(tmp, "size.npy"), np.asarray(self.size))
        try:
            os.rename(tmp, path)  # atomic publish; fails if another process published first
        except OSError as e:
            if e.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # same cache key = same deterministic content, so the other writer's copy is equivalent
            shutil.rmtree(tmp)
            print(f"[buffer] cache already published by another process -> {path} (own copy dropped)",
                  flush=True)
            return
        print(f"[buffer] saved processed cache -> {path}", flush=True)

    def _load_cache(self, path):
        L = lambda k, mmap=False: np.load(os.path.join(path, k + ".npy"),
                                          mmap_mode="r" if mmap else None)
        self.data = _StoredArrays(*(L(k, mmap=True) for k in self._CACHE_BIG))
        for k in self._CACHE_SMALL:
            setattr(self, k, L(k))
        self.size = int(L("size"))
        print(f"[buffer] loaded processed cache <- {path} ({self.size} transitions, mmap)", flush=True)

    def _prefetch_producer(self, num_steps, batch_size):
        # Runs in a daemon thread. Builds CHUNKS of `num_steps` batches each: every
        # field is shape [num_steps, batch_size, ...]. Numpy advanced indexing with a
        # 2D index array of shape (num_steps, batch_size) does the gather in one call.
        # device_put transfers the whole chunk in one async H2D copy, which the GPU
        # consumes via jax.lax.scan inside one fused compiled kernel.
        # The body is guarded so exceptions raised during interpreter / JAX shutdown
        # (runtime vanishing under device_put) get swallowed cleanly instead of
        # crashing with "FATAL: exception not rethrown".
        assert self._prefetch_stop is not None and self._prefetch_queue is not None
        stop_event = self._prefetch_stop
        q = self._prefetch_queue
        try:
            while not stop_event.is_set():
                idx = self.rng.integers(0, self.size, size=(num_steps, batch_size))
                s = self.data
                chunk_host = Batch(
                    states       = s.states[idx],          # [M, B, state_dim]
                    next_states  = s.next_states[idx],
                    actions      = s.actions[idx],
                    rewards      = s.rewards[idx],
                    init_states  = self.traj_init_states[s.traj_ids[idx]],
                    traj_ids     = s.traj_ids[idx],
                    raccs        = s.raccs[idx],
                    next_raccs   = s.next_raccs[idx],
                    timesteps    = s.timesteps[idx],       # [M, B] int32, t per transition
                )
                chunk_device = jax.device_put(chunk_host)
                # Short timeout on put() so we exit promptly on shutdown even when
                # the consumer has stopped draining the queue.
                while not stop_event.is_set():
                    try:
                        q.put(chunk_device, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception:
            if not stop_event.is_set():
                raise  # unexpected mid-run failure; surface it
            # otherwise: normal shutdown race, drop it silently

    def _start_prefetch(self, num_steps, batch_size, depth=2):
        self._prefetch_num_steps = num_steps
        self._prefetch_batch_size = batch_size
        self._prefetch_queue = queue.Queue(maxsize=depth)
        self._prefetch_stop = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_producer, args=(num_steps, batch_size), daemon=True
        )
        self._prefetch_thread.start()

    def close(self):
        # Graceful shutdown: signal stop, drain the queue so the producer doesn't
        # block on put(), and join briefly. daemon=True is a backstop if close() is
        # never called, but calling it avoids the JAX-teardown-in-thread race.
        if self._prefetch_stop is None:
            return
        self._prefetch_stop.set()
        if self._prefetch_queue is not None:
            try:
                while True:
                    self._prefetch_queue.get_nowait()
            except queue.Empty:
                pass
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1.0)

    def _call_F(self, raw, steps):
        """utility_F 호출 — 스텝 수가 필요한 F(CD form='rate')에만 steps 를 넘긴다."""
        if getattr(self.utility_F, "needs_steps", False):
            return self.utility_F(raw, keep_dims=True, steps=steps)
        return self.utility_F(raw, keep_dims=True)

    def _f_transform(self, raw_raccs, raw_next_raccs, steps, next_steps):
        """Apply the shifted utility F̃(x) := F(x) - F(0) and form the telescoped reward.

        Runs under JAX because self.utility_F is a JAX callable. Called once from __init__
        on device-resident inputs; results are pulled back to host by the caller.

        Returns (raccs_Ftilde, next_raccs_Ftilde, telescoped_rewards):
            raccs_Ftilde       = F(raw_raccs)        - F(0)                   [N, D]
            next_raccs_Ftilde  = F(raw_next_raccs)   - F(0)                   [N, D]
            telescoped_rewards = F̃(R_{t+1}) - F̃(R_t)  (= F(R_{t+1}) - F(R_t), u0 cancels)
        so the trajectory-wise sum of telescoped_rewards equals F̃(R_final) = F(R_final) - F(0).
        At any trajectory start (R_t = 0), raccs_Ftilde = 0 — no special-case offset needed.
        steps / next_steps: raw_raccs / raw_next_raccs 가 누적한 스텝 수 (t, t+1) — rate 형태 CD 용.
        """
        raccs_F      = self._call_F(raw_raccs,      steps)
        next_raccs_F = self._call_F(raw_next_raccs, next_steps)
        u0_j = jnp.asarray(self.u0)
        raccs_Ftilde      = raccs_F      - u0_j[None, :]
        next_raccs_Ftilde = next_raccs_F - u0_j[None, :]
        telescoped        = next_raccs_Ftilde - raccs_Ftilde
        return raccs_Ftilde, next_raccs_Ftilde, telescoped

    def sample_chunk(self, num_steps, batch_size):
        """Pops a pre-gathered, device-put CHUNK from the prefetch queue.

        Returned Batch has each field shaped [num_steps, batch_size, ...] — feed it
        to a jax.lax.scan-based train loop to fuse num_steps train steps into one
        compiled call.

        On the first call, lazily spawns a background producer thread sized to
        (num_steps, batch_size). Subsequent calls must use the same shape (the
        thread is pinned to it); otherwise raises ValueError.
        """
        if self._prefetch_queue is None:
            self._start_prefetch(num_steps, batch_size)
        elif (num_steps, batch_size) != (self._prefetch_num_steps, self._prefetch_batch_size):
            raise ValueError(
                f"Buffer.sample_chunk(num_steps={num_steps}, batch_size={batch_size}) but "
                f"prefetch was started with (num_steps={self._prefetch_num_steps}, "
                f"batch_size={self._prefetch_batch_size}). Variable shapes not supported."
            )
        return self._prefetch_queue.get()
