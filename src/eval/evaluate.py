"""Evaluation loop for continuous-action multi-objective environments (e.g. Walker2d).

Key differences from evaluate.py:
  - action is a continuous numpy array, not a discrete scalar
  - multi-objective rewards come from info['obj'], not the scalar reward return

Performance note: eval runs batch=1 per-step policy calls, which are dispatch-bound
not compute-bound. We move the policy (small MLP) onto CPU for eval and run all
jnp ops under a CPU default-device context — this collapses per-step H2D/D2H
round-trips, which are the real cost of eval on a GPU machine.
"""
import numpy as np

from src.utils import min_max_normalization
import jax
import jax.numpy as jnp
import wandb
from flax import nnx
from src.utility import Utility
# transform_reward is unplugged in the transformed pipeline (matches buffer.py).


def _move_to_device(module, device):
    """Relocate every jax.Array inside an nnx module onto `device`.
    Returns a new module (doesn't mutate the caller's)."""
    graphdef, state = nnx.split(module)
    state = jax.tree.map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x,
        state,
    )
    return nnx.merge(graphdef, state)


def evaluate(
    config,
    policy,
    env,
    save_dir,
    num_episodes=30,
    max_steps=500,
    t_env=None,
    key=jax.random.PRNGKey(0),
    reward_reshape: str = "none",
    action_noise_std: float = 0.0,
    disable_done: bool = False,
    return_racc_traj: bool = False,
    capture_positions: bool = False,
    deterministic: bool = False,
    det_mode: str = "mean",
    eval_batch: int = 0,
    mixture_commit: bool = False,
    score_reshape: str = "none",
):
    """Run `num_episodes` eval rollouts and score them in F̃ space.

    `action_noise_std` and `disable_done` are distribution-shift knobs for
    post-hoc robustness comparison (BC's narrow imitation vs DICE's wider d^π).
    Both default to no-op so existing callers and saved-dashboard behavior are
    unchanged.

    `action_noise_std`: σ of Gaussian noise added to the policy's sampled
    action (after action_scale, before env.step). 0 = clean eval (default).
    Use 0.05–0.2 for moderate shift.

    `disable_done`: ignore env.step's `done` flag; episode runs the full
    `max_steps` regardless. Tests "drop & die" policies that exploit early
    termination — once disable_done is on, a fallen walker keeps absorbing
    reward=0 (no velocity) + cost=force² for the remainder of the episode.
    """
    # Resolve the per-step reshape function up front (None if reward_reshape == 'none').
    # Buffer applied the matching per-trajectory reshape at load time, so reward_min /
    # reward_max / reward_scaling_factor / racc_mean / racc_std on `config` already
    # reflect the post-reshape space. Eval just has to translate env.step()'s 2-obj
    # output into the same space.
    if reward_reshape != "none":
        from src.data.reward_transform import RESHAPE_STEP
        if reward_reshape not in RESHAPE_STEP:
            raise ValueError(
                f"Unknown reward_reshape={reward_reshape!r}. "
                f"Supported: 'none' | {sorted(RESHAPE_STEP.keys())}"
            )
        _reshape_step = RESHAPE_STEP[reward_reshape]
    else:
        _reshape_step = None

    # 채점 전용 reshape (2026-11-05, D6.24 — 사용자: "Re-evaluate in safety env"): safety 채널 없이
    # 학습한 기준선을 safe-CD 효용으로 채점하려고, 정책 입력(racc)은 학습 때 reshape 그대로 두고
    # 같은 스텝에서 score_reshape 벡터를 **따로** 누적한다 (정책이 보는 공간은 바뀌지 않는다).
    if score_reshape != "none":
        from src.data.reward_transform import RESHAPE_STEP, CD_RESHAPES
        if score_reshape not in RESHAPE_STEP:
            raise ValueError(f"Unknown score_reshape={score_reshape!r}. Supported: {sorted(RESHAPE_STEP.keys())}")
        # safe-CD reshape 는 [CD 2 채널 + safety] 라 앞 2 채널이 학습 reshape 와 같아야 비교가 맞다.
        if not (reward_reshape in CD_RESHAPES and score_reshape in CD_RESHAPES):
            raise ValueError("score_reshape 는 CD 계열 reshape 끼리만 쓴다 (앞 2 채널 = (reward, cost))")
        _score_step = RESHAPE_STEP[score_reshape]
    else:
        _score_step = None
    score_returns = []

    # Pin eval to CPU: batch=1 policy calls are latency-bound on GPU (the H2D + D2H
    # sync per step dominates actual compute for a small MLP). CPU forward runs in
    # a few hundred microseconds and avoids the round-trip entirely. Move the policy
    # to CPU once, then wrap the whole per-step loop in jax.default_device(cpu) so
    # every jnp.array / jnp.asarray / jax.jit created inside also defaults to CPU.
    cpu = jax.devices("cpu")[0]
    policy = _move_to_device(policy, cpu)
    policy.eval()

    raw_returns = []
    norm_returns = []
    ep_lengths = []     # 에피소드 길이 — rate 형태 CD 점수와 조기 종료 비율용 (2026-10-24)

    # Per-objective reward normalization — MUST mirror load_d4morl_dataset exactly so
    # the raw accumulated return R_t we build here matches what Buffer fed into utility_F.
    reward_min_np = np.asarray(config.reward_min, dtype=np.float32)
    reward_max_np = np.asarray(config.reward_max, dtype=np.float32)
    reward_scaling_np = np.asarray(config.reward_scaling_factor, dtype=np.float32)

    # F̃(x) := F(x) - F(0) must be applied to raccs before feeding the policy — Buffer
    # stores F̃-transformed raccs, and RaccEncoder's (racc_mean, racc_std) were fit on that space.
    # FairDICE policy is state-only and ignores raccs entirely (skips the F̃ build below).
    utility_F = config.utility_F
    # 조건부 racc 인코딩 — Buffer 와 같은 값을 써야 한다 (다르면 조용히 다른 정책이 된다).
    _racc_encoding = getattr(config, "racc_encoding", "ftilde")
    # 에피소드 단위 잠재 z — Buffer 가 racc 벡터 뒤에 z 열을 붙였으면 eval 도 붙여야 한다.
    # racc_dim(= 저장된 조건부 입력 폭)과 reward_dim 의 차이로 몇 열인지 알 수 있다.
    _z_extra = int(getattr(config, "racc_dim", config.reward_dim)) - int(config.reward_dim)
    _z_val = float(getattr(config, "z_eval_value", 0.0))
    if _z_extra > 0:
        print(f"  [z] eval 에 z={_z_val:g} 열 {_z_extra} 개를 붙인다 "
              f"(racc_dim={config.racc_dim}, reward_dim={config.reward_dim})")
    # Captured as a Python-time constant in the select_action closure so the branch
    # below resolves at trace time (no runtime dispatch). Default "aet" preserves
    # behavior for old config.json files written before this flag existed.
    learner_type = getattr(config, "learner_type", "aet")
    # ★ 분기 기준은 learner 이름이 아니라 **정책이 racc 를 받는가** 다 (2026-09-16).
    #   BC 가 --bc_conditioning=racc 로도 돌 수 있게 된 뒤로 learner_type 만 보면 틀린다
    #   (racc-BC 를 state-only 로 호출해 TypeError 가 나거나, 더 나쁘게는 조용히 다른
    #   정책을 복원한다). learner 별 판정은 그 learner 모듈이 소유한다 (SSOT).
    from src.learners import is_state_only_policy
    _state_only_policy = is_state_only_policy(config)

    # Everything inside this block runs on CPU (jnp calls + jit compilation).
    with jax.default_device(cpu):
        state_mean = jnp.array(config.state_mean)
        state_std = jnp.array(config.state_std)
        action_scale = jnp.asarray(
            getattr(config, "action_scale", np.ones(config.action_dim, dtype=np.float32)),
            dtype=jnp.float32,
        )
        key = jax.device_put(key, cpu)

        # u0 = F(0), precomputed once; subtracted inside select_action to form F̃.
        # ftilde 조건부 경로에서만 쓴다 — rate 인코딩은 F 를 거치지 않고, rate 형태 CD 의 F 는
        # 스텝 수 없이 부를 수 없다 (2026-10-24, exp A).
        u0 = (utility_F(jnp.zeros((1, config.reward_dim), dtype=jnp.float32), keep_dims=True)
              if _racc_encoding == "ftilde" else None)  # [1, reward_dim]

        # Single eval path for the transformed pipeline (exponential_F × {linear, log} G scores).

        @jax.jit
        def select_action(state, racc_raw, timestep, rng_key, commit_k=None):
            state_b = jnp.expand_dims(state, 0)
            time_b = jnp.array([timestep])
            state_b = (state_b - state_mean) / state_std

            if _state_only_policy:
                # State-only policies — racc_raw is unused (still threaded
                # through the call signature so the per-step loop stays uniform).
                dist = policy(state_b, timesteps=time_b)
            else:
                # AET / IQL: 조건부 racc 는 Buffer 가 저장한 것과 **정확히 같은 공간**이어야
                # 한다. ftilde = F̃(R)=F(R)−F(0), rate = R/(t+1) (reward_transform 참조).
                racc_raw_b = jnp.expand_dims(racc_raw, 0)
                if _racc_encoding == "rate":
                    racc_cond_b = racc_raw_b / jnp.maximum(timestep.astype(jnp.float32), 1.0)
                else:
                    racc_cond_b = utility_F(racc_raw_b, keep_dims=True) - u0
                if _z_extra > 0:
                    racc_cond_b = jnp.concatenate(
                        [racc_cond_b, jnp.full((1, _z_extra), _z_val, racc_cond_b.dtype)], -1)
                dist = policy(state_b, racc_cond_b, timesteps=time_b)

            # deterministic → policy's actual visitation without sampling noise
            # (coworker's R3-Q1 request). Two flavors of "deterministic action":
            #   det_mode="mean" → mixture mean E[a] (dist.mean()).
            #   det_mode="mode" → highest-weight mixture component's mean (mode approx),
            #                     better when the mixture is multimodal.
            if deterministic and det_mode == "mode":
                comp_means = dist.components_distribution.mean()                    # [1, M, A]
                k_star = jnp.argmax(dist.mixture_distribution.logits_parameter(), axis=-1)  # [1]
                action = jnp.take_along_axis(
                    comp_means, k_star[:, None, None], axis=1).squeeze(1)           # [1, A]
            elif deterministic:
                action = dist.mean()                                               # [1, A]
            elif commit_k is not None:
                # ★ 혼합 성분 **에피소드 고정** (mixture_commit, 2026-09-21).
                # 왜: AET 는 비선형 목적이라 최적 정책이 **정책들의 혼합**이다 — 에피소드
                # 시작에 하나를 뽑아 끝까지 따르는 것 (Zahavy et al., Reward is Enough for
                # Convex MDPs, NeurIPS 2021). MixtureSameFamily.sample() 은 **매 스텝**
                # 성분을 새로 뽑으므로 Markov 확률정책이 되고, 그런 목적에서는 증명된
                # 준최적이다. 스텝마다 섞이면 특화 행동이 평균으로 뭉개진다.
                comp = dist.components_distribution                                # [1, M, A]
                a_all = comp.sample(seed=rng_key)                                  # [1, M, A]
                action = jnp.take_along_axis(
                    a_all, commit_k[None, None, None], axis=1).squeeze(1)          # [1, A]
            else:
                action = dist.sample(seed=rng_key)                                 # [1, A]
            action = jnp.clip(jnp.squeeze(action), -1.0, 1.0)
            action = action * action_scale
            return action  # [action_dim]

        # Optional per-step augmented-state visitation (t, R^acc_t) collection for
        # coverage diagnostics (R3-Q1). Records the EXCLUSIVE racc fed to the policy at
        # each step (= R^acc_t = Σ_{k<t} r_k), matching the buffer's stored convention.
        racc_visits = []  # each row: [t, y_0, ..., y_{reward_dim-1}]
        # 같은 스텝의 raw 관측 — 방문 상태가 데이터 support 안인지 재는 진단용 (2026-10-27, CD §10 D1).
        state_visits = []  # each row: raw obs [obs_dim], racc_visits 와 같은 순서
        # Optional per-episode (x,y) torso trajectory (viz — heading overlay).
        # Displacement from the episode's start pos, matching the buffer-side pos_0=0 space.
        pos_trajs = []

        # ── 배치 롤아웃 (기본 경로) ──────────────────────────────────────────────
        # 왜: eval 한계비용 실측(2026-09-10) 결과 **94% 가 정책 forward**였다
        # (1678 ms/ep vs 물리 111 ms/ep). `original` 정책은 매 스텝 H=500 head 를 전부
        # 계산하는데 그걸 batch=1 로 돌리면 CPU 가 완전히 놀고 dispatch 만 먹는다.
        # 에피소드를 묶어 한 번에 forward 하면 같은 계산을 배치로 분할상환한다.
        #
        # ★ RNG 규약이 바뀐다: 예전에는 key 를 **모든 에피소드·스텝에 걸쳐 하나의 스트림**으로
        #   흘렸다. 이제 (배치 그룹, 스텝)으로 fold_in 한다. 따라서
        #     - 같은 batch 크기로 다시 돌리면 **완전히 재현**된다.
        #     - **batch 크기를 바꾸면 표본이 달라진다** (배치 MoG 의 sample 이 배치 내 위치에
        #       의존하기 때문). 분포는 동일하니 점수는 통계적으로 같지만 숫자는 달라진다.
        #   ⇒ 예전 숫자와도 다르다. eval 재실행이 필요하다.
        #
        # 특수 진단 모드(위치 캡처 / racc 궤적 / reshape / 행동 노이즈)는 순차 경로를 그대로 쓴다.
        # mixture_commit 도 에피소드마다 성분을 고정해야 해서 순차 경로를 쓴다.
        _batchable = (not capture_positions and not return_racc_traj
                      and reward_reshape == "none" and score_reshape == "none" and action_noise_std == 0.0
                      and not disable_done and not mixture_commit)
        if _batchable:
            B = int(eval_batch) if eval_batch and eval_batch > 0 else num_episodes
            B = max(1, min(B, num_episodes))

            @jax.jit
            def select_action_batch(states, raccs_raw, timestep, rng_key):
                """[B, obs] → [B, act]. 단일판 select_action 과 **같은 식**이다."""
                sb = (states - state_mean) / state_std
                tb = jnp.full((states.shape[0],), timestep, jnp.int32)
                if _state_only_policy:
                    dist = policy(sb, timesteps=tb)
                else:
                    rc = (raccs_raw / jnp.maximum(jnp.float32(timestep), 1.0)
                          if _racc_encoding == "rate"
                          else utility_F(raccs_raw, keep_dims=True) - u0)
                    if _z_extra > 0:
                        rc = jnp.concatenate(
                            [rc, jnp.full((rc.shape[0], _z_extra), _z_val, rc.dtype)], -1)
                    dist = policy(sb, rc, timesteps=tb)
                if deterministic and det_mode == "mode":
                    cm = dist.components_distribution.mean()
                    ks = jnp.argmax(dist.mixture_distribution.logits_parameter(), axis=-1)
                    act = jnp.take_along_axis(cm, ks[:, None, None], axis=1).squeeze(1)
                elif deterministic:
                    act = dist.mean()
                else:
                    act = dist.sample(seed=rng_key)
                return jnp.clip(act, -1.0, 1.0) * action_scale

            envs = [env] + [type(env)() for _ in range(B - 1)]   # env 인스턴스 B 개 (1 회 생성)
            for g0 in range(0, num_episodes, B):
                eps = list(range(g0, min(g0 + B, num_episodes)))
                n = len(eps)
                obs0 = []
                for j, e in enumerate(eps):
                    envs[j].seed(e)          # ★ `seed(e) or reset()` 로 쓰면 안 된다 —
                    obs0.append(envs[j].reset())   # gym 의 seed() 는 리스트를 반환해 truthy 라
                states = np.stack(obs0).astype(np.float32)   # reset() 이 단축평가로 건너뛰어진다.
                raccs = np.zeros((n, config.reward_dim), np.float32)
                raw_sum = np.zeros((n, config.reward_dim), np.float32)
                alive = np.ones((n,), np.float32)
                lens = np.zeros((n,), np.int32)
                for t in range(max_steps):
                    acts = np.asarray(select_action_batch(
                        jnp.asarray(states), jnp.asarray(raccs), t,
                        jax.random.fold_in(jax.random.fold_in(key, g0), t)))
                    nxt = []
                    for j in range(n):
                        ob, _, dn, info = envs[j].step(acts[j])
                        obj = np.asarray(info["obj"], np.float32)
                        raw_sum[j] += alive[j] * obj
                        raccs[j] += alive[j] * reward_scaling_np * min_max_normalization(
                            obj, reward_min_np, reward_max_np)
                        lens[j] += int(alive[j])
                        alive[j] *= 0.0 if dn else 1.0
                        nxt.append(ob)
                    states = np.stack(nxt).astype(np.float32)
                for j in range(n):
                    raw_returns.append(raw_sum[j].copy())
                    norm_returns.append(raccs[j].copy())
                    ep_lengths.append(int(lens[j]))

        for iter in range(0 if not _batchable else num_episodes, num_episodes):
            env.seed(iter)
            state = env.reset()
            # stateful per-step reshape (위치 누적형)은 에피소드마다 원점화.
            if _reshape_step is not None and hasattr(_reshape_step, "reset"):
                _reshape_step.reset()
            if _score_step is not None and hasattr(_score_step, "reset"):
                _score_step.reset()
            score_rewards_list = []
            if capture_positions:
                start_xy = np.asarray(env.get_body_com("torso"), dtype=np.float32)[:2].copy()
                ep_pos = []

            done = False
            steps = 0
            # Raw (pre-F) normalized accumulated return. select_action applies F internally.
            raccs_raw = np.zeros((config.reward_dim,), dtype=np.float32)

            # ★ 혼합 성분을 **에피소드 시작에 한 번** 뽑아 끝까지 고정 (mixture_commit).
            #   AET 는 비선형 목적이라 최적 정책이 정책들의 혼합이다 (Zahavy et al. 2021).
            #   기본 경로(dist.sample)는 매 스텝 성분을 새로 뽑아 Markov 확률정책이 된다.
            ep_commit_k = None
            if mixture_commit:
                key, ck = jax.random.split(key)
                n_mix = int(getattr(policy, "n_mixture", 1))
                ep_commit_k = jax.random.randint(ck, (), 0, n_mix)

            raw_rewards_list = []
            norm_rewards_list = []

            # `disable_done` ignores env termination so episodes always run the
            # full `max_steps`. Tests early-termination exploits ("drop & die"):
            # post-fall steps in walker2d still produce cost (cost channel = force²) but
            # zero velocity (reward channel = 0), eroding any policy that wins by ending early.
            while (not done or disable_done) and steps < max_steps:
                key, subkey = jax.random.split(key)

                state_jax = jnp.array(state, dtype=jnp.float32)
                racc_raw_jax = jnp.array(raccs_raw, dtype=jnp.float32)

                if return_racc_traj:
                    # racc BEFORE this step's reward = R^acc_t (what the policy conditions on).
                    racc_visits.append(np.concatenate([[steps], raccs_raw.copy()]))
                    state_visits.append(np.asarray(state, dtype=np.float32).copy())

                action = np.array(select_action(state_jax, racc_raw_jax, steps, subkey,
                                                ep_commit_k))

                # Action-noise injection: simulates deployment-time perturbation
                # where the executed action diverges from the policy's intended
                # output. BC's narrow imitation should degrade faster than the
                # data-supported wider d^π that AET maintains.
                if action_noise_std > 0.0:
                    action = action + np.random.normal(
                        loc=0.0, scale=action_noise_std, size=action.shape
                    ).astype(np.float32)

                next_obs, _, done, info = env.step(action)
                env_obj = np.asarray(info['obj'], dtype=np.float32)
                if capture_positions:
                    ep_pos.append(np.asarray(env.get_body_com("torso"), dtype=np.float32)[:2] - start_xy)
                # Apply same per-step reshape the buffer used at load time, so eval-side
                # raw_rewards / raccs live in the same space the policy was trained on.
                if reward_reshape != "none":
                    raw_reward = _reshape_step(action, env_obj, next_obs)
                else:
                    raw_reward = env_obj
                if _score_step is not None:
                    score_reward = np.asarray(_score_step(action, env_obj, next_obs), dtype=np.float32)
                    assert np.allclose(score_reward[:2], raw_reward[:2]), (score_reward, raw_reward)
                    score_rewards_list.append(score_reward)
                # buffer(load_d4morl_dataset)와 동일 공식·eps 공유 (canonical) — 학습/평가 racc 공간 일치
                norm_reward = reward_scaling_np * min_max_normalization(raw_reward, reward_min_np, reward_max_np)

                raw_rewards_list.append(raw_reward)
                norm_rewards_list.append(norm_reward)
                state = next_obs
                raccs_raw += norm_reward
                steps += 1

            raw_returns.append(np.sum(raw_rewards_list, axis=0))
            norm_returns.append(np.sum(norm_rewards_list, axis=0))
            ep_lengths.append(int(steps))
            if _score_step is not None:
                score_returns.append(np.sum(score_rewards_list, axis=0))
            if capture_positions:
                pos_trajs.append(np.asarray(ep_pos, dtype=np.float32))  # [T, 2]

    return_array = np.array(raw_returns)        # [num_episodes, reward_dim], raw
    return_array_norm = np.array(norm_returns)  # [num_episodes, reward_dim], normalized
    max_return       = np.max(return_array, axis=0)
    min_return       = np.min(return_array, axis=0)
    avg_returns      = np.mean(return_array, axis=0)
    std_returns      = np.std(return_array, axis=0, ddof=0)
    # Normalized mirror — same space the policy was trained on AND the same space
    # data_analysis/* notebooks operate in. Logging both makes wandb directly
    # comparable to the post-hoc CD analysis without recomputing (r-min)/span.
    avg_returns_norm = np.mean(return_array_norm, axis=0)
    std_returns_norm = np.std(return_array_norm, axis=0, ddof=0)

    # All scores are E[G(F̃(R))] with F̃(x) = F(x) - F(0); the F̃ shift matches the
    # Buffer's stored augmented reward so values are directly comparable to the
    # training objective. Scores operate on NORMALIZED returns (the scale the Buffer
    # normalizes to). Each row is (name, G-utility, F-utility).
    #
    # Baselines (10 total):
    #   - score_OptiDICE  : G-linear  × F-linear × 3 pref weights
    #   - score_FairDICE  : G-log     × F-linear, single Nash-style score
    #   - score_linear-log: G-linear  × F-log    × 3 pref weights
    #   - score_linear-exp: G-linear  × F-exp    × 3 pref weights
    # AETDICE objective scores (2 total):
    #   - score_G-log_F-log
    #   - score_G-log_F-exp
    score_configs = [
        ("score_OptiDICE",     "linear", "linear"),
        ("score_FairDICE",     "log",    "linear"),
        ("score_linear-log",   "linear", "log"),
        ("score_linear-exp",   "linear", "exp"),
        ("score_G-log_F-log",  "log",    "log"),
        ("score_G-log_F-exp",  "log",    "exp"),
    ]
    # Short aliases used in the F slot of score_configs. Anything not aliased is
    # passed through to Utility() directly (so "linear" / "log" still work).
    F_TYPE_ALIAS = {"exp": "exponential_F"}
    # Multi-pref baselines test how the policy trades off objectives. Dispatch on
    # reward_reshape (determines reward_dim): default = 3 prefs for 2-obj d4morl;
    # walker2d_cd_safe uses 3-dim (speed, cost, safety) prefs; native M-obj
    # (reshape=none, reward_dim>2, e.g. mo_pointmaze) uses uniform-M.
    from src.data.reward_transform import WALKER2D_CD_SAFE_RESHAPES
    if reward_reshape in WALKER2D_CD_SAFE_RESHAPES:
        # safe_cd's training objective is G=plog × F=cobbdouglas+plogT — no prefs.
        # The linear-G eval baselines (OptiDICE / linear-log / linear-exp) below
        # still need *some* weight vector to form a scalar score; uniform [1,1,1]
        # is the simplest "no preference" default. Single pref → one suffix per
        # linear-G baseline instead of fanning out across pref combinations.
        LINEAR_BASELINE_PREFS = [
            np.ones(config.reward_dim, dtype=np.float32),   # (speed, cost, safety…) — pv 는 4 채널
        ]
    elif config.reward_dim == 2:
        # d4morl default (2-obj, reshape=none): 3 canonical prefs.
        LINEAR_BASELINE_PREFS = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    else:
        # Native M-obj env (reshape=none, reward_dim>2, e.g. mo_pointmaze): uniform
        # pref over M as the "no preference" baseline (mirrors antdir4/antdir7).
        m = config.reward_dim
        LINEAR_BASELINE_PREFS = [np.full(m, 1.0 / m, dtype=np.float32)]
    LINEAR_MULTI_PREF_NAMES = {"score_OptiDICE", "score_linear-log", "score_linear-exp"}

    scores = {}
    for name, g_type, f_type in score_configs:
        # exp_f_scale must match the training-side utility_F (setup.build_utility_F),
        # else eval scores the policy under a different F than it was trained on.
        f_util = Utility(type=F_TYPE_ALIAS.get(f_type, f_type),
                         exp_scale=getattr(config, "exp_f_scale", 50.0))
        # mean_Ftilde = mean_over_episodes[F(R) - F(0)] = mean(F(R)) - F(0)
        # (mean is linear, so the F(0) shift commutes out).
        u0_f = np.asarray(
            f_util(jnp.zeros((1, config.reward_dim), dtype=jnp.float32), keep_dims=True)
        ).squeeze(0)
        mean_Ftilde = np.asarray(
            np.mean(np.asarray(f_util(return_array_norm, keep_dims=True)), axis=0)
        ) - u0_f  # [reward_dim]

        if g_type == "linear" and name in LINEAR_MULTI_PREF_NAMES:
            # G-linear with explicit pref weights: weighted dot product per pref.
            for w in LINEAR_BASELINE_PREFS:
                assert len(w) == config.reward_dim, (
                    f"baseline pref length {len(w)} != reward_dim {config.reward_dim}"
                )
                w_str = "[" + ",".join(f"{x:g}" for x in w) + "]"
                scores[f"{name}_w{w_str}"] = float(np.sum(w * mean_Ftilde))
        else:
            # Single score (g_type == "log"): FairDICE / G-log_F-log / G-log_F-exp.
            # Utility("log")(mean_Ftilde, keep_dims=False) = sum_d log(mean_Ftilde_d + eps).
            g_util = Utility(type=g_type)
            scores[name] = float(g_util(mean_Ftilde, keep_dims=False))

    # 자기 목적 점수 (2026-10-25, docs/aet_objectives_worksheet.md §0): 이 런이 학습한 (G, F) 그대로
    # J = G(E_ep[F̃(R)]). 위 12 조합은 고정 (G, F) 라 새 목적(sigmoid F, 가중/등탄력 G)을 못 잰다.
    from src.utility import utility_G, g_kwargs_from_config
    _own_lens = np.asarray(ep_lengths, np.int32)
    _f_kw = {"steps": jnp.asarray(_own_lens)} if getattr(utility_F, "needs_steps", False) else {}
    _f0_kw = {"steps": jnp.zeros((1,), jnp.int32)} if _f_kw else {}
    _ft = (np.asarray(utility_F(jnp.asarray(return_array_norm), keep_dims=True, **_f_kw))
           - np.asarray(utility_F(jnp.zeros((1, config.reward_dim), jnp.float32), keep_dims=True, **_f0_kw)))
    _own_v = _ft.mean(axis=0)                                    # [F̃-dim] = E_ep[F̃(R)]
    scores["score_own"] = float(utility_G(jnp.asarray(_own_v), config.utility_G_type,
                                          **g_kwargs_from_config(config)))

    # Cobb-Douglas — fires for every reshape where dim 0 = reward, dim 1 = cost
    # (antcd / walker2dcd / half_cheetahcd / walker2d_cd_safe). Standalone non-convex
    # objective; not comparable to the 12 (G, F) scores above. Computed per-episode
    # and averaged so it mirrors E[CD(R)] (the training objective when
    # utility_F=cobbdouglas).
    #
    # rho-selection rule:
    #   - learner_type='bc' AND config.cd_filter_rho set → use cd_filter_rho
    #     (BC policy was filtered at this rho, so the matching evaluation rho is
    #     the filter rho — NOT cobbdouglas_rho, which BC doesn't set).
    #   - otherwise → cobbdouglas_rho (or 0.5 default for legacy configs).
    from src.data.reward_transform import CD_RESHAPES as _CD_RESHAPES  # SSOT
    cd_reshape_active = reward_reshape in _CD_RESHAPES
    if cd_reshape_active:
        from src.data.reward_transform import CobbDouglasUtility
        if (getattr(config, "learner_type", "aet") == "bc"
                and getattr(config, "cd_filter_rho", None) is not None):
            cd_rho = float(config.cd_filter_rho)
        else:
            cd_rho = float(getattr(config, "cobbdouglas_rho", 0.5))
        # CD 인자는 utility.cd_kwargs_from_config 에서만 온다 (SSOT). rho 만 넘기면
        # mode 가 기본 original 로 떨어져 modified 정책을 다른 목적으로 채점하게 된다.
        from src.utility import cd_kwargs_from_config
        from src.data.reward_transform import cd_rate_score, CD_REWARD_CH, CD_COST_CH
        # score_CobbDouglas = E[CD(R)] — 논문 지표. 학습 form 과 무관하게 **total 형태로** 잰다
        # (rate 형태 런도 같은 잣대로 비교되게).
        cd_kw = {**cd_kwargs_from_config(config, rho=cd_rho), "form": "total", "horizon": None}
        if cd_kw.get("safety", "plog") != "plog":
            # safe-CD linear/product 는 rate 형태에서만 정의된다 (D6.20). 이 total 지표는 safety 를 빼고
            # 순수 CD(speed, cost) 만 잰다 — safety 는 final_returns_normalized[:, 2] (S̄) 로 따로 본다.
            cd_kw.update(safety="plog", kappa=1.0, threshold=None, safety_range=None)
        cd_per_ep = np.asarray(
            CobbDouglasUtility(**cd_kw)(jnp.asarray(return_array_norm), keep_dims=True)
        )[..., 0]                                         # [num_episodes]
        scores["score_CobbDouglas"]     = float(cd_per_ep.mean())
        scores["score_CobbDouglas_std"] = float(cd_per_ep.std(ddof=0))
        scores["score_CobbDouglas_rho"] = cd_rho
        # exp A (rate 형태) 의 에피소드 점수 (L/T)·CD(R·T/L) — 길이 T 에피소드면 위와 같고,
        # 조기 종료 에피소드는 남은 스텝의 효율을 잃는다. amp 는 곱하지 않는다 (amp=1 기준 비교).
        lens = np.asarray(ep_lengths, np.float64)
        assert lens.shape == (return_array_norm.shape[0],), (lens.shape, return_array_norm.shape)
        T = float(config.max_seq_len)
        cd_rate_ep = np.asarray(cd_rate_score(
            return_array_norm[:, CD_REWARD_CH].astype(np.float64),
            return_array_norm[:, CD_COST_CH].astype(np.float64), lens, T, cd_rho, xp=np))
        scores["score_CobbDouglas_rate"]     = float(cd_rate_ep.mean())
        scores["score_CobbDouglas_rate_std"] = float(cd_rate_ep.std(ddof=0))
        scores["score_early_term_frac"]      = float(np.mean(lens < T))
        # ※ scores 는 전부 float 로 포맷돼 로깅된다 — 문자열을 끼우면
        #   "Unknown format code 'f' for object of type 'str'" 로 터진다 (2026-09-14 사고).

    if t_env is not None:
        # Per-objective raw + normalized stats. Normalized mirror lives in the same
        # space as the analysis notebooks (Reward ∈ [0, scaling·T], Cost ∈ [0, scaling·T])
        # so wandb plots are directly comparable to post-hoc plots without
        # reapplying (raw - min) / span.
        per_obj_metrics = {}
        for i in range(config.reward_dim):
            per_obj_metrics[f"eval/avg_returns_{i}"]      = float(avg_returns[i])
            per_obj_metrics[f"eval/std_returns_{i}"]      = float(std_returns[i])
            per_obj_metrics[f"eval/max_returns_{i}"]      = float(max_return[i])
            per_obj_metrics[f"eval/min_returns_{i}"]      = float(min_return[i])
            per_obj_metrics[f"eval/avg_returns_norm_{i}"] = float(avg_returns_norm[i])
            per_obj_metrics[f"eval/std_returns_norm_{i}"] = float(std_returns_norm[i])

        # Semantic aliases when the reshape produces a (reward, cost, ...) layout.
        # The reshape maps env obj[0] → reward=forward velocity, obj[1] → cost=‖a‖²
        # (`src/data/reward_transform.py:212-218`). Aliases log in NORMALIZED space —
        # same space the CD objective is evaluated in.
        if cd_reshape_active:
            per_obj_metrics["eval/return_mean"] = float(avg_returns_norm[0])
            per_obj_metrics["eval/return_std"]  = float(std_returns_norm[0])
            per_obj_metrics["eval/cost_mean"]   = float(avg_returns_norm[1])
            per_obj_metrics["eval/cost_std"]    = float(std_returns_norm[1])
            per_obj_metrics["eval/cd_mean"]     = scores["score_CobbDouglas"]
            per_obj_metrics["eval/cd_std"]      = scores["score_CobbDouglas_std"]
            per_obj_metrics["eval/cd_rho"]      = scores["score_CobbDouglas_rho"]

        if config.use_wandb:
            wandb.log(per_obj_metrics, step=t_env)
            wandb.log(
                {f"eval/{name}": val for name, val in scores.items()},
                step=t_env,
            )
        else:
            print(f"Eval at step {t_env}:")
            print(f"  Avg returns:        {avg_returns}")
            print(f"  Avg returns (norm): {avg_returns_norm}")
            for name, val in scores.items():
                print(f"  {name}: {val:.4f}")
            for i in range(config.reward_dim):
                print(
                    f"  Max/Min returns obj {i}: {max_return[i]:.2f} / {min_return[i]:.2f}"
                )

    return {
        **scores,
        "return_array": return_array,               # raw per-episode returns
        "return_array_normalized": return_array_norm,  # normalized per-episode returns (used for scores)
        "episode_lengths": np.asarray(ep_lengths, np.int32),  # [num_episodes] — 조기 종료 진단
        **({"racc_visits": np.array(racc_visits)} if return_racc_traj else {}),  # [total_steps, 1+reward_dim] (t, R^acc_t)
        **({"state_visits": np.array(state_visits)} if return_racc_traj else {}),  # [total_steps, obs_dim] raw
        **({"position_trajectories": pos_trajs} if capture_positions else {}),  # list of [T, 2] (x,y) disp
        **({"reshape_score_returns": np.array(score_returns)} if _score_step is not None else {}),  # [N, score_dim] raw
    }
