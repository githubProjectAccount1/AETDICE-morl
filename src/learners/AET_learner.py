"""AET learner for the transformed AETDICE formulation (d4morl / Walker2d, continuous actions).

Outer utility G is restricted to {"linear", "piecewise_log"} and the conjugate
is computed via `conjugate_G`. Inner utility F is absorbed into the augmented reward
upstream (buffer / reward_transform) and does not appear here.
`fair_alpha` is not used in this learner.
"""
import os

import jax
import jax.numpy as jnp
from flax import nnx
from src.nets.policy import (MuNetworkTransformed, build_continuous_policy,
                             EXTRACTION_MODES)
from src.nets.critic import Critic
from src.divergence import f, FDivergence, f_derivative_inverse
from src.utility import conjugate_G, g_kwargs_from_config
from src.learners import _common
from src.learners.Diffusion_learner import diffusion_wbc_loss
from src.learners.IProj_learner import iproj_loss
from src.learners import wbc_state_weight
from src.learners._common import (
    NetworkState, TrainState, Model,
    _mu_is_learnable, get_model, make_cosine_adam_tx,
    save_model, save_checkpoint,
)


def extraction_mode(config) -> str:
    """정책 추출 방식 — 두 줄기 중 하나. 기본값은 **논문 Table 4 원본**(all-H multi-head).

    옛 config.json 은 이 키가 없다. 그때는 `policy_time_cond` 로 되짚는다 (2026-09-06~09-07
    사이 런들이 그 플래그를 썼다) — 그래야 저장된 체크포인트를 같은 구조로 되살릴 수 있다.
    """
    mode = getattr(config, "extraction", None)
    if mode is None:
        mode = ("timecond" if bool(getattr(config, "policy_time_cond", False))
                else "original")
    assert mode in EXTRACTION_MODES, f"{mode!r} not in {EXTRACTION_MODES}"
    return mode


# --wbc_weight timestrat 의 층 폭 (스텝). 배치 512 / (500/25) = 층당 ~25 표본.
_TSTRAT_BIN = 25


def time_stratified(w_b, t_b, horizon):
    """자기 시점 w 를 시점 층(폭 _TSTRAT_BIN) 안에서 배치 평균 1 로 맞춘다 (--wbc_weight timestrat).

    왜 (2026-10-26, docs/CD_redesign_worksheet.md §7): per-timestep 채점에서 w 질량이 에피소드
    초반에 몰린다 (Walker CD: t<25 가 데이터 6% 인데 질량 48–69%, t≥200 은 데이터 55% 에 6–8%).
    WBC 가 중후반 상태를 거의 배우지 않고 정책이 그 구간에서 넘어진다. 층화 자기정규화는 각
    시점 구간이 데이터 비율만큼의 질량을 받게 하고, 층 안의 상대 가중(어떤 행동을 고르는가)은
    그대로 둔다. 층은 결과와 무관한 구조(시점)로 정해진다. 전부 0 인 층은 0 으로 남는다.
    """
    B = w_b.shape[0]
    assert t_b.shape == (B,), (t_b.shape, w_b.shape)
    n_bins = -(-horizon // _TSTRAT_BIN)
    k = jnp.clip(t_b // _TSTRAT_BIN, 0, n_bins - 1)
    s = jax.ops.segment_sum(w_b, k, num_segments=n_bins)
    n = jax.ops.segment_sum(jnp.ones_like(w_b), k, num_segments=n_bins)
    mean_k = (s / jnp.maximum(n, 1.0))[k]
    return w_b / (mean_k + 1e-12)

# Config keys that define the DICE dual. A borrowed critic is only "the same critic" if all of
# these match the run it came from (racc_mean/racc_std are checked separately as arrays).
_DUAL_KEYS = ("utility_G_type", "utility_F_type", "beta", "divergence", "hidden_dims",
              "max_seq_len", "mu_scaling_factor", "reward_scaling_factor", "dataset_path",
              "layer_norm", "nu_output_transform", "standardize_e", "w_max", "racc_dim",
              "env_name", "reward_reshape")


def dual_is_frozen(config) -> bool:
    return bool(getattr(config, "critic_from", None))


def load_frozen_dual(train_state: TrainState, config) -> TrainState:
    """Replace ν/μ with a finished run's (--critic_from) so only the policy trains.

    Why (2026-10-22): the ν/μ loss never reads the policy, so every extraction method relearns
    the identical dual. Borrowing the baseline's critic makes extraction comparisons paired —
    same w, only the extractor differs — and skips the dual backward pass.
    """
    import numpy as np
    from src.eval.rollout import load_config      # lazy: pulls in the env wrappers

    run_dir = config.critic_from
    base = load_config(run_dir)
    for k in _DUAL_KEYS:
        mine, theirs = getattr(config, k, None), getattr(base, k, None)
        same = (np.allclose(np.asarray(mine, float), np.asarray(theirs, float))
                if isinstance(mine, (int, float, list, tuple, np.ndarray))
                and not isinstance(mine, bool) else mine == theirs)
        assert same, f"--critic_from mismatch on {k}: this run {mine!r} vs {run_dir} {theirs!r}"
    for k in ("racc_mean", "racc_std"):
        assert np.allclose(np.asarray(getattr(config, k)), np.asarray(getattr(base, k))), k
    path = os.path.abspath(os.path.join(run_dir, "model"))
    if jax.default_backend() == "cpu":
        # GPU 에서 저장된 체크포인트는 저장 시 장치(cuda)를 기록해 CPU 부팅에서 복원이 거부된다 —
        # 평가 경로와 같은 CPU 복원기를 쓴다 (CPU smoke, 2026-10-26).
        from src.eval.rollout import load_model_cpu
        base_state = load_model_cpu(path, base, init_train_state)
    else:
        base_state = _common.load_model(path, base, init_train_state)
    print(f"[critic_from] frozen ν/μ from {run_dir} (base seed {base.seed}, step "
          f"{int(base_state.step)}); μ = {np.asarray(get_model(base_state.mu_state)[0]())}")
    return train_state._replace(nu_state=base_state.nu_state, mu_state=base_state.mu_state)


def _diffusion_policy_kwargs(config) -> dict:
    """Diffusion-only constructor args; empty for the MoG branches so their constructors are
    called exactly as before. Defaults mirror cli.py (config.json files without these keys)."""
    if extraction_mode(config) != "diffusion":
        return {}
    return {
        "diffusion_steps": int(getattr(config, "diffusion_steps", 20)),
        "uncond_prob": float(getattr(config, "diffusion_uncond_prob", 0.0)),
        "cfg_scale": float(getattr(config, "diffusion_cfg_scale", 1.0)),
    }


def init_train_state(config) -> TrainState:
    rngs = nnx.Rngs(config.seed)

    # racc_dim falls back to reward_dim if not set (pointwise F: linear/exp/plog/...).
    # For CobbDouglas with reward_dim != 2 (e.g. walker2d_cd_safe), F̃-dim=2 < reward_dim,
    # so the network input must be sized from racc_dim, not reward_dim.
    racc_dim = getattr(config, "racc_dim", config.reward_dim)
    policy = build_continuous_policy(
        extraction_mode(config),
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        reward_dim=config.reward_dim,
        racc_dim=racc_dim,
        racc_mean = config.racc_mean,
        racc_std = config.racc_std,
        hidden_dims=config.hidden_dims[:-1],
        horizon=config.max_seq_len,
        layer_norm=config.layer_norm,
        n_mixture=getattr(config, "n_mixture", 5),
        global_mixture=bool(getattr(config, "global_mixture", False)),
        rngs=rngs,
        **_diffusion_policy_kwargs(config),
    )

    nu = Critic(
        state_dim=config.state_dim,
        reward_dim=config.reward_dim,
        racc_dim=racc_dim,
        racc_mean = config.racc_mean,
        racc_std = config.racc_std,
        hidden_dims=config.hidden_dims,
        horizon=config.max_seq_len,
        layer_norm=config.layer_norm,
        nu_output_transform=getattr(config, "nu_output_transform", "none"),
        rngs=rngs,
    )

    # Un-clipped Adam chains (matches the paper's Table 4/5/6 runs). grad_clip removed
    # 2026-09-02 — the always-on clip lowered AET-DICE returns and the opt-in flag was
    # too error-prone to keep.
    policy_tx = make_cosine_adam_tx(config.policy_lr, config.total_train_steps)
    nu_tx = make_cosine_adam_tx(config.critic_lr, config.total_train_steps)
    mu_tx = make_cosine_adam_tx(config.mu_lr, config.total_train_steps)

    policy_optim = nnx.Optimizer(policy, policy_tx)
    (policy_gd, policy_state) = nnx.split((policy, policy_optim))

    nu_optim = nnx.Optimizer(nu, nu_tx)
    (nu_gd, nu_state) = nnx.split((nu, nu_optim))

    mu = MuNetworkTransformed(config)
    mu_optim = nnx.Optimizer(mu, mu_tx)
    (mu_gd, mu_state) = nnx.split((mu, mu_optim))

    nu_target = nu_state.filter(nnx.Param)
    policy_target = policy_state.filter(nnx.Param)
    mu_target = mu_state.filter(nnx.Param)

    return TrainState(
        policy_state=NetworkState(policy_gd, policy_state, policy_target),
        nu_state=NetworkState(nu_gd, nu_state, nu_target),
        mu_state=NetworkState(mu_gd, mu_state, mu_target),
        step=jnp.array(0),
    )


# --nu_grad (2026-10-27, docs/CD_redesign_worksheet.md §10 B): ν 기울기 모드 (Mao et al. ICLR 2024 = ODICE).
#   true  : 지금까지 그대로 — ν(s) 와 ν(s′) 양쪽으로 흐르는 true-gradient.
#   semi  : backward 항(ν(s′) 쪽)을 뺀다 = 행동 수준 제약만 (원문 식 4).
#   orth  : backward 항을 forward 항의 법평면에 투영해 η 배로 더한다 (원문 Algorithm 1: g_lin + g→ + η g⊥).
# 투영은 원문처럼 배치 손실의 **전체 파라미터 벡터** 단위로 한다.
# --nu_target_tau > 0 : 원문의 양방향 target network — forward 항은 ν_target(s′), backward 항은 ν_target(s) 로 계산하고
#   target 을 τ 로 EMA 갱신한다 (원문 τ = 5e-3). 0 이면 target 대신 stop_gradient(온라인 ν) (2026-10-27 첫 구현).
# --nu_linear_term data : 원문처럼 선형 항을 초기 상태가 아니라 **데이터의 모든 상태**(자기 시점 head)에 건다
#   ("every state as initial state", 원문 §2.1 — d0 를 d_D 로 넓힌다). init (기본) 은 AETDICE 원래 목적.
#   원문의 λ 재매개화 (1−λ)·선형 + λ·비선형 은 λ=0.5 가 우리 1:1 가중과 같고 Adam 이 기울기 스케일에 무관해서 넣지 않는다.
# 2026-10-27 Walker 실측: target 없이 init 선형 항으로 semi/orth 를 쓰면 ν 가 10 배 부풀고 w ≡ 0 으로 퇴화했다.
NU_GRAD_MODES = ("true", "semi", "orth")
NU_LINEAR_TERMS = ("init", "data")


def _tree_dot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def _reshape_nu_grads(config, g_full, nu_loss_fn, nu_network, mu_network):
    mode = str(getattr(config, "nu_grad", "true"))
    assert mode in NU_GRAD_MODES, mode
    if mode == "true":
        return g_full
    part_grad = lambda part: nnx.grad(lambda n: nu_loss_fn(n, mu_network, part=part)[0])(nu_network)
    g_lin, g_f = part_grad("lin"), part_grad("fwd")
    if mode == "semi":
        return jax.tree_util.tree_map(lambda l, f: l + f, g_lin, g_f)
    g_b = part_grad("bwd")
    coef = _tree_dot(g_b, g_f) / (_tree_dot(g_f, g_f) + 1e-12)
    eta = float(getattr(config, "orth_eta", 1.0))
    return jax.tree_util.tree_map(lambda l, f, b: l + f + eta * (b - coef * f), g_lin, g_f, g_b)


def _ema_target(config, target_params, nu_state_):
    """ν target 의 EMA 갱신 (τ>0 일 때만 — 0 이면 지금까지처럼 초기값 그대로 둔다)."""
    tau = float(getattr(config, "nu_target_tau", 0.0))
    if tau <= 0.0:
        return target_params
    online = nu_state_.filter(nnx.Param)
    return jax.tree_util.tree_map(lambda t, o: (1.0 - tau) * t + tau * o, target_params, online)

def train_step(config, train_state: TrainState, batch, key: jax.random.PRNGKey):
    key, subkey = jax.random.split(key)
    step = train_state.step
    # β is now schedule-driven. For --beta_schedule=constant the closure always
    # returns config.beta unchanged (back-compat). For non-constant schedules,
    # β anneals from beta_init to beta_final over (total_train_steps - warmup).
    beta = config.beta_schedule_fn(step)
    f_divergence = FDivergence[config.divergence]
    # Standardize Bellman residual by its batch-wise std before f' inverse so β
    # is in units of σ_e instead of absolute reward scale. stop_gradient on σ
    # so the network can't game scale up/down. EPS guards against zero-std edge.
    standardize_e = bool(getattr(config, "standardize_e", False))
    # w_max only matters for f_divergence=CLIP_CHI; default ∞ for other divergences.
    w_max = float(getattr(config, "w_max", float("inf")))
    # 정책 추출 두 줄기 중 하나 (구조 + WBC 채점이 한 축으로 묶여 있다).
    extraction = extraction_mode(config)

    rewards = batch.rewards      # [B, RewardDim]
    states = batch.states        # [B, StateDim]
    next_states = batch.next_states
    init_states = batch.init_states
    raccs = batch.raccs          # [B, RewardDim]  (F̃-transformed in Buffer; F̃(0) = 0)
    next_raccs = batch.next_raccs                 # (F̃-transformed in Buffer)
    # init_raccs = F̃(0) = 0 by construction (F̃(x) = F(x) - F(0)). Lives in the same
    # F̃-space as `raccs` / `next_raccs` so the Bellman residual is consistent.
    init_raccs = jnp.zeros_like(raccs)

    policy, policy_optim, _ = get_model(train_state.policy_state)
    nu_network, nu_optim, _ = get_model(train_state.nu_state)
    mu_network, mu_optim, _ = get_model(train_state.mu_state)

    use_target = float(getattr(config, "nu_target_tau", 0.0)) > 0.0
    nu_target = get_model(train_state.nu_state)[2] if use_target else None
    linear_term = str(getattr(config, "nu_linear_term", "init"))
    assert linear_term in NU_LINEAR_TERMS, linear_term

    def nu_loss_fn(nu_network, mu_network, part="full"):
        """part: full = 쌍대 손실 전체. lin = 선형 항만. fwd / bwd = 비선형 항 β·f*(e/β) 만, 각각 ν(s′) / ν(s) 를
        고정(또는 target) — ODICE 의 forward / backward 기울기 (--nu_grad semi|orth 가 쓴다)."""
        nu_values = nu_network(states, raccs, timesteps=None)
        next_nu_values = nu_network(next_states, next_raccs, timesteps=None)
        nu_values_lin = nu_values
        if part == "fwd":
            next_nu_values = jax.lax.stop_gradient(
                nu_target(next_states, next_raccs, timesteps=None) if use_target else next_nu_values)
        elif part == "bwd":
            nu_values_lin = nu_values
            nu_values = jax.lax.stop_gradient(
                nu_target(states, raccs, timesteps=None) if use_target else nu_values)

        nu_t = nu_values[:, :-1]      # [B, 0:H]
        nu_tp1 = next_nu_values[:, 1:]  # [B, 1:H+1]

        init_nu_values = nu_network(init_states, init_raccs, timesteps=None)
        init_nu = init_nu_values[:, 0]  # Value at t=0

        mu = mu_network()

        r_scalar = (rewards @ mu).reshape(-1, 1)
        e_matrix = r_scalar + nu_tp1 - nu_t
        e_flat_raw = e_matrix.flatten()
        # When standardize_e is on, normalize e by batch σ (stop-grad). Keeps β
        # unit-stable across reward-scale shifts (F-exp / telescoped reward → wide |e|).
        if standardize_e:
            sigma_e = jax.lax.stop_gradient(jnp.std(e_flat_raw) + 1e-8)
            e_flat = e_flat_raw / sigma_e
        else:
            sigma_e = jnp.float32(1.0)
            e_flat = e_flat_raw

        w_flat = jax.nn.relu(f_derivative_inverse(e_flat / beta, f_divergence, w_max=w_max))

        # Split the inner Lagrangian term so wandb can show reg vs Bellman pull
        # separately. Mathematically: loss_2 = mean(w·e) − β·mean(f(w)).
        f_w_flat     = f(w_flat, f_divergence)
        bellman_term = jnp.mean(w_flat * e_flat)
        reg_term     = beta * jnp.mean(f_w_flat)

        if linear_term == "data":    # 원문 ODICE: 데이터의 모든 상태를 초기 상태로 (자기 시점 head)
            t_lin = batch.timesteps.astype(jnp.int32)
            v_lin = nu_values_lin if part == "bwd" else nu_values
            loss_1 = jnp.mean(v_lin[jnp.arange(v_lin.shape[0]), t_lin])
        else:
            loss_1 = jnp.mean(init_nu)
        if part == "lin":
            return loss_1, None
        loss_2 = bellman_term - reg_term # derivative of beta * f*(e_flat / beta) w.r.t. e_flat is f_derivative_inverse(e_flat / beta, f_divergence, w_max=w_max) = w_flat
        loss_3 = conjugate_G(mu, config.utility_G_type, config.mu_scaling_factor,
                             **g_kwargs_from_config(config))

        total_loss = loss_1 + loss_2 + loss_3 if part == "full" else loss_2

        # Distributional summaries of the critic's outputs across this batch —
        # surfaced via the aux tuple → update_info → wandb. Tracks how the value
        # estimates scale and spread over training (`debug_*` prefix triggers
        # the auto-logger in main.py). Excludes the last timestep
        # column because critic.py:74-76 hard-masks it to zero (terminal-value
        # convention, not a network prediction); including it would bias the
        # mean toward zero and inflate the std with the artificial 0-vs-real
        # spread. Matches the slice the loss uses for nu_t.
        nu_predict = nu_values[:, :-1]
        nu_mean    = jnp.mean(nu_predict)
        nu_std     = jnp.std(nu_predict)

        # Diagnostics on the closed-form ratio w and the regularizer. Computed inside
        # nu_loss_fn so they share the same JIT trace as the loss. For w_max=inf the
        # saturated fraction degenerates to 0 (w can't reach inf via finite e/β).
        w_max_obs        = jnp.max(w_flat)
        w_p99            = jnp.quantile(w_flat, 0.99)
        w_frac_gt1       = jnp.mean((w_flat > 1.0).astype(jnp.float32))
        w_frac_sat       = jnp.mean((w_flat >= w_max * 0.999).astype(jnp.float32))
        ess              = (jnp.sum(w_flat) ** 2) / (w_flat.size * (jnp.sum(w_flat ** 2) + 1e-12))
        reg_over_bellman = reg_term / (jnp.abs(bellman_term) + 1e-8)
        dbg = {
            "debug_bellman_term":     bellman_term,
            "debug_reg_term":         reg_term,
            "debug_reg_over_bellman": reg_over_bellman,
            "debug_f_w_mean":         jnp.mean(f_w_flat),
            "debug_w_max":            w_max_obs,
            "debug_w_p99":            w_p99,
            "debug_w_frac_gt1":       w_frac_gt1,
            "debug_w_frac_sat":       w_frac_sat,
            "debug_ess":              ess,
        }

        return total_loss, (loss_1, loss_2, loss_3, w_flat, e_flat_raw, sigma_e, mu, nu_mean, nu_std, dbg)

    has_aux = True
    argnums = (0, 1) if _mu_is_learnable(config) else 0

    if dual_is_frozen(config):
        # --critic_from: ν/μ stay exactly as loaded; the loss is still evaluated for the logs.
        nu_loss, (loss_1, loss_2, loss_3, w_flat, e_flat_raw, sigma_e, mu, nu_mean, nu_std,
                  dbg) = nu_loss_fn(nu_network, mu_network)
    else:
        grad_fn = nnx.value_and_grad(nu_loss_fn, argnums=argnums, has_aux=has_aux)
        (nu_loss, (loss_1, loss_2, loss_3, w_flat, e_flat_raw, sigma_e, mu, nu_mean, nu_std, dbg)), grads = grad_fn(nu_network, mu_network)

        if _mu_is_learnable(config):
            nu_grads, mu_grads = grads
            mu_optim.update(mu_grads)
        else:
            nu_grads = grads

        nu_grads = _reshape_nu_grads(config, nu_grads, nu_loss_fn, nu_network, mu_network)
        nu_optim.update(nu_grads)

    nu_state_ = nnx.state((nu_network, nu_optim))
    mu_state_ = nnx.state((mu_network, mu_optim))

    def policy_loss_fn(policy):
        # w (importance ratio) over all H timesteps. nu/mu here are the current
        # train_state's networks — constants w.r.t. the policy gradient — so this is
        # numerically identical to the previous all-H w; only how the policy is scored
        # against it changes in the branch below.
        mu = mu_network()
        r_scalar = (rewards @ mu).reshape(-1, 1)
        nu_values = nu_network(states, raccs, timesteps=None)
        next_nu_values = nu_network(next_states, next_raccs, timesteps=None)
        e_matrix = r_scalar + next_nu_values[:, 1:] - nu_values[:, :-1]       # [B, H]
        # standardize_e: normalize e by batch σ (stop-grad), matching nu_loss_fn.
        if standardize_e:
            e_matrix = e_matrix / jax.lax.stop_gradient(jnp.std(e_matrix) + 1e-8)
        w_matrix = jax.nn.relu(f_derivative_inverse(e_matrix / beta, f_divergence, w_max=w_max))  # [B,H]
        w_detached = jax.lax.stop_gradient(w_matrix)
        # ── WBC 가중치 정형 (--wbc_weight) ────────────────────────────────────
        # 왜 사전 정당화되는가 (AGENT.md A1): 중요도 가중은 무한 분산을 가질 수 있어
        # 절단·정규화가 표준 분산 감소 기법이다. 우리 w 는 ESS 0.027%, 꼬리가 밀도비
        # 단위로 3.7e5 까지 가고, per-timestep 채점에서는 배치당 512 개만 평균해
        # CV 1.49 (all-H 는 0.18), 최악 배치에서 **한 샘플이 가중의 89%** 를 차지한다.
        #   raw    : 그대로 (논문)
        #   norm   : 배치 **전체** 평균으로 나눔 — 배치별 유효 학습률 변동 제거
        #   clip99 : p99 에서 절단 후 정규화 — 꼬리 제거
        #   rank   : 양수인 것들의 **순위**로 치환 후 정규화 — 크기를 버리고 순서만
        #            (실측: 선택된 것들 사이에서 w 크기는 corner 정보가 없다, AUC≈0.44)
        #   binary : **support 지시자** 1[w>0] 로 대체 — 크기를 전부 버리고 선택만 남긴다.
        #            왜 사전 정당화되는가 (2026-09-21 측정): 조건부 지점을 이웃으로 맞춰
        #            w-가중 행동분포와 무가중 행동분포를 비교하면 평균·2 차 모멘트·MMD 세
        #            층위 전부 치환 귀무와 구분되지 않는다 (ρ ≤ 1.33 / 1.06 / 1.48, 형태
        #            전용 합성 가중은 2.44 / 4.16 / 7.18). 즉 **크기에는 행동 정보가 없고
        #            정보는 support 에만 있다**. 잡음 가중을 그 support 지시자로 바꾸는 것은
        #            같은 타겟의 분산을 줄이는 표준 극한이고(rank 는 양수 사이 기울기를
        #            남겼다, binary 는 전부 버린다) ESS 가 0.027% → 4.43% 로 오른다.
        #            자유 파라미터가 없다.
        #   perstep: **시점축으로** 각 열 t 를 평균 1 로 — 헤드별 유효 학습률을 균등화.
        #            왜 사전 정당화되는가: all-H 채점에서 w 는 [B,H] 이고 열 t 가 곧 출력
        #            슬롯 head t 의 학습 신호다. w 는 궤적 후반에 몰리므로(실측 Σw 의
        #            98.4% 가 뒤 250 스텝, 앞 절반은 1.6%) head t 가 받는 질량이 t 에 따라
        #            수십 배 차이난다 → 앞쪽 헤드가 구조적으로 미학습된다. 열별 정규화는
        #            중요도 가중의 표준 **층화(stratified) 정규화**이고, 층은 결과와 무관한
        #            구조(출력 슬롯 인덱스)로 정해지므로 post-hoc 자유도가 없다.
        #            주의: 시점 간 상대 크기를 **의도적으로 버린다** — DICE 최적해의
        #            시점 가중을 이탈하는 대가를 치르고 헤드 학습량을 사는 거래다.
        _ww = str(getattr(config, "wbc_weight", "raw"))
        # timestrat / traj / action 은 자기 시점 w 를 뽑은 뒤 적용한다 (_own_weight).
        if _ww not in ("raw", "timestrat") + wbc_state_weight.WEIGHT_MODES:
            _flat = w_detached.reshape(-1)
            if _ww == "clip99":
                _hi = jnp.quantile(_flat, 0.99)
                w_detached = jnp.minimum(w_detached, _hi)
            elif _ww == "binary":
                w_detached = (w_detached > 0).astype(w_detached.dtype)
            elif _ww == "rank":
                _pos = (_flat > 0).astype(_flat.dtype)
                _r = jnp.argsort(jnp.argsort(_flat)).astype(_flat.dtype) / _flat.size
                w_detached = (_r * _pos).reshape(w_detached.shape)
            if _ww == "perstep":
                # 열(시점) 별 평균 1. 전 열이 0 인 시점은 0 으로 남는다 (신호가 없으므로).
                w_detached = w_detached / (jnp.mean(w_detached, axis=0, keepdims=True) + 1e-12)
            else:
                w_detached = w_detached / (jnp.mean(w_detached) + 1e-12)
        Bn = states.shape[0]
        assert w_detached.shape == (Bn, config.max_seq_len), w_detached.shape

        def _own_weight(t_b):
            """per-timestep 채점의 자기 시점 가중 w[b, t_b] 와 그 정형 (--wbc_weight)."""
            w_b = w_detached[jnp.arange(Bn), t_b]                           # [B]
            if _ww == "timestrat":
                return time_stratified(w_b, t_b, config.max_seq_len)
            if _ww == "traj":
                return jnp.asarray(config.wbc_traj_table)[batch.traj_ids.astype(jnp.int32)]
            if _ww == "action":
                return wbc_state_weight.action_ratio(config, w_b, states, raccs, t_b)
            return w_b

        # ── 궤적 단위 성분 배정 (--traj_mixture, EM 식 하드 배정) ─────────────
        # 왜 (2026-09-21, gmix 반증이 가리킨 지점): 성분의 **정체성이 상태마다 제각각**이다.
        # mean_head 가 상태 의존이라 "성분 3" 이 상태 A 와 B 에서 같은 정책일 이유가 없고,
        # MoG 는 likelihood 로만 학습되니 성분 라벨이 상태별로 자유롭게 뒤바뀐다. 그래서
        # 롤아웃에서 인덱스를 고정하면 무관한 모드들을 이어붙인 꼴이 된다
        # (gmix+commit: 12.7551 → 11.7975, 에피소드 σ 51.8 → 36.5).
        # AET 최적해는 **정책들의 혼합**이므로(Zahavy et al. 2021) 성분이 하나의 정책이어야
        # 한다 → 같은 궤적의 모든 transition 을 **같은 성분**으로만 채점해 정체성을 궤적에
        # 고정한다. 배정은 traj_id 의 결정적 해시 — 라벨이나 수익을 쓰지 않는다(post-hoc 아님).
        _traj_mix = bool(getattr(config, "traj_mixture", False))
        if _traj_mix:
            M = int(getattr(config, "n_mixture", 5))
            # traj_id 는 데이터셋 순서(임의)라 mod M 으로 충분하다. 해시 상수는 int32 를
            # 넘쳐 OverflowError 가 났고, 애초에 필요 없다.
            k_b = jnp.mod(batch.traj_ids.astype(jnp.int32), M)                 # [B]
            comp = policy(states, raccs, timesteps=None).components_distribution
            a_bc = jnp.repeat(batch.actions[:, None, :], config.max_seq_len, axis=1)  # [B,H,A]
            lp_all = comp.log_prob(a_bc[:, :, None, :])                        # [B,H,M]
            log_probs = jnp.take_along_axis(
                lp_all, k_b[:, None, None].astype(jnp.int32), axis=2).squeeze(2)      # [B,H]
            wbc_loss = -jnp.mean(w_detached * log_probs)
        elif extraction == "original":
            # 논문 Table 4 원본. 한 transition 의 action 을 H 개 head **전부**에 복제해 채점한다.
            # (racc, t) 가 action 과 decorrelate 되는 대가로, 이게 헤드라인 AET 13.69 를 낸 경로다.
            dist = policy(states, raccs, timesteps=None)                    # batch [B, H]
            actions_broad = jnp.repeat(batch.actions[:, None, :], config.max_seq_len, axis=1)
            log_probs = dist.log_prob(actions_broad)                        # [B, H]
            wbc_loss = -jnp.mean(w_detached * log_probs)
        elif extraction == "iproj":
            # I-projection (OptiDICE §3.3): 자기 시점 e 를 e_net 이 회귀하고, actor 는 데이터 밖 행동의 log w 로
            # π* ∝ w·π_D 에 KL 투영된다. w 를 데이터 행동에 곱하지 않으므로 --wbc_weight 는 쓰지 않는다 (cli 가 막음).
            t_b = batch.timesteps.astype(jnp.int32)                         # [B]
            e_b = jax.lax.stop_gradient(e_matrix[jnp.arange(Bn), t_b])      # [B]
            wbc_loss, log_probs = iproj_loss(                               # log_probs := debug dict
                policy, states, raccs, t_b, batch.actions, e_b, beta, f_divergence,
                step, config.total_train_steps, subkey)
        elif extraction == "diffusion":
            # timecond 와 같은 per-timestep 채점에서 −log π 만 w-가중 denoising 손실로 바꾼다.
            # subkey 는 다른 분기가 쓰지 않으므로 그들의 RNG 스트림은 그대로다.
            t_b = batch.timesteps.astype(jnp.int32)                         # [B]
            w_b = _own_weight(t_b)                                          # [B]
            wbc_loss, log_probs = diffusion_wbc_loss(                       # log_probs := mse
                policy, states, raccs, t_b, batch.actions, w_b, subkey)
        else:
            # per-timestep 정렬: 각 transition 을 **자기 t_b 에서만** 채점(weight w[b, t_b]).
            # all-H 복제는 하나의 고정 racc 로 모든 t 의 state→action 을 학습시켜 (racc, t) 를
            # action 과 decorrelate 시키고, 정책이 racc 를 무시한 π(a|s) 로 붕괴한다(측정 민감도 ~3%).
            # 비용도 B·H 가 아니라 B 행이라 time-조건부 backbone 에 싸다.
            t_b = batch.timesteps.astype(jnp.int32)                         # [B]
            w_b = _own_weight(t_b)                                          # [B]
            dist = policy(states, raccs, timesteps=t_b)                     # batch [B]
            log_probs = dist.log_prob(batch.actions)                        # [B]
            wbc_loss = -jnp.mean(w_b * log_probs)

        return wbc_loss, log_probs

    (policy_loss, log_probs), policy_grads = nnx.value_and_grad(
        policy_loss_fn, has_aux=True
    )(policy)
    policy_optim.update(policy_grads)
    policy_state_ = nnx.state((policy, policy_optim))
    iproj_dbg = log_probs if extraction == "iproj" else {}

    train_state = train_state._replace(
        policy_state=train_state.policy_state._replace(state=policy_state_),
        nu_state=train_state.nu_state._replace(
            state=nu_state_, target_params=_ema_target(config, train_state.nu_state.target_params, nu_state_)),
        mu_state=train_state.mu_state._replace(state=mu_state_),
        step=step + 1,
    )

    return train_state, {
        "policy_loss": policy_loss,
        "nu_loss": nu_loss,
        "mu": mu,
        "debug_w_mean": jnp.mean(w_flat),
        "debug_e_mean": jnp.mean(e_flat_raw),
        "debug_sigma_e": sigma_e,    # 1.0 when --standardize_e off; running batch σ otherwise
        "debug_nu_mean": nu_mean,
        "debug_nu_std":  nu_std,
        "debug_beta_current": beta,  # schedule-driven; constant for --beta_schedule=constant
        "loss_1": loss_1,
        "loss_2": loss_2,
        "loss_3": loss_3,
        **dbg,
        **iproj_dbg,
    }


# save_model / save_checkpoint imported from _common. load_model / load_checkpoint
# are thin wrappers binding this module's init_train_state as the restore template.
def load_model(path: str, config) -> TrainState:
    return _common.load_model(path, config, init_train_state)


def load_checkpoint(save_dir: str, config):
    return _common.load_checkpoint(save_dir, config, init_train_state)
