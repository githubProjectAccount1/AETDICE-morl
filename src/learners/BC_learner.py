"""Behavior Cloning learner — pure MLE on (state, action) pairs. 보상은 쓰지 않는다.

Designed as a baseline for the AETDICE / FairDICE / IQL pipelines: train a
state-only Mixture-of-Gaussians policy on a CD-filtered subset of the dataset
(top-K% by trajectory-level CD score under a chosen rho — see
`buffer.load_d4morl_dataset(cd_filter=...)` and
`reward_transform.filter_top_k_cd`).

★ `--bc_conditioning` 축 (2026-09-16 추가). **perf 분모를 알고리즘과 같은 MDP 에 두기 위해서다.**

    "state"  π(a | s)          — 표준 BC. base MDP 의 Markov 정책. FairDICEContinuousPolicy.
    "racc"   π(a | s, racc)    — **augmented MDP** 의 history-dependent 정책.
                                 AETDICE 와 **같은 정책 클래스·같은 all-H 채점**을 쓰고
                                 가중치만 w ≡ 1 이다 (= AET WBC 에서 학습된 w 를 뺀 것).

왜 이 축이 필요한가: 이 목적함수는 return 에 대해 비선형(G∘F)이라 **convex MDP** 다.
convex MDP 에서는 Markov 정책이 최적이 아니고 history-dependence(또는 mixture)가 필요하다
— 그게 이 논문과 Zahavy et al.(2021) 의 전제다. 그런데 바닥선만 Markov 정책이면
perf 의 일부가 "알고리즘이 잘해서"가 아니라 **"바닥선이 racc 를 못 봐서"** 가 된다.
알고리즘(AETDICE)도 오라클(FW mixture / racc_aware PPO)도 전부 augmented 쪽에 있는데
BC 만 혼자 base MDP 에 있었다 (2026-09-16 발견 — 사용자 지적).
`bc_conditioning="racc"` 로 두면 세 축이 같은 정책 클래스가 되고, perf 는 **학습된 w 가
벌어주는 것만** 측정한다.

What this learner does NOT use (두 모드 공통):
  - critic / nu network, mu network / preference weights
  - augmented per-step F̃-rewards (batch.rewards is ignored) — 보상 신호가 전혀 안 들어간다

In DICE-method language this is "weighted BC with w(s,a) = 1" — 학습된 importance ratio가
아니라 상수다. (cd_filter 를 켜면 top-K% 궤적만 남기는 이진 w 가 되지만, 바닥선 런은
k_pct=1.0 = 필터 없음으로 돌린다.)

Conventions kept identical to the other learner modules so the runner doesn't
need any special-casing: TrainState carries policy_state + nu_state + mu_state
even though the latter two are dummies (allocated but never updated). This keeps
checkpoint shape / save format uniform across learner_type values.
"""
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from src.nets.policy import (FairDICEContinuousPolicy, MuNetworkTransformed,
                             build_continuous_policy)
from src.learners.AET_learner import extraction_mode
from src.nets.critic import FairDICECritic
from src.learners import _common
from src.learners._common import (
    NetworkState, TrainState, Model,
    get_model, make_cosine_adam_tx,
    save_model, save_checkpoint,
)


BC_CONDITIONINGS = ("state", "racc")


def bc_conditioning(config) -> str:
    """BC 정책이 무엇을 조건으로 받는가 — 위 docstring 의 단일 축.

    옛 config.json 에는 이 키가 없다 → "state" 로 되짚는다 (2026-09-16 이전 런 전부).
    그래야 저장된 체크포인트가 같은 구조로 복원된다.
    """
    mode = getattr(config, "bc_conditioning", None) or "state"
    assert mode in BC_CONDITIONINGS, f"{mode!r} not in {BC_CONDITIONINGS}"
    return mode


def init_train_state(config) -> TrainState:
    rngs = nnx.Rngs(config.seed)

    if bc_conditioning(config) == "racc":
        # ★ AETDICE 와 **같은 정책 클래스**. 같은 extraction 축을 쓴다 (SSOT — BC 전용
        #   구조를 새로 만들지 않는다). 차이는 학습 손실의 가중치가 w ≡ 1 이라는 것뿐이다.
        racc_dim = getattr(config, "racc_dim", config.reward_dim)
        policy = build_continuous_policy(
            extraction_mode(config),
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            reward_dim=config.reward_dim,
            racc_dim=racc_dim,
            racc_mean=config.racc_mean,
            racc_std=config.racc_std,
            hidden_dims=config.hidden_dims[:-1],
            horizon=config.max_seq_len,
            layer_norm=config.layer_norm,
            n_mixture=getattr(config, "n_mixture", 5),
            rngs=rngs,
        )
    else:
        # 표준 BC — base MDP 의 Markov 정책 (racc 를 보지 않는다).
        policy = FairDICEContinuousPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims[:-1],
            horizon=config.max_seq_len,
            layer_norm=config.layer_norm,
            rngs=rngs,
        )

    # Dummy nu / mu — never optimized but kept for TrainState shape parity with
    # the other learner modules so checkpoint format and save_dir layout stay
    # uniform across learner_types.
    nu = FairDICECritic(
        state_dim=config.state_dim,
        hidden_dims=config.hidden_dims,
        horizon=config.max_seq_len,
        layer_norm=config.layer_norm,
        rngs=rngs,
    )
    mu = MuNetworkTransformed(config)

    # Un-clipped Adam chains (grad_clip removed 2026-09-02).
    policy_tx = make_cosine_adam_tx(config.policy_lr, config.total_train_steps)
    # nu / mu are dummies (never updated) — plain Adam, kept as-is for shape parity.
    nu_tx = optax.scale_by_adam()
    mu_tx = optax.scale_by_adam()

    policy_optim = nnx.Optimizer(policy, policy_tx)
    (policy_gd, policy_state) = nnx.split((policy, policy_optim))

    nu_optim = nnx.Optimizer(nu, nu_tx)
    (nu_gd, nu_state) = nnx.split((nu, nu_optim))

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


def train_step(config, train_state: TrainState, batch, key: jax.random.PRNGKey):
    """One gradient step: MLE on (state, action) pairs. batch.rewards 는 항상 무시한다.

    conditioning="racc" 면 batch.raccs (Buffer 가 F̃ 변환해 저장한 표준 racc) 를 정책
    입력으로 준다 — AET_learner 의 extraction="original" 분기와 **같은 all-H 채점**이고
    가중치만 w ≡ 1 이다.
    """
    step = train_state.step
    policy, policy_optim, _ = get_model(train_state.policy_state)
    cond = bc_conditioning(config)

    states  = batch.states         # [B, state_dim]
    actions = batch.actions        # [B, action_dim]
    raccs   = batch.raccs          # [B, RaccDim]  (F̃-transformed in Buffer; F̃(0) = 0)

    def policy_loss_fn(policy):
        # 두 모드 모두 batch_shape=[B, H] 를 받아 action 을 H 개 head 전부에 복제해 채점한다
        # (AET extraction="original" 과 동일). MLE: minimize -mean(log_prob).
        dist = policy(states, raccs, timesteps=None) if cond == "racc" else policy(states)
        actions_broad = jnp.repeat(actions[:, None, :], config.max_seq_len, axis=1)
        log_probs = dist.log_prob(actions_broad)        # [B, H]
        assert log_probs.shape == (states.shape[0], config.max_seq_len), log_probs.shape
        return -jnp.mean(log_probs), log_probs

    (policy_loss, log_probs), policy_grads = nnx.value_and_grad(
        policy_loss_fn, has_aux=True
    )(policy)
    policy_optim.update(policy_grads)
    policy_state_ = nnx.state((policy, policy_optim))

    train_state = train_state._replace(
        policy_state=train_state.policy_state._replace(state=policy_state_),
        step=step + 1,
    )
    # Keys must contain "loss" / "grad" / "debug" to be logged as scalars by
    # main's wandb loop; anything else goes through a per-objective
    # branch that expects a [reward_dim]-shaped value and fails on scalars.
    return train_state, {
        "policy_loss":         policy_loss,
        "debug_log_prob_mean": jnp.mean(log_probs),
        "debug_log_prob_std":  jnp.std(log_probs),
    }


# ── Checkpoint helpers (mirror the other learners) ────────────────────────────
# save_model / save_checkpoint imported from _common. load_model / load_checkpoint
# are thin wrappers binding this module's init_train_state as the restore template.
def load_model(path: str, config) -> TrainState:
    return _common.load_model(path, config, init_train_state)


def load_checkpoint(save_dir: str, config):
    return _common.load_checkpoint(save_dir, config, init_train_state)
