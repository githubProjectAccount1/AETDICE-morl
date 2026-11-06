"""I-projection 추출 손실 (--extraction iproj) — 모듈 설명은 src/nets/iproj_policy.py.

AET_learner.train_step 의 정책 손실 자리에서 불린다. 쌍대(ν, μ) 는 그대로이고 e 표적 e(s,a) 만 받는다.
세 손실을 더해 한 옵티마이저로 갱신한다. 파라미터 집합이 서로 겹치지 않고, actor 손실에서는 e_net·data_policy 를
상수로 묶으므로(_frozen) 더해도 각자 따로 갱신한 것과 같다.
  e_loss     = E[(e_net(s,a) − e)²]                         원문 e_loss_type='mse'
  bc_loss    = −E[log π_D(a|s)]                            원문 data_policy_loss
  actor_loss = −E_s Σ_k p_k [log w(s,a_k) − log π_ψ(a_k|s) + log π_D(a_k|s)],  a_k = μ_k + σ_k·ε
기대값 Σ_k p_k 는 MoG 의 성분 선택이 재매개화가 안 되므로 성분마다 한 표본씩 뽑아 혼합 가중으로 정확히 합친다
(원문은 단일 tanh-Gaussian 이라 이 문제가 없다). actor 는 warmup 동안 갱신하지 않는다 (원문 warmup 5e5 / 3e6).
원문의 자동 엔트로피 계수 항은 넣지 않았다 — KL 안의 −log π_ψ 가 이미 엔트로피 항이다.
"""
import jax
import jax.numpy as jnp
from flax import nnx

from src.divergence import log_f_derivative_inverse

WARMUP_FRAC = 1.0 / 6.0      # OptiDICE 기본값 warmup_iterations 5e5 / total_iterations 3e6


def _frozen(module):
    """같은 계산을 하되 파라미터로 기울기가 가지 않는 사본 (행동 입력으로는 흐른다)."""
    graphdef, state = nnx.split(module)
    return nnx.merge(graphdef, jax.lax.stop_gradient(state))


def _component_samples(dist, key):
    """MoG 성분마다 재매개화 표본 하나: [M, B, A] 와 혼합 확률 [B, M]."""
    comp = dist.components_distribution                                   # batch [B, M], event [A]
    loc, scale = comp.mean(), comp.stddev()                               # [B, M, A]
    a = loc + scale * jax.random.normal(key, loc.shape)
    probs = jax.nn.softmax(dist.mixture_distribution.logits, axis=-1)     # [B, M]
    return jnp.swapaxes(a, 0, 1), probs


def iproj_loss(policy, states, raccs, t_b, actions, e_b, beta, f_divergence, step, total_steps, key):
    B = states.shape[0]
    assert e_b.shape == (B,) and t_b.shape == (B,) and actions.shape[0] == B, (e_b.shape, t_b.shape)
    e_pred = policy.e_value(states, raccs, t_b, actions)
    e_loss = jnp.mean((e_pred - jax.lax.stop_gradient(e_b)) ** 2)
    bc_loss = -jnp.mean(policy.data_policy(states, raccs, t_b).log_prob(actions))

    a_k, probs = _component_samples(policy.actor(states, raccs, t_b), key)  # [M, B, A], [B, M]
    lp_actor = policy.actor(states, raccs, t_b).log_prob(a_k)               # [M, B]
    lp_data = _frozen(policy.data_policy)(states, raccs, t_b).log_prob(a_k)
    log_w = log_f_derivative_inverse(_frozen(policy).e_value(states, raccs, t_b, a_k) / beta,
                                     f_divergence)                         # [M, B]
    objective = jnp.sum(probs.T * (log_w - lp_actor + lp_data), axis=0)    # [B]
    active = (step >= jnp.asarray(WARMUP_FRAC * total_steps)).astype(jnp.float32)
    actor_loss = -jnp.mean(objective) * active
    total = e_loss + bc_loss + actor_loss
    aux = {"debug_iproj_e_loss": e_loss, "debug_iproj_bc_loss": bc_loss, "debug_iproj_actor_obj": jnp.mean(objective),
           "debug_iproj_log_w": jnp.mean(jnp.sum(probs.T * log_w, axis=0)),
           "debug_iproj_kl": jnp.mean(jnp.sum(probs.T * (lp_actor - lp_data), axis=0))}
    return total, aux
