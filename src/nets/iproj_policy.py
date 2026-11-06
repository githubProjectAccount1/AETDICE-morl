"""I-projection 정책 추출용 모듈 묶음 (--extraction iproj, 2026-10-28).

OptiDICE (Lee et al. ICML 2021, §3.3; 공식 코드 secury/optidice `neural_dice_rl.py`) 의 추출:

    min_ψ  E_{s~d_D} KL( π_ψ(·|s) ‖ π*(·|s) ),   π*(a|s) ∝ w(s,a) · π_D(a|s)
         = −E_{s~d_D, a~π_ψ} [ log w(s,a) − log π_ψ(a|s) + log π_D(a|s) ]

weighted BC 는 데이터 행동에 w 를 곱해 모방하므로 w 가 0 에 가까운 상태에서는 정책이 아무것도 배우지 않는다
(OptiDICE §3.3 "does not work well in practice"; 우리 실측: w 질량이 소수 상태에 몰린다). I-projection 은 상태를
d_D(s) 그대로 두고 상태마다 π* 쪽으로 행동 분포를 옮긴다. 데이터 밖 행동 a 의 w 가 필요하므로 원문처럼
  e_net(s, racc, t, a) ≈ e(s, a)            — 데이터 Bellman 잔차의 회귀 (원문 e_loss_type='mse')
  data_policy π_D(a | s, racc, t)           — 데이터 행동의 BC (원문 data_policy='tanh_mdn', 성분 5)
를 함께 둔다. 세 부분이 한 모듈이라 정책 옵티마이저·체크포인트 하나로 다뤄진다 (TrainState 를 넓히지 않는다).
평가(evaluate.py)는 `policy(states, raccs, timesteps)` 만 부르므로 actor 의 MoG 분포를 그대로 돌려준다.
"""
from flax import nnx
import jax.numpy as jnp

from src.nets.policy import MLP, RaccEncoder, TimeConditionedPolicy, _timestep_raw


class IProjPolicy(nnx.Module):
    def __init__(self, state_dim, action_dim, reward_dim, racc_mean, racc_std, hidden_dims, horizon,
                 layer_norm: bool = False, n_mixture: int = 5, racc_dim=None,
                 global_mixture: bool = False, rngs: nnx.Rngs = nnx.Rngs(0)):
        kw = dict(state_dim=state_dim, action_dim=action_dim, reward_dim=reward_dim,
                  racc_mean=racc_mean, racc_std=racc_std, hidden_dims=hidden_dims, horizon=horizon,
                  layer_norm=layer_norm, n_mixture=n_mixture, racc_dim=racc_dim, rngs=rngs)
        self.actor = TimeConditionedPolicy(global_mixture=global_mixture, **kw)
        self.data_policy = TimeConditionedPolicy(**kw)
        self.horizon = horizon
        self.racc_encoder = RaccEncoder(racc_mean, racc_std)
        rd = reward_dim if racc_dim is None else racc_dim
        self.e_net = MLP(din=state_dim + rd + 1 + action_dim, hidden_dims=hidden_dims, dout=1,
                         rngs=rngs, layer_norm=layer_norm)

    def e_value(self, states, raccs, timesteps, actions):
        """e(s, racc, t, a) — actions 는 [..., B, A] (앞 축은 표본 축으로 브로드캐스트)."""
        t = jnp.clip(timesteps.astype(jnp.int32), 0, self.horizon - 1)
        cond = jnp.concatenate([states, self.racc_encoder(raccs), _timestep_raw(t, self.horizon)], -1)
        cond = jnp.broadcast_to(cond, actions.shape[:-1] + cond.shape[-1:])
        return self.e_net(jnp.concatenate([cond, actions], -1))[..., 0]

    def __call__(self, states, raccs, timesteps):
        return self.actor(states, raccs, timesteps)
