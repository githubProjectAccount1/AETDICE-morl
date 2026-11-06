from flax import nnx
import jax
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp
from src.utils import symlog, normalization

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def build_mog_dist(means, log_stds, logits, tanh_squash):
    """Diagonal-Gaussian mixture (optionally tanh-squashed) distribution builder.

    Shared verbatim by all three policies so they stay byte-identical in how they
    assemble the mixture dist. Exposes a closed-form `log_prob` — required by the
    WBC policy loss (−E[w·log π]).

    means:    [..., n_mixture, action_dim]
    log_stds: [..., n_mixture, action_dim]
    logits:   [..., n_mixture]
    """
    assert means.shape == log_stds.shape, (means.shape, log_stds.shape)
    assert logits.shape == means.shape[:-1], (logits.shape, means.shape)

    log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
    if not tanh_squash:
        means = nnx.tanh(means)

    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)),
    )

    if tanh_squash:
        return tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())
    return dist


class MuNetworkTransformed(nnx.Module):
    """Dual variable mu for the transformed AETDICE outer utility G.

    Gate is `config.utility_G_type` (not fair_alpha):
      - "linear"                        → mu fixed to config.pref_weights (or ones if unset).
      - "piecewise_log" → mu = softplus(theta), learnable,
                                          positivity-constrained (required because G*(mu)
                                          contains log(mu)).
    """

    def __init__(self, config):
        # mu gets dotted with F̃-telescoped rewards (shape [..., F̃-dim]), so its
        # length must match F̃-dim — NOT raw reward_dim. They coincide for pointwise
        # F (linear/exp/plog) but differ for CobbDouglas (F̃-dim=2 always).
        # 2026-11-03: racc_dim (조건부 입력 폭) 이 아니라 F 출력 차원에서 잰다 — rate 인코딩 3 채널
        # safe-CD 에서 둘이 달랐다 (3 vs 2). utility.utility_out_dim docstring 참조.
        from src.utility import LEARNABLE_G_TYPES, utility_out_dim   # SSOT
        self.reward_dim = utility_out_dim(config.utility_F, config.reward_dim)
        self.learnable = config.utility_G_type in LEARNABLE_G_TYPES
        pref_weights = getattr(config, "pref_weights", None)
        self.pref_weights = (
            jnp.array(pref_weights, dtype=jnp.float32)
            if pref_weights is not None
            else None
        )

        if self.learnable:
            # restrict mu's domain within the positive orthant by parameterizing with softplus.
            # softplus(theta) = 1.0  →  theta = log(exp(1.0) - 1) ≈ 0.5413
            init_theta = float(jnp.log(jnp.exp(jnp.float32(1.0)) - 1.0))
            self.theta = nnx.Param(jnp.full((self.reward_dim,), init_theta))

    def __call__(self):
        if self.learnable:
            return jax.nn.softplus(self.theta.value)
        if self.pref_weights is not None:
            return self.pref_weights
        return jnp.ones((self.reward_dim,), dtype=jnp.float32) # default to 1,1 pref weights for multi-objective setting, which corresponds to simple summation of reward dimensions (i.e. no preference bias).


class RaccEncoder(nnx.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = normalization(x, self.mean, self.std)

        return x



class MLP(nnx.Module):
    def __init__(
        self,
        din,
        dout=1,
        hidden_dims=[256, 256],
        activation=nnx.relu,
        rngs: nnx.Rngs = nnx.Rngs(0),
        activate_final: bool = False,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        zero_final: bool = False,   # zero-init the output Linear → module starts as f(x)=0
        # kernel_init=PYTORCH_INIT,
    ):
        dims = [din] + hidden_dims

        layer = []
        for i in range(len(dims) - 1):
            layer.append(
                nnx.Linear(
                    dims[i],
                    dims[i + 1],
                    rngs=rngs,
                    kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
                )
            )
            if dropout_rate > 0:
                layer.append(nnx.Dropout(dropout_rate, rngs=rngs))
            if layer_norm:
                layer.append(nnx.LayerNorm(dims[i + 1], rngs=rngs))
            layer.append(activation)

        if activate_final:
            layer.append(activation)

        if dout is not None:
            out_kernel_init = (
                nnx.initializers.zeros if zero_final
                else nnx.initializers.orthogonal(jnp.sqrt(2))
            )
            layer.append(
                nnx.Linear(
                    dims[-1],
                    dout,
                    rngs=rngs,
                    kernel_init=out_kernel_init,
                )
            )

        self.layer = nnx.Sequential(*layer)

    def __call__(self, x):
        return self.layer(x)


class ContinuousPolicy(nnx.Module):
    """Branch A (논문 Table 4 원본): racc-조건부 **multi-head** MoG policy.

    시간은 **입력이 아니라 출력 슬롯**이다 — backbone(state ⊕ racc)이 시간을 못 보고,
    마지막 Linear 가 `H · M · A` 를 뱉은 뒤 t 로 인덱싱한다. 즉 t 는 선형 사상만 고르고
    state×racc×time 상호작용을 만들 수 없다. 이 설계가 논문 헤드라인(AET 13.69)을 낸 것이고,
    Branch B(`TimeConditionedPolicy`)는 정확히 이 한계를 겨냥한 대안이다.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        reward_dim,
        racc_mean,
        racc_std,
        hidden_dims,
        horizon,
        layer_norm: bool = False,
        n_mixture: int = 5,
        tanh_squash_distribution: bool = False,
        log_std_scale: float = 1e-3,
        racc_dim=None,
        global_mixture: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_mixture = n_mixture
        self.tanh_squash = tanh_squash_distribution
        # ★ global_mixture (2026-09-21): 혼합 가중을 **상태에 의존하지 않는 전역 파라미터**로
        #   만든다. 그러면 성분 k 가 "이 상태에서의 국소 모드" 가 아니라 **하나의 정책**이 된다.
        #   왜 필요한가: AET 는 비선형 목적이라 최적해가 **정책들의 혼합**이다
        #   (Zahavy et al., Reward is Enough for Convex MDPs, NeurIPS 2021) — 에피소드 시작에
        #   정책 하나를 뽑아 끝까지 따르는 형태다. 상태 의존 logit 은 매 스텝 성분을 다시
        #   고르게 해서 Markov 확률정책이 되고, 그런 목적에서 증명된 준최적이다.
        #   실측 반증(2026-09-21): 기존 상태 의존 MoG 에서 성분을 에피소드에 고정하면
        #   12.9810 → 11.4116 으로 나빠진다 (성분이 정책이 아니라 국소 모드이기 때문).
        self.global_mixture = global_mixture

        self.racc_encoder = RaccEncoder(racc_mean, racc_std)

        # racc_dim = F̃-output dim flowing into the racc input. Defaults to reward_dim
        # for back-compat with pointwise F (linear/exp/plog/...) where the two coincide;
        # set explicitly when F changes the dim (e.g. CobbDouglas: F̃-dim=2 regardless
        # of reward_dim).
        if racc_dim is None:
            racc_dim = reward_dim
        input_dim = state_dim + racc_dim

        self.backbone = MLP(
            din=input_dim,
            hidden_dims=hidden_dims,
            dout=None,
            rngs=rngs,
            layer_norm=layer_norm,
        )
        self.mean_head = nnx.Linear(
            hidden_dims[-1], horizon * action_dim * n_mixture, rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
        )
        self.log_std_head = nnx.Linear(
            hidden_dims[-1], horizon * action_dim * n_mixture, rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(log_std_scale),
        )
        if global_mixture:
            # [H, M] — 시간별 혼합 가중, 상태 비의존. 성분 = 정책.
            self.logit_head = None
            self.global_logits = nnx.Param(jnp.zeros((horizon, n_mixture)))
        else:
            self.logit_head = nnx.Linear(
                hidden_dims[-1], horizon * n_mixture, rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            )

    def _build_dist(self, means, log_stds, logits):
        return build_mog_dist(means, log_stds, logits, self.tanh_squash)

    def __call__(self, states, raccs, timesteps=None):
        racc_embedding = self.racc_encoder(raccs)

        fused_feature = jnp.concatenate([states, racc_embedding], axis=-1)
        feature = self.backbone(fused_feature)

        B = states.shape[0]
        M, A = self.n_mixture, self.action_dim

        # [B, H, n_mixture, action_dim]
        all_means    = self.mean_head(feature).reshape(B, self.horizon, M, A)
        all_log_stds = self.log_std_head(feature).reshape(B, self.horizon, M, A)
        # [B, H, n_mixture]
        if self.global_mixture:
            all_logits = jnp.broadcast_to(self.global_logits.value[None], (B, self.horizon, M))
        else:
            all_logits = self.logit_head(feature).reshape(B, self.horizon, M)

        if timesteps is None:
            # batch_shape=[B, H], event_shape=[action_dim]
            return self._build_dist(all_means, all_log_stds, all_logits)

        t = jnp.clip(timesteps.astype(jnp.int32), 0, self.horizon - 1)

        # index along H dim for means/log_stds: [B, H, M, A] → [B, M, A]
        t_mA = jnp.broadcast_to(t.reshape(B, 1, 1, 1), (B, 1, M, A))
        means    = jnp.take_along_axis(all_means,    t_mA, axis=1).squeeze(1)
        log_stds = jnp.take_along_axis(all_log_stds, t_mA, axis=1).squeeze(1)

        # index along H dim for logits: [B, H, M] → [B, M]
        t_m = jnp.broadcast_to(t.reshape(B, 1, 1), (B, 1, M))
        logits = jnp.take_along_axis(all_logits, t_m, axis=1).squeeze(1)

        # batch_shape=[B], event_shape=[action_dim]
        return self._build_dist(means, log_stds, logits)







def _timestep_raw(t, horizon):
    """Raw normalized TIME-TO-GO (remaining time) in [0, 1], shape [..., 1]. Ablation: NO
    Fourier embedding — feeds the plain scalar so the MLP must learn its own time features.
    Uses remaining time (H-1-t)/(H-1): 1.0 at episode start (t=0) → 0.0 at the last step
    (t=H-1). 'How much time is left' is the ESR-relevant axis (when to commit)."""
    denom = float(max(horizon - 1, 1))
    return (((horizon - 1) - t.astype(jnp.float32)) / denom)[..., None]


class TimeConditionedPolicy(nnx.Module):
    """Branch B: racc + **remaining-time** 조건부 MoG policy — multi-head 없음.

    시간이 backbone **입력**으로 들어간다: `backbone(state ⊕ racc_norm ⊕ time_to_go)`.
    그래서 비선형 MLP 가 state×racc×time 상호작용을 계산할 수 있다 ("에피소드 후반 + obj0
    쪽으로 기울었으면 그쪽에 commit"). Branch A(multi-head)는 H 개 선형 head 가 **시간을 못 보는**
    feature 를 읽으므로 t 가 선형 사상만 고를 뿐 상호작용을 못 만든다 — 그 한계를 겨냥한 설계다.

    head 는 `F → M·A` 하나뿐이라(H 곱셈 없음) 파라미터가 A 대비 H 배 적다.
    AET/ESR 목적의 최적 정책은 (누적 보상, 남은 시간)에 대해 non-stationary 인데 racc 만으로는
    시간축이 없어 *언제* corner 에 commit 할지 결정할 수 없다 — 그게 이 분기의 동기다.

    시간 표현은 **raw time-to-go 스칼라 하나**로 고정한다 (Fourier 임베딩 경로는 2026-09-07 삭제).
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        reward_dim,
        racc_mean,
        racc_std,
        hidden_dims,
        horizon,
        layer_norm: bool = False,
        n_mixture: int = 5,
        tanh_squash_distribution: bool = False,
        log_std_scale: float = 1e-3,
        racc_dim=None,
        global_mixture: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_mixture = n_mixture
        self.tanh_squash = tanh_squash_distribution
        # ★ global_mixture (2026-09-21): 혼합 가중을 **상태에 의존하지 않는 전역 파라미터**로
        #   만든다. 그러면 성분 k 가 "이 상태에서의 국소 모드" 가 아니라 **하나의 정책**이 된다.
        #   왜 필요한가: AET 는 비선형 목적이라 최적해가 **정책들의 혼합**이다
        #   (Zahavy et al., Reward is Enough for Convex MDPs, NeurIPS 2021) — 에피소드 시작에
        #   정책 하나를 뽑아 끝까지 따르는 형태다. 상태 의존 logit 은 매 스텝 성분을 다시
        #   고르게 해서 Markov 확률정책이 되고, 그런 목적에서 증명된 준최적이다.
        #   실측 반증(2026-09-21): 기존 상태 의존 MoG 에서 성분을 에피소드에 고정하면
        #   12.9810 → 11.4116 으로 나빠진다 (성분이 정책이 아니라 국소 모드이기 때문).
        self.global_mixture = global_mixture

        self.racc_encoder = RaccEncoder(racc_mean, racc_std)
        if racc_dim is None:
            racc_dim = reward_dim
        # +1 = raw time-to-go 스칼라. racc 와 같은 자격의 입력.
        self.backbone = MLP(
            din=state_dim + racc_dim + 1, hidden_dims=hidden_dims, dout=None,
            rngs=rngs, layer_norm=layer_norm,
        )
        F = hidden_dims[-1]
        self.mean_head = nnx.Linear(
            F, action_dim * n_mixture, rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
        )
        self.log_std_head = nnx.Linear(
            F, action_dim * n_mixture, rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(log_std_scale),
        )
        if global_mixture:
            self.logit_head = None
            self.global_logits = nnx.Param(jnp.zeros((horizon, n_mixture)))
        else:
            self.logit_head = nnx.Linear(
                F, n_mixture, rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            )

    def _build_dist(self, means, log_stds, logits):
        return build_mog_dist(means, log_stds, logits, self.tanh_squash)

    def __call__(self, states, raccs, timesteps):
        """timesteps 는 **필수**다 — 이 분기는 per-timestep WBC 전용이라 all-H 경로가 없다.
        (Branch A 와 달리 `timesteps=None` 분기를 두지 않는다 — 죽은 분기 제거.)"""
        assert timesteps is not None, "TimeConditionedPolicy 는 timesteps 가 필수다"
        B, M, A = states.shape[0], self.n_mixture, self.action_dim
        sr = jnp.concatenate([states, self.racc_encoder(raccs)], axis=-1)     # [B, S+racc]
        t = jnp.clip(timesteps.astype(jnp.int32), 0, self.horizon - 1)
        feat = self.backbone(jnp.concatenate([sr, _timestep_raw(t, self.horizon)], axis=-1))
        assert feat.shape[0] == B, (feat.shape, B)
        logits = (self.global_logits.value[t] if self.global_mixture
                  else self.logit_head(feat).reshape(B, M))                   # [B, M]
        return self._build_dist(
            self.mean_head(feat).reshape(B, M, A),
            self.log_std_head(feat).reshape(B, M, A),
            logits,
        )                                                                     # batch [B]


EXTRACTION_MODES = ("original", "timecond", "perstep_multihead", "diffusion", "iproj")
# ★ iproj (2026-10-28) — OptiDICE 의 I-projection 추출. actor 는 timecond 와 같은 MoG 이고, e_net·π_D 를
#   한 모듈(src/nets/iproj_policy.py)에 같이 둔다. 손실은 src/learners/IProj_learner.py.
# ★ diffusion (2026-10-22) — per-timestep 채점 + time 입력은 timecond 와 같고, 정책족만
#   MoG → DDPM denoiser (src/nets/diffusion_policy.py), WBC 손실만 w-가중 denoising 으로
#   바꾼다 (src/learners/Diffusion_learner.py). DICE dual 은 AET_learner 그대로다.
# ★ perstep_multihead (2026-09-21) — **채점만 바꾼 대조군**이다.
#   original           = multi-head 구조 + all-H 채점 (논문 Table 4 재현 경로)
#   perstep_multihead  = multi-head 구조 + per-timestep 채점   ← 구조는 그대로, 채점만
#   timecond           = time-입력 구조  + per-timestep 채점
#   original vs perstep_multihead = **채점의 효과**, perstep_multihead vs timecond = **구조의 효과**.
#   CLAUDE.md 가 금지한 것은 "별도 플래그를 만들어 의도치 않은 조합이 생기는 것" 이고,
#   필요한 조합은 여기 EXTRACTION_MODES 에 이름을 붙여 추가하라고 명시한다 — 그 경로다.
#
# ★★ perstep_multihead 는 **ablation 전용이며 논문 설정이 아니다** (사용자 지시 2026-09-21
#     "필요하면 격리된 패키지로 분리해서 관리해. 다만 명확히 표시해").
#     · 결과는 **results_ablation_extraction/** 에만 쓴다 (results_Transformed / results_idrl 금지)
#     · 런처는 experiments/ablation_extraction/ 하나뿐이다
#     · 문서는 docs/ablation_extraction_2026-09-21.md
#     논문 표에 이 모드의 숫자를 올리지 마라 — 인과 분리용 대조군이다.


def build_continuous_policy(extraction, **kwargs):
    """정책 추출 방식 두 줄기 중 하나를 만든다 (제3의 조합을 만들지 않기 위한 단일 축).

      "original"      → ContinuousPolicy       + all-H WBC   (논문 Table 4 원본)
      "timecond" → TimeConditionedPolicy  + per-timestep WBC (remaining-time 입력)

    정책 구조와 WBC 채점 방식을 **한 축으로 묶는다.** 두 축을 따로 두면 "multi-head +
    per-timestep" 같은 의도치 않은 제3 조합이 생기고(2026-09-07 실제 발생), 어느 게 논문
    설정인지 추적 불가능해진다.
    """
    if extraction == "original":
        return ContinuousPolicy(**kwargs)
    if extraction == "perstep_multihead":
        return ContinuousPolicy(**kwargs)      # 구조는 original 과 동일 — 채점만 다르다
    if extraction == "timecond":
        return TimeConditionedPolicy(**kwargs)
    if extraction == "diffusion":
        # Lazy import: diffusion_policy imports MLP / RaccEncoder from this module.
        from src.nets.diffusion_policy import DiffusionPolicy
        return DiffusionPolicy(**kwargs)
    if extraction == "iproj":
        from src.nets.iproj_policy import IProjPolicy     # lazy: iproj_policy imports this module
        return IProjPolicy(**kwargs)
    raise ValueError(f"unknown extraction: {extraction!r} (택: {EXTRACTION_MODES})")


class FairDICEContinuousPolicy(nnx.Module):
    """Mixture-of-Gaussians policy for FairDICE finite-horizon (SER).

    No racc conditioning — input is state only.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims,
        horizon,
        layer_norm: bool = False,
        n_mixture: int = 5,
        tanh_squash_distribution: bool = False,
        log_std_scale: float = 1e-3,
        global_mixture: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_mixture = n_mixture
        self.tanh_squash = tanh_squash_distribution

        self.backbone = MLP(
            din=state_dim,
            hidden_dims=hidden_dims,
            dout=None,
            rngs=rngs,
            layer_norm=layer_norm,
        )
        self.mean_head = nnx.Linear(
            hidden_dims[-1], horizon * action_dim * n_mixture, rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
        )
        self.log_std_head = nnx.Linear(
            hidden_dims[-1], horizon * action_dim * n_mixture, rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(log_std_scale),
        )
        if global_mixture:
            # [H, M] — 시간별 혼합 가중, 상태 비의존. 성분 = 정책.
            self.logit_head = None
            self.global_logits = nnx.Param(jnp.zeros((horizon, n_mixture)))
        else:
            self.logit_head = nnx.Linear(
                hidden_dims[-1], horizon * n_mixture, rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            )

    def _build_dist(self, means, log_stds, logits):
        return build_mog_dist(means, log_stds, logits, self.tanh_squash)

    def __call__(self, states, timesteps=None):
        feature = self.backbone(states)

        B = states.shape[0]
        M, A = self.n_mixture, self.action_dim

        all_means    = self.mean_head(feature).reshape(B, self.horizon, M, A)
        all_log_stds = self.log_std_head(feature).reshape(B, self.horizon, M, A)
        all_logits   = self.logit_head(feature).reshape(B, self.horizon, M)

        if timesteps is None:
            return self._build_dist(all_means, all_log_stds, all_logits)

        t = jnp.clip(timesteps.astype(jnp.int32), 0, self.horizon - 1)
        t_mA = jnp.broadcast_to(t.reshape(B, 1, 1, 1), (B, 1, M, A))
        means    = jnp.take_along_axis(all_means,    t_mA, axis=1).squeeze(1)
        log_stds = jnp.take_along_axis(all_log_stds, t_mA, axis=1).squeeze(1)
        t_m = jnp.broadcast_to(t.reshape(B, 1, 1), (B, 1, M))
        logits = jnp.take_along_axis(all_logits, t_m, axis=1).squeeze(1)

        return self._build_dist(means, log_stds, logits)



