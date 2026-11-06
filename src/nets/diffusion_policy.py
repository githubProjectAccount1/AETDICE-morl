"""Diffusion policy for the `--extraction diffusion` branch (2026-10-22).

왜 diffusion 인가: WBC (−E[w·log π]) 는 closed-form log_prob 가 필요해서 정책족이
MoG / Laplace / flow 로 묶였다. denoising 손실은 log_prob 없이 **같은 타겟**
π*(a|s) ∝ π_D(a|s)·w(s,a) 를 forward-KL 쪽으로 맞추므로 (mode-covering) AETDICE 의
양 corner 혼합을 한 모드로 무너뜨리지 않는다. Q-최대화 항(Diffusion-QL)은 넣지 않는다 —
mode-seeking 이라 DICE-최적 w¹ 투영을 이탈한다 (soft-w³ 가 −0.11 손해본 것과 같은 이유).

조건부 입력은 TimeConditionedPolicy 와 같다: state ⊕ racc_norm ⊕ raw time-to-go.
classifier-free guidance(CFG) 는 **racc 에만** 건다 — 정책이 racc 를 무시하는 것이 측정된
병목이고(민감도 0.004 vs critic 16.25), guidance 스케일 λ 가 그 조건 신호를 직접 증폭한다.
null 토큰은 racc 를 0 으로 두고 `racc_present` 비트를 0 으로 둔 것이다 (racc_norm=0 은 평균
racc 라 비트 없이는 "racc 없음" 과 구분되지 않는다).

eval 은 `policy(states, raccs, timesteps)` 가 돌려준 객체의 `.sample(seed=)` 만 부른다
(src/eval/evaluate.py). 그래서 `DiffusionSampler` 가 그 계약을 그대로 구현하고 eval 코드는
바꾸지 않는다. deterministic eval(`mean()` / mixture mode)은 diffusion 에 정의가 없으므로
조용히 대체하지 않고 에러를 낸다.
"""
import math

import jax
import jax.numpy as jnp
from flax import nnx

from src.nets.policy import MLP, RaccEncoder, _timestep_raw

# Cosine schedule offset (Nichol & Dhariwal 2021) — keeps β_1 from being vanishingly small.
_COSINE_S = 0.008
# Upper clip on β_k: the cosine schedule's last step otherwise approaches β=1 (singular).
_BETA_MAX = 0.999
# Number of sinusoidal frequencies for the diffusion-step embedding (→ 2× features).
_STEP_EMB_FREQS = 8
# Valid action range: evaluate.py clips to [-1, 1] before scaling, and the MoG policies
# squash means with tanh, so dataset actions live in this box.
ACTION_LOW, ACTION_HIGH = -1.0, 1.0


def cosine_alpha_bars(n_steps: int) -> jnp.ndarray:
    """ᾱ_k for k = 1..K, shape [K], with β_k clipped at _BETA_MAX."""
    assert n_steps >= 1, n_steps
    k = jnp.arange(n_steps + 1, dtype=jnp.float32)
    f = jnp.cos(((k / n_steps) + _COSINE_S) / (1.0 + _COSINE_S) * (math.pi / 2.0)) ** 2
    alpha_bar = f / f[0]
    betas = jnp.clip(1.0 - alpha_bar[1:] / alpha_bar[:-1], 0.0, _BETA_MAX)
    return jnp.cumprod(1.0 - betas)                                         # [K]


def _step_embedding(k_idx, n_steps):
    """Sinusoidal features of the diffusion step, [..., 2·_STEP_EMB_FREQS]."""
    x = (k_idx.astype(jnp.float32) / float(n_steps))[..., None]
    freqs = (2.0 ** jnp.arange(_STEP_EMB_FREQS, dtype=jnp.float32)) * math.pi
    return jnp.concatenate([jnp.sin(x * freqs), jnp.cos(x * freqs)], axis=-1)


class DiffusionPolicy(nnx.Module):
    """ε-prediction denoiser ε_θ(a_k, k | state, racc, time-to-go) + DDPM sampler."""

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
        racc_dim=None,
        diffusion_steps: int = 20,
        uncond_prob: float = 0.0,
        cfg_scale: float = 1.0,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **unused_mog_kwargs,
    ):
        # build_continuous_policy forwards the MoG-only knobs to every branch. They have no
        # meaning here — reject a non-default rather than silently ignoring it.
        assert not unused_mog_kwargs.get("global_mixture", False), \
            "global_mixture 는 MoG 전용이다 — diffusion 과 함께 쓸 수 없다"
        assert 0.0 <= uncond_prob < 1.0, uncond_prob
        assert diffusion_steps >= 1, diffusion_steps
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_steps = int(diffusion_steps)
        self.uncond_prob = float(uncond_prob)
        self.cfg_scale = float(cfg_scale)
        self.racc_encoder = RaccEncoder(racc_mean, racc_std)
        if racc_dim is None:
            racc_dim = reward_dim
        self.racc_dim = racc_dim
        # a_k ⊕ state ⊕ racc_norm·present ⊕ present-bit ⊕ time-to-go ⊕ step-emb
        din = action_dim + state_dim + racc_dim + 1 + 1 + 2 * _STEP_EMB_FREQS
        self.denoiser = MLP(
            din=din, dout=action_dim, hidden_dims=list(hidden_dims),
            activation=jax.nn.mish, layer_norm=layer_norm, rngs=rngs,
        )
        # Fixed schedule, not a parameter (nnx.Variable keeps it out of the optimizer).
        self.alpha_bars = nnx.Variable(cosine_alpha_bars(self.n_steps))

    def predict_eps(self, noisy_actions, k_idx, states, raccs, timesteps, racc_present):
        """ε̂, shape [B, A]. `racc_present` ∈ {0,1}^[B] — 0 is the CFG null token."""
        B = states.shape[0]
        assert noisy_actions.shape == (B, self.action_dim), noisy_actions.shape
        assert k_idx.shape == (B,) and racc_present.shape == (B,), (k_idx.shape,
                                                                    racc_present.shape)
        present = racc_present.astype(jnp.float32)[:, None]                  # [B, 1]
        racc_in = self.racc_encoder(raccs) * present                          # [B, R]
        t = jnp.clip(timesteps.astype(jnp.int32), 0, self.horizon - 1)
        x = jnp.concatenate([
            noisy_actions, states, racc_in, present,
            _timestep_raw(t, self.horizon), _step_embedding(k_idx, self.n_steps),
        ], axis=-1)
        eps = self.denoiser(x)
        assert eps.shape == (B, self.action_dim), eps.shape
        return eps

    def __call__(self, states, raccs, timesteps):
        assert timesteps is not None, "DiffusionPolicy 는 timesteps 가 필수다 (per-timestep 전용)"
        return DiffusionSampler(self, states, raccs, timesteps)


class DiffusionSampler:
    """Distribution-like handle that evaluate.py can `.sample(seed=)` from."""

    def __init__(self, policy: DiffusionPolicy, states, raccs, timesteps):
        self.policy, self.states, self.raccs, self.timesteps = policy, states, raccs, timesteps

    def _guided_eps(self, x, k_idx):
        p = self.policy
        B = x.shape[0]
        eps_c = p.predict_eps(x, k_idx, self.states, self.raccs, self.timesteps,
                              jnp.ones((B,), jnp.float32))
        if p.cfg_scale == 1.0:      # Python-time branch: plain conditional, skip the null pass
            return eps_c
        eps_u = p.predict_eps(x, k_idx, self.states, self.raccs, self.timesteps,
                              jnp.zeros((B,), jnp.float32))
        return eps_u + p.cfg_scale * (eps_c - eps_u)

    def sample(self, seed):
        """Ancestral DDPM sampling with x̂0 clipped to the action box. Shape [B, A]."""
        p = self.policy
        B, A, K = self.states.shape[0], p.action_dim, p.n_steps
        alpha_bars = p.alpha_bars.value
        key, sub = jax.random.split(seed)
        x = jax.random.normal(sub, (B, A))
        # Python loop: K is small and static; unrolling keeps the nnx module call trivially
        # traceable inside evaluate.py's jax.jit closure.
        for k in reversed(range(K)):
            ab = alpha_bars[k]
            ab_prev = alpha_bars[k - 1] if k > 0 else jnp.float32(1.0)
            beta = 1.0 - ab / ab_prev
            eps = self._guided_eps(x, jnp.full((B,), k, jnp.int32))
            x0 = jnp.clip((x - jnp.sqrt(1.0 - ab) * eps) / jnp.sqrt(ab),
                          ACTION_LOW, ACTION_HIGH)
            coef_x0 = beta * jnp.sqrt(ab_prev) / (1.0 - ab)
            coef_xt = (1.0 - ab_prev) * jnp.sqrt(1.0 - beta) / (1.0 - ab)
            x = coef_x0 * x0 + coef_xt * x
            if k > 0:
                key, sub = jax.random.split(key)
                var = beta * (1.0 - ab_prev) / (1.0 - ab)
                x = x + jnp.sqrt(var) * jax.random.normal(sub, (B, A))
        assert x.shape == (B, A), x.shape
        return x

    def mean(self):
        raise NotImplementedError(
            "diffusion 정책에는 closed-form mean/mode 가 없다 — deterministic eval 은 쓰지 마라")

    def log_prob(self, actions):
        raise NotImplementedError(
            "diffusion 정책의 log_prob 는 intractable 이다 — Diffusion_learner 의 denoising 손실을 써라")
