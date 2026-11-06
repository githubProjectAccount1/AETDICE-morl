from typing import Optional

import jax
import jax.numpy as jnp

EPS = 1e-8


def piecewise_log(x):
    """piecewise_log 형태(jnp): x≥1 → log(x+EPS) (포화), x<1 → −0.5·(x−2)²+0.5 (quadratic stitch).
    이 프로젝트의 piecewise_log 단일 소스 — Utility._piecewise_log, CobbDouglas, filter scoring 이
    모두 이 함수를 쓴다 (SSOT; 이전 numpy mirror `_piecewise_log_np` 는 제거)."""
    return jnp.where(x >= 1.0, jnp.log(x + EPS), -0.5 * (x - 2.0) ** 2 + 0.5)


def piecewise_log_threshold(x, threshold):
    """piecewise_log 를 (x − threshold) 에 적용. threshold utility 와 CobbDouglas dim 이 공유."""
    return piecewise_log(x - threshold)


def _cfg_get(cfg):
    """config 가 SimpleNamespace(학습)든 dict(체크포인트 reload)든 같은 방식으로 읽게 한다."""
    if isinstance(cfg, dict):
        return lambda k, d=None: cfg.get(k, d)
    return lambda k, d=None: getattr(cfg, k, d)


def cd_kwargs_from_config(cfg, *, rho=None):
    """config → `CobbDouglasUtility(**kwargs)` 인자 dict. **CD 설정의 단일 소스.**

    왜 이 함수가 있나 (2026-09-14): 소비처(학습 setup, eval-reload rollout, CD 채점
    evaluate)가 **각자 손으로** 인자를 받아 쓰다 보니 같은 부류의 버그가 세 번 났다 —
    체크포인트가 학습 때와 다른 CD 로 복원되는 divergence 다. 그래서 **config 키 목록을
    여기 한 곳에만** 둔다. 인자가 늘거나 줄면 이 함수만 고친다.

    ★ 2026-09-16: CD 를 단일 수식으로 확정하면서 mode/denom_offset/offset_shift/horizon/
      denom_offset_step/numerator_eps 를 전부 없앴다. 남은 건 rho / threshold / amp 뿐이다.
    ★ 2026-10-24: form ∈ {total, rate} (exp A) 추가. rate 는 horizon=max_seq_len 이 필요하다.

    rho: BC 처럼 채점 rho 가 학습 rho 와 다른 경우의 override (cd_filter_rho).
    """
    get = _cfg_get(cfg)
    # form 이 없는 옛 config 는 'total' (2026-10-24 이전 런은 전부 total 형태다).
    form = get("cobbdouglas_form", None) or "total"
    return dict(
        rho=float(rho if rho is not None else get("cobbdouglas_rho", 0.5)),
        threshold=get("reward_threshold"),
        amp=float(get("cobbdouglas_amp", 1.0)),
        form=form,
        horizon=(float(get("max_seq_len")) if form == "rate" else None),
        # 옛 config 에는 없다 → 기존 동작 (plog, κ=1). 2026-11-03 safe-CD 결합 방식.
        safety=get("cobbdouglas_safety", None) or "plog",
        kappa=float(get("cobbdouglas_kappa", None) or 1.0),
        safety_range=_cd_safety_range(get),
    )


def _cd_safety_range(get):
    """violog (D6.30) 만 스텝 safety 범위가 필요하다 — reshape 이름에서 (SSOT: reward_transform.SAFETY_STEP_RANGE)."""
    if (get("cobbdouglas_safety", None) or "plog") != "violog":
        return None
    from src.data.reward_transform import cd_safety_step_ranges
    return cd_safety_step_ranges(get("reward_reshape"))


def utility_out_dim(utility_F, reward_dim):
    """F 의 출력 차원 (= μ 의 길이, telescoped 보상 r̃ 의 차원). 0 벡터를 한 번 넣어 잰다.

    왜 (2026-11-03): μ 길이를 racc_dim (조건부 입력 폭) 으로 잡아 왔는데, 둘은 ftilde 인코딩에서만 같다.
    rate 인코딩 + 3 채널 safe-CD 는 racc_dim = 3 (원 채널의 스텝당 평균) 이고 CD 의 F 출력은 2 라
    μ·r̃ 에서 shape 가 어긋났다. F 출력 차원은 F 만이 알므로 F 에서 직접 잰다 (SSOT)."""
    x = jnp.zeros((1, int(reward_dim)), dtype=jnp.float32)
    kw = {"steps": jnp.ones((1,), dtype=jnp.float32)} if getattr(utility_F, "needs_steps", False) else {}
    out = utility_F(x, keep_dims=True, **kw)
    assert out.ndim == 2 and out.shape[0] == 1, out.shape
    return int(out.shape[-1])


def make_utility_F_from_config(cfg, *, strict_threshold=False):
    """config → inner utility F. 학습(setup.build_utility_F)과 eval-reload
    (rollout._rebuild_utility_F)가 **같은 F** 를 만들도록 하는 단일 진입점.
    CD 인자는 cd_kwargs_from_config 에서만 나온다."""
    get = _cfg_get(cfg)
    cd = cd_kwargs_from_config(cfg)
    return make_utility_F(
        get("utility_F_type"),
        power_p=get("power_p", 3.0),
        reward_threshold=cd["threshold"],
        cobbdouglas_rho=cd["rho"],
        cobbdouglas_amp=cd["amp"],
        cobbdouglas_form=cd["form"],
        cd_horizon=cd["horizon"],
        cd_safety=cd["safety"],
        cd_kappa=cd["kappa"],
        cd_safety_range=cd["safety_range"],
        exp_f_scale=get("exp_f_scale", 50.0),
        f_tau=get("f_tau"),
        f_sigmoid_scale=get("f_sigmoid_scale"),
        strict_threshold=strict_threshold,
    )


def make_utility_F(f_type, *, power_p=3.0, cobbdouglas_rho=0.5, reward_threshold=None,
                   cobbdouglas_amp=1.0, cobbdouglas_form="total", cd_horizon=None,
                   cd_safety="plog", cd_kappa=1.0, cd_safety_range=None,
                   exp_f_scale=50.0, f_tau=None, f_sigmoid_scale=None, strict_threshold=False):
    """inner utility F 인스턴스 생성 — training(`setup.build_utility_F`)과 eval-reload
    (`rollout._rebuild_utility_F`)의 단일 소스. 두 경로가 같은 F 를 만들도록 (checkpoint 가
    학습 때와 다른 F 로 reload 되는 divergence 방지 — 특히 exponential_F 의 exp_scale).
    cobbdouglas 는 reward_transform 에 있어 lazy import (순환 회피).
    strict_threshold=True 면 piecewise_log_threshold 의 threshold 누락 시 loud 실패 (checkpoint reload)."""
    if f_type == "power_F":
        return Utility(type="power_F", p=power_p)
    if f_type == "cobbdouglas":
        from src.data.reward_transform import CobbDouglasUtility
        return CobbDouglasUtility(rho=cobbdouglas_rho, threshold=reward_threshold,
                                  amp=cobbdouglas_amp, form=cobbdouglas_form, horizon=cd_horizon,
                                  safety=cd_safety, kappa=cd_kappa, safety_range=cd_safety_range)
    if f_type == "piecewise_log_threshold":
        if strict_threshold and reward_threshold is None:
            raise KeyError("reward_threshold missing for piecewise_log_threshold checkpoint")
        return Utility(type="piecewise_log_threshold", threshold=reward_threshold)
    if f_type == "sigmoid":
        if f_tau is None or f_sigmoid_scale is None:
            raise ValueError("F=sigmoid needs f_tau (목적별 목표) and f_sigmoid_scale")
        return Utility(type="sigmoid", tau=f_tau, sigmoid_scale=f_sigmoid_scale)
    return Utility(type=f_type, exp_scale=exp_f_scale)


class Utility:
    def __init__(self, type="linear", p: float = 3.0, threshold: float = 0.0,
                 exp_scale: float = 50.0, tau=None, sigmoid_scale=None):
        assert type in (
            "linear",
            "log",
            "piecewise_log",
            "piecewise_log_threshold",
            "exponential_F",
            "power_F",
            "sigmoid",
        )
        # sigmoid (2026-10-25, 후보 ① 목표 달성): f_d(R) = σ((R_d − τ_d)/s) ≈ 1[R_d ≥ τ_d].
        # τ 는 목적마다 다르다 (차원별 벡터) — 다른 F 의 스칼라 모수와 달리 [D] 로 브로드캐스트한다.
        self.tau = None if tau is None else jnp.asarray(tau, dtype=jnp.float32)
        self.sigmoid_scale = None if sigmoid_scale is None else float(sigmoid_scale)
        self.type = type
        self.p = p  # exponent for power_F; ignored by other utility types.
        self.threshold = threshold  # used by piecewise_log_threshold; ignored otherwise.
        # exp_scale: divisor in exponential_F = exp(x / exp_scale). Must match the racc
        # magnitude of the env (d4morl racc~300-400 → 50; MO-PointMaze racc~0-170 → ~10).
        # Too large → exp is ~linear over the racc range (F-convexity inactive).
        self.exp_scale = exp_scale

    def _log(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(x + EPS)

    def _piecewise_log(self, x: jnp.ndarray) -> jnp.ndarray:
        return piecewise_log(x)

    def _piecewise_log_threshold(self, x: jnp.ndarray) -> jnp.ndarray:
        """Plain piecewise_log applied to (x − threshold). No extra shift.

        Equivalent to feeding `x − T` into the existing _piecewise_log:
            x − T ≥ 1  (i.e., x ≥ T + 1):  log(x − T)            saturating reward
            x − T < 1  (i.e., x < T + 1):  −0.5·(x − T − 2)² + 0.5  quadratic stitch

        The kink (where the two pieces meet smoothly) sits at x = T + 1.
        At x = T exactly: F = −1.5 (one unit below the kink). Below threshold,
        the quadratic gradient grows linearly with the deficit, pushing x up;
        above threshold + 1, log diminishing returns kicks in.
        """
        return self._piecewise_log(x - self.threshold)

    def _exponential_F(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(x / self.exp_scale)

    def _power_F(self, x: jnp.ndarray, p: float) -> jnp.ndarray:
        if p < 0:
            raise ValueError("Power p must be non-negative.")
        return jnp.power(x + EPS, p)

    def _transform(self, R: jnp.ndarray) -> jnp.ndarray:
        if self.type == "linear":
            return R
        elif self.type == "log":
            return self._log(R)
        elif self.type == "piecewise_log":
            return self._piecewise_log(R)
        elif self.type == "piecewise_log_threshold":
            return self._piecewise_log_threshold(R)
        elif self.type == "exponential_F":
            return self._exponential_F(R)
        elif self.type == "power_F":
            return self._power_F(R, self.p)  # p is set at construction time
        elif self.type == "sigmoid":
            assert R.shape[-1] == self.tau.shape[-1], (R.shape, self.tau.shape)
            return jax.nn.sigmoid((R - self.tau) / self.sigmoid_scale)
        else:
            raise NotImplementedError

    def __call__(self, R: jnp.ndarray, *, keep_dims: bool = False) -> jnp.ndarray:
        X = self._transform(R)  # [..., D]
        return X if keep_dims else jnp.sum(X, axis=-1)


def _log_conjugate(mu, eps):
    """G*(-mu) for G = piecewise_log. Standard domain mu < 1 uses -1 - log(mu);
    mu >= 1 uses a smooth quadratic extension matching piecewise_log's primal."""
    is_standard_domain = mu < 1.0
    val_standard = -1.0 - jnp.log(mu + eps)
    val_extended = 0.5 * jnp.power(mu, 2) - 2.0 * mu + 0.5
    return jnp.where(is_standard_domain, val_standard, val_extended)


# ------- Outer utility G: primal + conjugate (SSOT) -------
# G 는 기대값 수준 집계다: J = G(E[F̃(R)]). F̃ 는 buffer 의 telescoping 보상에 흡수되고, G 는 학습되는 쌍대 변수
# μ 와 켤레 G* 로 들어간다 (AET_learner: loss_3 = conjugate_G(μ)). 평가의 자기 목적 점수는 primal 을 쓴다.
#
# 켤레 규약: G*(μ) = sup_v ( G(v) − μ·v ).  mu_scaling_factor S 는 G 를 S 배 한 것과 같다:
#   S·G*(μ/S) = (S·G)*(μ)  →  min_μ [μ·v + S·G*(μ/S)] = S·G(v).
#
# 종류 (가산분리 G(v) = Σ_d w_d g(v_d), w = pref_weights 또는 1):
#   linear          w·v                         — μ = w 고정 (MuNetwork), G* = 0
#   piecewise_log   Σ plog(v_d)                 — 기존 AETDICE (가중치 없음)
#   weighted_log    Σ w_d log v_d               — 후보 ① (가중 Nash). 2026-10-25
#   isoelastic      Σ w_d h_p(v_d)              — 후보 ② 의 SER 비교군 (기대 return 의 등탄력 후생)
#   isoelastic_ce   Σ w_d h_p(θ·log(1 + v_d))   — 후보 ② AET: F̃ = exp(R/θ)−1 의 확실성 등가 CE 에 h_p.
#                                                 CES(CE) 와 최대화점이 같다 (단조 변환). θ = exp_f_scale.
#   h_p(u) = u^p / p (p<1, p≠0),  log u (p=0).  p→−∞ max-min, p=0 Nash, p→1 공리주의.
LEARNABLE_G_TYPES = ("piecewise_log", "weighted_log", "isoelastic", "isoelastic_ce")
WEIGHTED_G_TYPES = ("linear", "weighted_log", "isoelastic", "isoelastic_ce")   # pref_weights 를 받는 G
_CE_NEWTON_ITERS = 40          # isoelastic_ce 켤레의 1-D Newton 반복 (log u 공간, 단조 수렴)
_LOG_U_CLIP = (-20.0, 20.0)


def g_kwargs_from_config(cfg):
    """config → G 모수 dict (conjugate_G / utility_G 공용 — 학습·평가가 같은 G 를 쓰게)."""
    get = _cfg_get(cfg)
    return dict(weights=get("pref_weights"), iso_p=get("g_iso_p"), theta=get("exp_f_scale", 50.0))


def _h_p(u, p):
    return jnp.log(u) if p == 0 else jnp.power(u, p) / p


def _g_weights(weights, d):
    w = jnp.ones((d,), jnp.float32) if weights is None else jnp.asarray(weights, jnp.float32)
    assert w.shape == (d,), (w.shape, d)
    return w


def utility_G(v, utility_type, *, weights=None, iso_p=None, theta=50.0):
    """primal G(v) — v = E[F̃(R)] [D]. 평가의 자기 목적 점수용."""
    v = jnp.asarray(v, jnp.float32)
    d = v.shape[-1]
    if utility_type == "piecewise_log":
        return jnp.sum(piecewise_log(v), axis=-1)
    w = _g_weights(weights, d)
    if utility_type == "linear":
        return jnp.sum(w * v, axis=-1)
    if utility_type == "weighted_log":
        return jnp.sum(w * jnp.log(v), axis=-1)
    if utility_type == "isoelastic":
        return jnp.sum(w * _h_p(v, iso_p), axis=-1)
    if utility_type == "isoelastic_ce":
        return jnp.sum(w * _h_p(theta * jnp.log1p(v), iso_p), axis=-1)
    raise ValueError(f"utility_G: unsupported utility_type={utility_type!r}")


def _isoelastic_ce_conjugate(mu, w, p, theta):
    """sup_v [w·h_p(θ log(1+v)) − μ v] (원소별). 1계 조건을 u = θ log(1+v) 로 쓰면
        ψ(z) = (p−1)·z − e^z/θ + log(wθ/μ) = 0,  z = log u   (ψ 는 감소·오목 → Newton 이 한 번 넘어간 뒤 단조 수렴)
    해 u* 는 stop_gradient — envelope 정리로 ∂G*/∂μ = −v* 가 그대로 나온다."""
    c = jnp.log(w * theta / mu)
    z = jnp.full_like(mu, jnp.log(theta))
    for _ in range(_CE_NEWTON_ITERS):
        psi = (p - 1.0) * z - jnp.exp(z) / theta + c
        dpsi = (p - 1.0) - jnp.exp(z) / theta
        z = jnp.clip(z - psi / dpsi, *_LOG_U_CLIP)
    u = jax.lax.stop_gradient(jnp.exp(z))
    v = jnp.expm1(u / theta)
    return w * _h_p(u, p) - mu * v


def conjugate_G(mu, utility_type, scale_factor=1, eps=1e-6, *, weights=None, iso_p=None, theta=50.0):
    scaled_mu = mu / scale_factor # usually 10k or 1million, to account for scale difference between raw return and AET dual variables. This scaling is important for numerical stability of the conjugate utility, especially when G is nonlinear. The choice of scaling factor can depend on the environment and the expected range of returns; it may require tuning to ensure that the inputs to the conjugate utility are in a numerically stable range.

    if utility_type == "linear":
        # G(x) = w·x is linear; its Legendre transform is the indicator of {mu = w}.
        # Preference weights already pin mu = w via MuNetwork, so G*(mu) = 0.
        return 0.0

    if utility_type == "piecewise_log":
        return scale_factor * jnp.sum(_log_conjugate(scaled_mu, eps))

    w = _g_weights(weights, scaled_mu.shape[-1])
    m = scaled_mu + eps
    if utility_type == "weighted_log":
        # sup_v w log v − μ v  →  v* = w/μ
        return scale_factor * jnp.sum(w * (jnp.log(w / m) - 1.0))
    if utility_type == "isoelastic":
        if iso_p is None or not iso_p < 1.0:
            raise ValueError(f"isoelastic G needs p < 1 (got {iso_p!r})")
        if iso_p == 0:
            return scale_factor * jnp.sum(w * (jnp.log(w / m) - 1.0))
        vstar = jnp.power(w / m, 1.0 / (1.0 - iso_p))            # v* = (w/μ)^{1/(1−p)}
        return scale_factor * jnp.sum(m * vstar * (1.0 - iso_p) / iso_p)
    if utility_type == "isoelastic_ce":
        if iso_p is None or not iso_p < 1.0:
            raise ValueError(f"isoelastic_ce G needs p < 1 (got {iso_p!r})")
        return scale_factor * jnp.sum(_isoelastic_ce_conjugate(m, w, float(iso_p), float(theta)))

    raise ValueError(
        f"conjugate_G: unsupported utility_type='{utility_type}'. "
        f"Supported: linear, {LEARNABLE_G_TYPES}."
    )
