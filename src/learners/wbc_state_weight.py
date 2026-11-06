"""고정 critic 위에서 WBC 가중을 궤적 단위 / 행동 비율로 바꾸는 준비물 (--wbc_weight traj | action, 2026-10-26).

왜 (docs/CD_redesign_worksheet.md §7–§9): w(s,a) = d*(s)/d_D(s) · π*(a|s)/π_D(a|s) 인데, 상태 비율 쪽이 소수
상태(에피소드 초반 등)에 몰려 WBC 가 나머지 상태를 거의 배우지 않는다 (Walker CD: t<25 가 데이터 6% 에 질량 48–69%).
CD 상위 5% 필터 BC (E[CD_rate] 1.344) 는 고른 궤적의 모든 상태를 같은 가중으로 배운다.
  traj   : 궤적 j 의 모든 transition 에 m_j = 궤적 평균 자기시점 w. 필터 BC 의 연속판이고, 궤적 질량
           Σ_t w 는 그대로 두면서 궤적 안 상태 쏠림만 없앤다.
  action : ŵ(s, racc, t) ≈ E_D[w | s, racc, t] 를 회귀하고 w / ŵ 를 쓴다 = 행동 비율만 남긴다.
           모든 상태를 데이터 비율로 배우고, 상태 안에서 critic 이 좋게 본 행동 쪽으로 기운다.
           데이터는 상태마다 행동이 하나라 ŵ 가 외우면 w/ŵ ≡ 1 (= 무가중 BC) 이 된다 → 궤적 단위로
           검증 분할해 조기 종료한다. 상태로 예측 가능한 부분만 ŵ 에 들어간다.
둘 다 critic 이 고정이어야 한다 (가중을 학습 시작 전에 한 번 만든다). w 공식은 src/ops/w_reweight.py (SSOT).
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax

from src.divergence import FDivergence

_PER_TRAJ = 20            # 궤적당 균등 간격 표본 transition 수 (w_reweight.sample_per_traj)
_FIT_HIDDEN = (256, 256)
_FIT_LR = 1e-3
_FIT_BATCH = 4096
_FIT_MAX_STEPS = 20000
_FIT_EVAL_EVERY = 500
_FIT_PATIENCE = 6         # 검증 MSE 가 이만큼 연속으로 안 좋아지면 멈춘다 (평가 단위)
_VAL_FRAC = 0.1           # 궤적 단위 검증 비율
_FLOOR = 0.05             # ŵ 하한 (전체 평균 w = 1 단위) — w/ŵ 가 터지지 않게
WEIGHT_MODES = ("traj", "action")


def _frozen_run(config, buffer, train_state, learner):
    nu = learner.get_model(train_state.nu_state)[0]
    mu = np.asarray(learner.get_model(train_state.mu_state)[0]())
    assert not bool(getattr(config, "standardize_e", False)), "transition_w 는 standardize_e 를 재현하지 않는다"
    return dict(config=config, buffer=buffer, nu=nu, mu=mu, beta=float(config.beta),
                fdiv=FDivergence[config.divergence],
                w_max=float(getattr(config, "w_max", float("inf"))))


def _sampled_w(config, buffer, train_state, learner):
    # 지연 import: w_reweight → src.eval.rollout → learners → AET_learner 가 이 모듈을 import 하는 순환을 피한다.
    from src.ops.w_reweight import sample_per_traj, transition_w
    d = buffer.data
    idx, owner, lengths = sample_per_traj(np.asarray(d.traj_ids), _PER_TRAJ)
    w = transition_w(_frozen_run(config, buffer, train_state, learner), idx, head="own")
    assert w.shape == idx.shape and np.all(np.isfinite(w)), (w.shape, idx.shape)
    return idx, owner, lengths, w


def traj_weight_table(config, buffer, train_state, learner):
    """[n_traj] 궤적 평균 자기시점 w (전체 평균 1 로 정규화)."""
    _, owner, lengths, w = _sampled_w(config, buffer, train_state, learner)
    n = len(lengths)
    m = np.bincount(owner, weights=w, minlength=n) / np.bincount(owner, minlength=n)
    m = m / np.average(m, weights=lengths)          # transition 평균이 1 이 되게
    ess = (m * lengths).sum() ** 2 / ((m * lengths) ** 2).sum() / n
    print(f"[wbc traj] {n} 궤적, 궤적 ESS {ess:.3f}, max/mean {m.max():.1f}")
    return m.astype(np.float32)


def _features(d, idx, horizon):
    return np.concatenate([np.asarray(d.states[idx], np.float32), np.asarray(d.raccs[idx], np.float32),
                           (np.asarray(d.timesteps[idx], np.float32) / horizon)[:, None]], axis=1)


def _mlp_init(key, sizes):
    params = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        params.append((jax.random.normal(k, (din, dout)) * jnp.sqrt(2.0 / din), jnp.zeros(dout)))
    return params


def state_weight_apply(params, x):
    """ŵ(x) > 0 — 순수 함수 (train_step 의 jit 안에서 상수 params 로 부른다)."""
    mean, std, layers = params
    h = (x - mean) / std
    for W, b in layers[:-1]:
        h = jax.nn.relu(h @ W + b)
    W, b = layers[-1]
    return jax.nn.softplus(h @ W + b)[..., 0]


def fit_state_weight(config, buffer, train_state, learner, seed=0):
    """ŵ(s, racc, t) ≈ E_D[w | s, racc, t] (전체 평균 w = 1 단위). 반환 params (state_weight_apply 용)."""
    d = buffer.data
    idx, owner, lengths, w = _sampled_w(config, buffer, train_state, learner)
    y = (w / w.mean()).astype(np.float32)
    x = _features(d, idx, config.max_seq_len)
    rng = np.random.default_rng(seed)
    val_traj = rng.random(len(lengths)) < _VAL_FRAC
    va, tr = val_traj[owner], ~val_traj[owner]
    mean, std = x[tr].mean(0), x[tr].std(0) + 1e-6
    layers = _mlp_init(jax.random.PRNGKey(seed), (x.shape[1],) + _FIT_HIDDEN + (1,))
    tx = optax.adam(_FIT_LR)
    opt = tx.init(layers)
    xt, yt, xv, yv = (jnp.asarray(a) for a in (x[tr], y[tr], x[va], y[va]))

    def loss(layers, xb, yb):
        return jnp.mean((state_weight_apply((mean, std, layers), xb) - yb) ** 2)

    @jax.jit
    def step(layers, opt, key):
        i = jax.random.randint(key, (_FIT_BATCH,), 0, xt.shape[0])
        l, g = jax.value_and_grad(loss)(layers, xt[i], yt[i])
        upd, opt = tx.update(g, opt, layers)
        return optax.apply_updates(layers, upd), opt, l

    val_loss = jax.jit(loss)
    base = float(jnp.mean((yv - yt.mean()) ** 2))       # 상수 예측의 검증 MSE
    best, best_layers, bad, key = np.inf, layers, 0, jax.random.PRNGKey(seed + 1)
    for s in range(1, _FIT_MAX_STEPS + 1):
        key, k = jax.random.split(key)
        layers, opt, _ = step(layers, opt, k)
        if s % _FIT_EVAL_EVERY:
            continue
        v = float(val_loss(layers, xv, yv))
        if v < best:
            best, best_layers, bad = v, layers, 0
        else:
            bad += 1
            if bad >= _FIT_PATIENCE:
                break
    r2 = 1.0 - best / base
    print(f"[wbc action] ŵ(s,racc,t) 회귀: {len(y)} 표본, 검증 R² {r2:.3f} (상태가 설명하는 w 분산), step {s}")
    return (jnp.asarray(mean), jnp.asarray(std), best_layers), float(r2)


def attach(config, buffer, train_state, learner):
    """--wbc_weight traj | action 일 때 가중 준비물을 config 에 붙인다 (config 는 train_step 의 정적 클로저).

    config.json 은 이미 저장된 뒤라 큰 배열이 거기 들어가지 않는다. 재개(resume)해도 고정 critic 에서
    같은 표/회귀를 다시 만든다 (회귀는 seed 고정).
    """
    mode = str(getattr(config, "wbc_weight", "raw"))
    if mode not in WEIGHT_MODES:
        return
    assert getattr(config, "critic_from", None), "--wbc_weight traj|action 은 --critic_from (고정 critic) 이 필요하다"
    if mode == "traj":
        config.wbc_traj_table = traj_weight_table(config, buffer, train_state, learner)
    else:
        config.wbc_state_params, config.wbc_state_r2 = fit_state_weight(
            config, buffer, train_state, learner, seed=int(config.seed))


def action_ratio(config, w_b, states, raccs, t_b):
    """w / max(ŵ, floor) — w_b 는 원 단위라 ŵ (평균 1 단위) 와 맞추려 배치 평균으로 나누지 않고 비율 뒤 정규화한다."""
    x = jnp.concatenate([states, raccs, (t_b.astype(jnp.float32) / config.max_seq_len)[:, None]], axis=1)
    sw = jnp.maximum(state_weight_apply(config.wbc_state_params, x), _FLOOR)
    r = w_b / sw
    return r / (jnp.mean(r) + 1e-12)
