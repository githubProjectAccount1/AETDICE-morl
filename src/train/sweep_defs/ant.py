"""Sweep dict definitions for MO-Ant-v2.

Each `<ENV>_<DATASET>_<METHOD>` constant is a sweep dict consumed by
`sweeps_active.SWEEPS` and expanded into per-job dicts by
`src.train.dispatcher.sweep_to_jobs`.
"""
from src.train.dispatch_settings import DIVERGENCE, MAX_SEQ_LEN
from src.train.sweep_builder import BETAS, PREFS_3, build_sweep


# Table 4 (MO-Ant Expert, Concave G × Convex F) AET-DICE row — 5 seeds to match
# the paper (Table 4 caption: mean ± std over 5 random seeds). χ² divergence,
# β∈{1e-4,1e-5,1e-6}, musf=1e4. No std_e / clip_chi / symexp tricks (plain CHI).
ANT_EXPERT_PLOG_EXP   = build_sweep("ant", "expert", "plog_exp",   learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None], seeds=[1, 2, 3, 4, 5])

# Clip-off reproduction check: AET-DICE best-β (1e-6) × 5 seeds, gradient clip OFF
# (grad_clip defaults to 0 now). Tests whether removing the refactor-era always-on
# clip recovers the original G,F≈13.69 / higher obj0. Canonical dirs (clipped
# originals archived to _archive_clipped_aet/).
ANT_EXPERT_PLOG_EXP_B6 = build_sweep("ant", "expert", "plog_exp",  learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None], seeds=[1, 2, 3, 4, 5], beta=["1e-6"])


# OPTIDICE and FAIRDICE
# Table 4 LIN-[...] rows = OptiDICE (linear G × linear F, per fixed pref).
# Exact Table 4 HP: β=1e-6 (matched from the paper's reported per-method best-β). 5 seeds.
ANT_EXPERT_OPTIDICE   = build_sweep("ant", "expert", "optidice",   learner_type="fairdice", g_type="linear",        f_type="linear",        prefs=PREFS_3, seeds=[1, 2, 3, 4, 5], beta=["1e-6"])

# Table 4 FAIRDICE row (concave-G plog × linear F, learnable μ). Exact HP: β=1e-5. 5 seeds.
ANT_EXPERT_FAIRDICE   = build_sweep("ant", "expert", "fairdice",   learner_type="fairdice", g_type="piecewise_log", f_type="linear",        prefs=[None], seeds=[1, 2, 3, 4, 5], beta=["1e-5"])

# OptiDICe and FairDICe with Racc in. We use the same setup as in OptiDICe and FairDICE but in AET learner.
# ANT_EXPERT_OPTIDICE_AETDICE = build_sweep("ant", "expert", "opti-aetdice", learner_type="aet",      g_type="linear",        f_type="linear", prefs=PREFS_3)
# ANT_EXPERT_FAIRDICE_AETDICE = build_sweep("ant", "expert", "fair-aetdice", learner_type="aet",      g_type="piecewise_log", f_type="linear", prefs=[None])

# IQL ESR baselines (state+racc; G-linear only — μ fixed to pref_weights). One sweep per F:
# Table 4 ESR-IQL rows (state+racc Q+V+AWR, linear G × exp F, per fixed pref).
# 5 seeds. β pinned to a single value: IQL_learner does NOT consume β/divergence
# (no DICE dual), so β is irrelevant. Pinned to 1e-5 to match Table 4's ESR-IQL
# label; the β token in the run name is cosmetic for IQL.
ANT_EXPERT_IQL_EXP     = build_sweep("ant", "expert",  "iql_exp",     learner_type="iql", g_type="linear", f_type="exponential_F",  prefs=PREFS_3, seeds=[1, 2, 3, 4, 5], beta=["1e-5"])

ANT_EXPERT_IQL_PLOG    = build_sweep("ant", "expert",  "iql_plog",    learner_type="iql", g_type="linear", f_type="piecewise_log",  prefs=PREFS_3)


# linear-exp & log-exp
ANT_AMATEUR_LINEAR_EXP = build_sweep("ant", "amateur", "linear_exp", learner_type="aet",      g_type="linear",        f_type="exponential_F", prefs=PREFS_3)

ANT_AMATEUR_PLOG_EXP   = build_sweep("ant", "amateur", "plog_exp",   learner_type="aet",      g_type="piecewise_log", f_type="exponential_F", prefs=[None])

# linear-log & log-log
ANT_AMATEUR_LINEAR_PLOG = build_sweep("ant", "amateur", "linear_plog", learner_type="aet",      g_type="linear",        f_type="piecewise_log", prefs=PREFS_3)

ANT_AMATEUR_PLOG_PLOG   = build_sweep("ant", "amateur", "plog_plog",   learner_type="aet",      g_type="piecewise_log", f_type="piecewise_log", prefs=[None])

# OPTIDICE and FAIRDICE
ANT_AMATEUR_OPTIDICE   = build_sweep("ant", "amateur", "optidice",   learner_type="fairdice", g_type="linear",        f_type="linear",        prefs=PREFS_3)

ANT_AMATEUR_FAIRDICE   = build_sweep("ant", "amateur", "fairdice",   learner_type="fairdice", g_type="piecewise_log", f_type="linear",        prefs=[None])

# OptiDICe and FairDICe with Racc in. We use the same setup as in OptiDICe and FairDICE but in AET learner.
# ANT_AMATEUR_OPTIDICE_AETDICE = build_sweep("ant", "amateur", "opti-aetdice", learner_type="aet",      g_type="linear",        f_type="linear", prefs=PREFS_3)
# ANT_AMATEUR_FAIRDICE_AETDICE = build_sweep("ant", "amateur", "fair-aetdice", learner_type="aet",      g_type="piecewise_log", f_type="linear", prefs=[None])

# IQL ESR baselines (state+racc; G-linear only — μ fixed to pref_weights). One sweep per F:
ANT_AMATEUR_IQL_EXP    = build_sweep("ant", "amateur", "iql_exp",     learner_type="iql", g_type="linear", f_type="exponential_F",  prefs=PREFS_3)

ANT_AMATEUR_IQL_PLOG   = build_sweep("ant", "amateur", "iql_plog",    learner_type="iql", g_type="linear", f_type="piecewise_log",  prefs=PREFS_3)


# ── MO-Ant CD 본 격자 (CD 단일수식 확정 후 전면 재학습, 2026-09-16) ────────────
# CD = Reward^ρ / Cost^(1−ρ),  Reward = Σ(vx+vy),  Cost = Σ(1 + ‖a‖²).
#   정규화는 shift 없는 scale-only (0, T) — shift 를 넣으면 정지 정책이 1 등이 된다.
#   이전 CD 런/오라클은 전부 폐기했다 (정규화 shift 오류 + 고정 T 가정 버그).
#
# ★ ρ 격자를 {0.1,0.5,0.9} → {0.2,0.3,0.4} 로 옮긴 이유 (2026-09-16 실측):
#   CD 는 점근적으로 (2ρ−1)-동차라 **ρ≥0.5 에는 내부 최적이 없다** — 전부 "최고속"이라는
#   같은 답으로 수렴한다. 내부 최적이 있는 구간은 ρ<0.5 뿐이고, 에너지 프론티어
#   (‖a‖² ≈ 0.15·v, 오라클 4 동작점 적합) 기준으로 속도 2.3~12.8 m/s 를 덮는 구간이
#   ρ∈[0.20, 0.41] 이다. 예상 v*: ρ=0.2→2.0, 0.3→4.4, 0.4→11.7 m/s.
#   ※ 프론티어 적합이 4 점뿐이라 v* 예측은 어림이다 — 오라클 결과로 격자를 재조정한다.
#
# ρ 격자는 **오라클과 정렬**한다 — 그래야 perf 를 계산할 수 있다.
# β 는 축으로 둔다: 1e-5 = FairDICE/IQL 관례값, 1e-6 = AETDICE headline 값.
# ★ ρ 마다 버퍼 캐시가 새로 만들어진다 (ρ 당 ~6.6GB). β·seed 는 같은 ρ 안에서 공유한다.
# G=linear 인 이유: CD 는 두 목적을 스칼라 하나로 접으므로(dim0=CD, dim1≡0) 바깥 G 가
#   단조증가이기만 하면 argmax 가 같다 — 오라클도 같은 이유로 단일 PPO 를 썼다.
ANT_EXPERT_CD_GRID = build_sweep(
    "ant", "expert", "cd_grid", learner_type="aet", g_type="linear", f_type="cobbdouglas",
    prefs=[[1.0, 0.0]], reward_reshape="antcd", seeds=[1, 2, 3],
    cobbdouglas_rhos=[0.2, 0.3, 0.4],
    beta=["1e-5", "1e-6"],
)


# ── BC 바닥선 (CD 표의 worst 기준) ───────────────────────────────────────────────
# 왜: random uniform 은 "아무것도 못 하는" 바닥이라 기준이 너무 낮다 (v=-0.93, 낙상 55%).
#   더 의미 있는 바닥은 **데이터셋을 그대로 따라 하는 정책** = BC 다. 데이터셋의 99.63% 가
#   v∈[5,8] 이므로 BC 는 v≈6.7 을 낼 것이고, 그러면 "오프라인 CD 가 BC 를 넘는가"가
#   핵심 질문이 된다 (넘지 못하면 CD 목적이 아무 일도 안 한 것이다).
# k_pct=1.0 = 전 궤적 유지 → **필터 없는 순수 BC**. cd_filter_rho 는 CLI 검증이 쌍으로
#   요구해서 넣을 뿐, k_pct=1.0 이면 no-op 이라 학습에 영향이 없다.
ANT_EXPERT_CD_BC = build_sweep(
    "ant", "expert", "cd_bc", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], reward_reshape="antcd", seeds=[1],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
)


# ── BC 바닥선 (AET/ESR/SER 표용) ────────────────────────────────────────────────
# CD 용 BC(ANT_EXPERT_CD_BC)는 **antcd 공간**이라 AET/ESR/SER 을 못 잰다 — 그쪽은
# (reward=vx+vy, cost=‖a‖²) 로 재조립돼 원래 두 목적(vx, vy)이 사라지기 때문이다.
# canonical 표의 바닥선은 reshape="none" 으로 따로 돌려 (vx, vy) 누적을 그대로 얻어야 한다.
# k_pct=1.0 = 전 궤적 유지 → 필터 없는 순수 BC (cd_filter_rho 는 CLI 가 쌍으로 요구할 뿐 no-op).
# ★ seed 5 개 — 알고리즘 3 팔(5 seed)·오라클(5 seed)과 표본 크기를 맞춘다. BC 는 perf 의
#   **분모(worst)** 라 seed 가 적으면 분모 산포가 과소평가되어 perf 가 과신된다
#   (2026-09-16: 3 seed 였던 것을 5 로 올림).
ANT_EXPERT_BC = build_sweep(
    "ant", "expert", "bc", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3, 4, 5],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
)


# ── ★ racc-조건부 BC 바닥선 (augmented MDP) ───────────────────────────────────
# 왜: perf 의 분모가 알고리즘과 **같은 정책 클래스**에 있어야 한다. 이 목적함수는 return 에
# 비선형(G∘F)이라 convex MDP 이고, convex MDP 에서 Markov 정책 π(a|s) 는 최적이 될 수 없다
# (Zahavy et al. 2021). 알고리즘(AETDICE)도 오라클(FW mixture / racc_aware PPO)도 전부
# history-dependent 인데 BC 만 base MDP 의 Markov 정책이었다 → perf 의 일부가 "알고리즘이
# 잘해서"가 아니라 "바닥선이 racc 를 못 봐서" 였다 (2026-09-16 사용자 지적).
# racc-BC 는 AETDICE 와 같은 ContinuousPolicy + 같은 all-H 채점을 쓰고 가중치만 w ≡ 1 이다
# → perf 가 **학습된 w 가 벌어주는 것만** 측정한다.
ANT_EXPERT_BC_RACC = build_sweep(
    "ant", "expert", "bc_racc", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3, 4, 5], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
)





# ══ mjx-native 데이터셋 실험 (2026-09-16 착수) ═══════════════════════════════
# 왜: perf 표가 알고리즘을 offline env(mujoco-py), 오라클을 mjx 에서 재고 있었고 그
# sim-to-sim 격차가 **순위를 뒤집었다** (offline: BC-corner 13.99 > AETDICE 13.64 /
# mjx: AETDICE 8.35 > BC-corner 5.44; FW 오라클 seed701 은 mjx 13.71 → offline 10.80).
# 모든 정책이 자기가 학습된 시뮬레이터에서만 이긴다. 데이터를 mjx 쪽으로 옮겨 데이터·
# 알고리즘·오라클을 한 물리에 모은다.
#
# 데이터: preference 전문가 13 점(w0 ∈ [0.24, 0.70]) × mjx PPO 50M step 을 확률적으로
#   롤아웃해 만든 10,000 궤적 (experiments/oracle/build_mjx_dataset.py).
#   파일럿 검증: 방향 오차 3° 이내로 원본 프론티어가 재현된다.
# dataset="mjx" → job_runner.dataset_path 가 MO-Ant-v2_10000_mjx_uniform.pkl 로 푼다.
ANT_MJX_AETDICE = build_sweep(
    "ant", "mjx", "aetdice", learner_type="aet", g_type="piecewise_log", f_type="exponential_F",
    prefs=[None], seeds=[1, 2, 3], beta=["1e-6"],
)

# 바닥선 — 같은 데이터, 같은 정책 클래스(augmented MDP). perf 의 분모.
ANT_MJX_BC_RACC = build_sweep(
    "ant", "mjx", "bc_racc", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
)

# corner 바닥선 — 목적별 상위 5% 합집합. AET 의 mixture 구조를 정조준한다.


# ══ amateur 데이터셋 3 종 (2026-09-17 사용자 지시) ═══════════════════════════
# 왜: expert 에서 본 구도(AETDICE vs 같은 정책 클래스의 BC 바닥선)가 **데이터 품질이
#   낮을 때도** 유지되는지 본다. expert 에서는 corner top-10% BC 가 오라클보다도 높은
#   AET 를 냈다 — 데이터 안에 이미 좋은 mixture 가 있었기 때문이다. amateur 는 그
#   mixture 상한 자체가 낮으므로, 학습된 w 가 버는 몫이 드러난다면 여기서 드러난다.
# ★ 물리: amateur 는 원본 d4morl 수집물이라 offline env(mujoco-py) 소산이다. mjx-native
#   amateur 판이 아직 없으므로 이 3 종은 **offline env 트랙**에서 돈다 (데이터와 eval 이
#   같은 물리). mjx-native 와 숫자를 직접 나란히 놓지 말 것 — sim-to-sim 격차가 순위를
#   뒤집는다는 게 이 프로젝트의 확립된 사실이다.
ANT_AMATEUR_AETDICE = build_sweep(
    "ant", "amateur", "aetdice", learner_type="aet", g_type="piecewise_log",
    f_type="exponential_F", prefs=[None], seeds=[1, 2, 3, 4, 5], beta=["1e-6"],
)

# 바닥선 1 — 필터 없음. 같은 augmented MDP, w ≡ 1 (perf 의 분모).
ANT_AMATEUR_BC_RACC = build_sweep(
    "ant", "amateur", "bc_racc", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3, 4, 5], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
)

# ── 에피소드 단위 잠재 z 조건부 단일 정책 (2026-09-18) ─────────────────────────
# 왜: 커밋 장치가 필요한데 남은 후보가 z 뿐이다.
#   · racc — 개시가 구조적으로 불가능(racc0=0)하고, 인코딩을 고쳐 신호를 2547 배 살려도
#     정책이 안 쓴다 (실험 G: 행동 반응 0.001, 성능 12.5207 로 오히려 하락).
#   · 혼합 성분 — 잘 분리돼 있으나(성분간/σ=15.25) head(t) 사이 정체성이 없다 (실험 E:
#     자기 슬롯이 최근접인 비율 20~80%) → 성분 고정 롤아웃은 손해 (11.4137).
#   · z 는 **전 구간 상수**라 정체성 문제가 없고, 학습 시 **관측 라벨**이라 posterior
#     collapse 가 원리적으로 불가능하다.
# 목표: 전문가 2 개 에피소드 혼합(13.4680)을 **단일 망**으로 실현. 상한 q=1% 13.7723.
# 구성: traj_filter = union keep-list(w 상위 k%), z_keeplist = 그 mode0 분할 → z=1.
#   조건부 입력은 [racc(2), z(1)] 로 racc_dim 이 자동 +1 된다 (config.racc_dim 파생).
# 평가: z=0/z=1 로 각각 100ep 굴린 뒤 p 로 해석적 혼합 (mix_specialists 와 같은 규약) —
#   전문가 2 개 결과와 직접 비교 가능하다.
# headline 수치이므로 프로젝트 표준인 5 seed 로 채운다 (2 seed 에서 13.4826/13.4221 로
# seed 폭이 0.06 이라 단일 seed 로 단정할 수 없다).
ANT_AMATEUR_BC_ZCOND1 = build_sweep(
    "ant", "amateur", "bc_zcond1", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3, 4, 5], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_99pct.npy"],
    z_keeplists=("experiments/moant/keeplists_amateur_s1/keep_dropbottom_99pct_mode0.npy",),
)

ANT_AMATEUR_BC_ZCOND5 = build_sweep(
    "ant", "amateur", "bc_zcond5", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_95pct.npy"],
    z_keeplists=("experiments/moant/keeplists_amateur_s1/keep_dropbottom_95pct_mode0.npy",),
)

# ── racc 조건부 인코딩 수정 (2026-09-18) ──────────────────────────────────────
# 왜: 정책이 racc 를 전혀 못 쓰고 있었다. 원인은 정책이 아니라 **인코딩**이다 —
#   버퍼가 F̃=exp(R/50)−1 을 저장하고 RaccEncoder 가 그 위에서 전역 z-score 를 하는데,
#   통계(mean [81.0,99.2] std [166.9,193.8])를 에피소드 후반의 거대한 값이 지배해
#   초반 t 의 모드 차이가 z-score 후 5e-4~9e-3 으로 눌린다. 실측: 같은 상태에서 racc 를
#   mode0↔mode1 로 바꿔도 t≤250 에서 행동 변화 **0.0000** (데이터 모드 행동차 0.32~1.55).
# 수정: --racc_encoding rate = R/(t+1) (스텝당 평균 보상). 학습 전 수치 검증에서
#   z-diff 가 t=25 에서 0.000482 → 1.2281 (2547 배), t=100 에서 0.0086 → 2.2204 로 살아났다.
# ★ 조건부 입력만 바뀐다 — per-step 스칼라 보상(텔레스코핑 F̃ 차)은 목적함수라 그대로다.
# ※ 단, MO-Ant 은 두 목적(vx, vy)이 동시 획득이라 racc 가 원래 크게 필요 없을 수 있다
#   (2026-09-06 확립). 고쳐도 같으면 그것대로 결론이고, 정책 추출 쪽으로 넘어간다.
ANT_AMATEUR_AETDICE_RACCRATE = build_sweep(
    "ant", "amateur", "aetdice_raccrate", learner_type="aet", g_type="piecewise_log",
    f_type="exponential_F", prefs=[None], seeds=[1, 2, 3], beta=["1e-6"],
    racc_encodings=("rate",),
)


# ── (a) w-하드필터 BC — "추출이 버리는가"의 직접 검정 ────────────────────────
# 왜 (2026-09-17 진단): 같은 체크포인트에서 w-재가중 데이터 분포의 AET 는 13.49 인데
#   (데이터 12.96, corner 부분집합 13.60) WBC 로 뽑은 정책 롤아웃은 12.965 였다.
#   w 는 corner 궤적을 lift ×5.3 으로 골라내고 특화도와 ρ=+0.66 으로 상관되는데
#   (총수익과는 ρ=−0.30), 추출이 그 이득의 99% 를 버린다.
# 그래서 **soft w 가중을 버리고 하드 선별 + 평범한 BC** 로 같은 정보를 넣어 본다.
#   keep-list 는 src.ops.filter_trajectories_by_w 가 학습된 w 로 만든 것이다
#   (dropbottom_90pct = w 상위 10% 만 남김).
# 이게 13.3 근처를 내면 원인은 **추출 규칙**(soft 가중이 조건부 사상을 뭉갠다)이고,
# 12.96 에 머물면 정책 클래스·조건부 쪽 문제다.
ANT_AMATEUR_BC_WTOP10 = build_sweep(
    "ant", "amateur", "bc_wtop10", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_90pct.npy"],
)

ANT_AMATEUR_BC_WTOP5 = build_sweep(
    "ant", "amateur", "bc_wtop5", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_95pct.npy"],
)

ANT_AMATEUR_BC_WTOP15 = build_sweep(
    "ant", "amateur", "bc_wtop15", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_85pct.npy"],
)

ANT_AMATEUR_BC_WTOP33 = build_sweep(
    "ant", "amateur", "bc_wtop33", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_67pct.npy"],
)

# 필터 비율 최적화 (2026-09-18): β-ESS 스캔에서 신호가 있는 β 영역의 w 는 준-지시함수이고
# e>0 인 transition 이 33% 다. 상위 10%(13.3192) > 25%(13.2045) 였으므로 5/15/33% 를 더 본다.
ANT_AMATEUR_BC_WTOP25 = build_sweep(
    "ant", "amateur", "bc_wtop25", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2, 3], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_75pct.npy"],
)


# ── 모드별 전문가 + 에피소드 단위 커밋 (2026-09-18) ─────────────────────────
# 왜: 추출 실험이 두 사실로 좁혀졌다 —
#   · 스텝 단위 행동 재선택(argmax / ∝w 재표집)은 최선이 "변화 없음"
#   · 하드 선별 + BC 는 +0.354, 그 상한은 필터 비율이 정한다
# AET 는 볼록 F 라 에피소드 **간** 분산을 보상하고(convex-MDP: Markov 정책 비최적),
# mixture_ceiling 실측 상한은 모드 q 별로 q=10% 13.4974 / 5% 13.6029 / 1% 13.7723,
# 최적 혼합은 p*≈0.5 (표면 평평). 합집합 필터 BC 는 t=0 에 정보 없는 racc 로 모드를
# **추론**해야 하는데, 모드별 전문가 둘을 에피소드 단위로 섞으면 **구성상 커밋**한다.
# keep-list 는 학습된 w 상위 k% 를 sign(R0−R1) 로 이분할한 것이다 (DICE 유래 — w 선별이
# 오라클 corner 모드를 거의 정확히 복원함을 확인: w-top5% [368.6,250.5]/[226.1,372.2]
# vs 오라클 [368.5,255.0]/[231.3,373.1]).
# ★ 평가는 별도 롤아웃이 필요 없다 — 에피소드마다 한 정책을 끝까지 쓰므로 두 전문가의
#   100ep return 분포를 해석적으로 혼합하면 된다 (experiments/moant/mixture_ceiling 수식).
ANT_AMATEUR_BC_SPEC1_M0 = build_sweep(
    "ant", "amateur", "bc_spec1_m0", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_99pct_mode0.npy"],
)

ANT_AMATEUR_BC_SPEC1_M1 = build_sweep(
    "ant", "amateur", "bc_spec1_m1", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_99pct_mode1.npy"],
)

ANT_AMATEUR_BC_SPEC5_M0 = build_sweep(
    "ant", "amateur", "bc_spec5_m0", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_95pct_mode0.npy"],
)

ANT_AMATEUR_BC_SPEC5_M1 = build_sweep(
    "ant", "amateur", "bc_spec5_m1", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_95pct_mode1.npy"],
)

ANT_AMATEUR_BC_SPEC10_M0 = build_sweep(
    "ant", "amateur", "bc_spec10_m0", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_90pct_mode0.npy"],
)

ANT_AMATEUR_BC_SPEC10_M1 = build_sweep(
    "ant", "amateur", "bc_spec10_m1", learner_type="bc", g_type="linear", f_type="linear",
    prefs=[[1.0, 0.0]], seeds=[1, 2], bc_conditionings=["racc"],
    cd_filter_rhos=[0.5], cd_filter_k_pcts=[1.0], beta=["1e-5"],
    traj_filter_paths=["experiments/moant/keeplists_amateur_s1/keep_dropbottom_90pct_mode1.npy"],
)

# ── AETDICE 하이퍼파라미터 스윕 (β × musf) ──────────────────────────────────
# 왜 (사용자 지시 2026-09-17 "aetdice 가 할 수 있는 최선을 다하는 거야. 단순 seed 만 많이
#   돌리지 말고"): 지금까지 β 는 1e-6, musf 는 **전역 상수 1e4** 로 고정이었다. 둘 다 DICE
#   dual 의 수치 스케일을 정하는 축이라 (β 는 f-div 정규화 세기, musf 는 conjugate_G 입력
#   스케일) 고정한 채 seed 만 늘리는 건 탐색이 아니다.
# 규모: β 6 × musf 5 = 30 조합, **seed 1 고정**. 전 조합을 2M 까지 돌리는 건 60 GPU-시간이라
#   먼저 짧게 줄세운다 (AET_TOTAL_TRAIN_STEPS=300000 으로 실행 — 곡선상 300k 면 상승분의
#   대부분이 들어온다). ★ 다만 300k 순위가 2M 순위와 같다는 보장은 없으므로, 상위 1 개가
#   아니라 상위 몇 개를 길게 다시 돌려 확인한다.
ANT_AMATEUR_AETDICE_HP = build_sweep(
    "ant", "amateur", "aetdice_hp", learner_type="aet", g_type="piecewise_log",
    f_type="exponential_F", prefs=[None], seeds=[1],
    beta=["1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8"],
    mu_scaling_factors=(1e2, 1e3, 1e4, 1e5, 1e6),
)


# 바닥선 2 — corner top-10% (목적별 상위 5% 합집합). 이진 w 로 mixture 상한을 정조준.


# ── [격리 트랙] 학습용 F 스케일 사다리 (tempered surrogate) ────────────────────
# 왜 (사용자 지시 2026-09-19 "log 가 들어가니까 스케일이 굉장히 중요하다 … 앞서 했던 것과
#   isolated 된 폴더에 새로 표를 따로 작성하고 실험해"):
#   Buffer 가 만드는 학습용 per-step 보상은 r̃_t = F̃(R_{t+1}) − F̃(R_t) 다. 현재
#   F = exp(x/50) 이면 r̃_t ∝ exp(R_t/50) 라 **에피소드 후반으로 갈수록 지수적으로 커져**
#   앞 250 스텝이 전체 신호의 4.1% 밖에 안 된다 (experiments/scale/reward_time_profile.py).
#   커밋(어느 corner 로 갈지)은 바로 그 앞 구간에서 일어나야 하는데 거기에 그래디언트가 없다.
#   S 를 키우면 exp(x/S) → 1 + x/S 로 평평해진다: S=100 앞50%=16.9%, S=200 30.8%, S=500 41.5%.
#   즉 이 축은 **AETDICE(S=50) → SER-DICE(F-linear)** 를 연속으로 잇는 tempered surrogate 다.
#
# ★ 평가는 AET(S=50) **고정**이다 — 학습용 F 만 바꾼다. src/eval/evaluate.py 는 config 의
#   exp_f_scale 로 채점하므로 npy 의 score_* 열은 스케일마다 다르다. 이 트랙의 채점은 반드시
#   final_returns_normalized.npy 재채점(experiments/scale/*)으로 한다.
#
# β 는 보상 스케일에 **비례해 옮긴다** — e = r·μ+ν'−ν 가 보상과 같이 작아지므로 β 를 그대로
#   두면 w = relu(1+e/β) 가 전부 1 로 눌려 BC 로 붕괴한다. 비례 β 와 그 10 배를 함께 건다
#   (새 스케일에서 β 최적점을 모르므로 bracket).
_SCALE_LADDER = {          # exp_f_scale → (비례 β, 10×β)   [mean|r̃| 비 × 1e-6, 소수 2자리]
    100.0: ["4.28e-8", "4.28e-7"],
    200.0: ["7.50e-9", "7.50e-8"],
    500.0: ["1.74e-9", "1.74e-8"],
}
ANT_AMATEUR_FSCALE_LADDER = [
    build_sweep(
        "ant", "amateur", f"fscale{int(_s)}", learner_type="aet", g_type="piecewise_log",
        f_type="exponential_F", prefs=[None], seeds=[1, 2],
        beta=_betas, exp_f_scales=(_s,),
    )
    for _s, _betas in _SCALE_LADDER.items()
]
