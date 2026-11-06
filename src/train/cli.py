"""CLI argument parsing + post-parse validation for the train entrypoint.

Public:
    parse_and_validate() → SimpleNamespace `config`

Everything that touches argparse lives here; `main.py` consumes only the
validated `config`. Validation is in a separate pass (after `parse_known_args`)
so we can fail-fast on multi-arg constraints (e.g. `--power_p` is required iff
`--utility_F_type == 'power_F'`) and emit user-friendly `parser.error(...)`
messages instead of cryptic asserts deeper in the stack.
"""
import argparse
import os

from src.nets.policy import EXTRACTION_MODES
from types import SimpleNamespace

from src.learners import LEARNER_MODULES
from src.data.reward_transform import CD_SAFETY_MODES, CD_SAFETY_THRESHOLDED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1e-5, help="beta hyperparameter (canonical default 1e-5)")
    parser.add_argument(
        "--beta_schedule", type=str, default="constant",
        choices=["constant", "linear", "exp", "cosine"],
        help="β decay schedule. constant: use --beta unchanged. Otherwise "
             "--beta_init AND --beta_final are required; β anneals from init to "
             "final over (total_train_steps - beta_warmup_steps).",
    )
    parser.add_argument(
        "--beta_init", type=float, default=None,
        help="Initial β for non-constant schedules. REQUIRED when --beta_schedule != constant.",
    )
    parser.add_argument(
        "--beta_final", type=float, default=None,
        help="Final β for non-constant schedules. REQUIRED when --beta_schedule != constant.",
    )
    parser.add_argument(
        "--beta_warmup_steps", type=int, default=0,
        help="Hold β at --beta_init for this many steps before decay starts. Default 0.",
    )
    parser.add_argument(
        "--divergence", type=str, default="CHI",
        choices=["KL", "CHI", "SOFT_CHI", "DUAL_DICE", "CLIP_CHI", "SOFT_CLIP_CHI"],
        help="f-divergence for the DICE Bellman dual (FDivergence enum names). "
             "CLIP_CHI = χ² with the importance ratio w clipped at --w_max; "
             "mitigates the small-β saturation where w ≈ 1 + e/β explodes and "
             "β·f(w) loses meaning. Other choices match the original FDivergence.",
    )
    parser.add_argument(
        "--w_max", type=float, default=float("inf"),
        help="Max importance-ratio for ClipChi (ignored otherwise). Smaller = "
             "more aggressive clipping → effectively a trust region on w. "
             "Sensible values: 10/100/1000. Default ∞ → no clipping (= plain χ²).",
    )
    parser.add_argument(
        "--standardize_e", action="store_true", default=False,
        help="Standardize Bellman residual e by its batch-wise std (stop-gradient) "
             "before computing w = (f')⁻¹(e/β) and the β·f(w) penalty. Decouples β "
             "from absolute reward scale: at small β with F-exp shaping, raw |e| can "
             "span multi-OOM and saturate the dual; standardizing makes β a unitless "
             "constraint budget on σ_e. Targets the small-β failure mode complementary "
             "to CLIP_CHI / SOFT_CHI (which fix tail saturation).",
    )
    parser.add_argument(
        "--nu_output_transform", type=str, default="none",
        choices=["none", "symexp"],
        help="Reparameterize the ν critic head. 'none' (default — back-compat): "
             "raw output is ν. 'symexp': output = symexp(raw); the network "
             "effectively learns symlog(true_ν). Helps when ν has wide dynamic "
             "range and small β would otherwise make e/β explode (DreamerV3-style). "
             "Affects Critic and FairDICECritic; QCritic (IQL) is untouched.",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the network")
    parser.add_argument("--layer_norm", type=bool, default=True, help="Use layer normalization")
    parser.add_argument("--n_mixture", type=int, default=5,
                        help="Number of Gaussian-mixture components in the AET policy. "
                             "The action distribution is multimodal (e.g. heading toward different "
                             "objectives); with many objectives more components may be needed "
                             "(default 5 for d4morl 2-obj; MO-PointMaze 10-obj may want ~10-15).")
    parser.add_argument("--extraction", type=str, default="original",
                        choices=list(EXTRACTION_MODES),
                        help="Policy-extraction branch — ONE axis binding the policy architecture "
                             "to the WBC scoring rule (see src/nets/policy.py). "
                             "original (default, the paper's Table 4 setting): multi-head "
                             "ContinuousPolicy (time = output slot) scored with all-H WBC (the "
                             "action is cloned onto every timestep head). "
                             "timecond: TimeConditionedPolicy (raw remaining-time fed "
                             "into the backbone, no multi-head) scored only at each transition's "
                             "own timestep. Keeping these as one axis prevents the unintended "
                             "third combination that arose when they were separate flags.")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--policy_lr", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--mu_lr", type=float, default=3e-4, help="Mu learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument(
        "--max_seq_len", type=int, default=500, help="Max episode length (Walker2d: 500)"
    )
    parser.add_argument(
        "--total_train_steps", type=int, default=int(1e6), help="Total training steps"
    )
    parser.add_argument("--log_interval", type=int, default=int(1e4), help="Log interval")
    parser.add_argument(
        "--chunk_size", type=int, default=1000,
        help="Number of train steps fused into one jax.lax.scan call. Larger = less Python "
             "dispatch overhead (better GPU util) but more device memory per chunk. Must "
             "divide log_interval. Default 1000 → ~65MB on device for walker2d at batch 256.",
    )
    parser.add_argument("--eval_episodes", type=int, default=30, help="Evaluation episodes (mid-training).")
    parser.add_argument(
        "--eval_batch", type=int, default=0,
        help="Eval 롤아웃을 몇 에피소드씩 묶어 정책 forward 를 한 번에 돌릴지. "
             "0(기본)=num_episodes 전부 한 배치. 1 이면 예전 순차 경로와 같은 계산량이 된다. "
             "근거(2026-09-10 실측): eval 비용의 94%%가 정책 forward 였고(1678 vs 물리 111 ms/ep), "
             "묶으면 100 ep 179초 → 19.8초(9x). ★ 배치 크기를 바꾸면 표본(난수)이 달라진다.")
    parser.add_argument(
        "--final_eval_episodes", type=int, default=100,
        help="Evaluation episodes used at the FINAL training iteration. Sets the size of "
             "eval/final_returns.npy and the wandb-logged final score. Larger gives a less "
             "noisy final number; mid-training evals still use --eval_episodes.",
    )
    parser.add_argument(
        "--eval_every_n_logs", type=int, default=1,
        help="Run eval every N log windows. 1 = every log (default). Final log window "
             "always evals regardless (final_returns.npy depends on it).",
    )
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb")
    parser.add_argument("--save_path", type=str, default="./results2", help="Path to save")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tag", type=str, default="aet", help="Prefix for auto-generated run name.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name. If unset, falls back to --tag.")
    parser.add_argument("--run_name", type=str, default=None, help="Explicit run name for wandb and save_dir (overrides auto-generated)")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./environment/dataset/MO-Walker2d-v2/MO-Walker2d-v2_50000_expert_narrow.pkl",
    )
    parser.add_argument(
        "--utility_G_type",
        type=str,
        default="linear",
        choices=["linear", "piecewise_log", "weighted_log", "isoelastic", "isoelastic_ce"],
        help="Outer utility G (src/utility.py G registry). linear: mu fixed to pref_weights. "
             "piecewise_log: mu learned (softplus). 2026-10-25 (docs/aet_objectives_worksheet.md): "
             "weighted_log = Σ w log v; isoelastic = Σ w v^p/p; isoelastic_ce = Σ w h_p(θ log(1+v)) "
             "(certainty-equivalent of F=exponential_F, θ=--exp_f_scale). w = --pref_weights (default 1), "
             "p = --g_iso_p.",
    )
    parser.add_argument(
        "--utility_F_type",
        type=str,
        default="power_F",
        choices=["power_F", "exponential_F", "piecewise_log", "piecewise_log_threshold",
                 "linear", "cobbdouglas", "sigmoid"],
        help="Inner utility F, applied per-step/per-trajectory via Buffer. "
             "power_F uses --power_p; exponential_F uses exp(x / --exp_f_scale); "
             "piecewise_log is log(x) for x>=1, quadratic for x<1; "
             "piecewise_log_threshold is the same shape but shifted so the kink sits "
             "at x = --reward_threshold (above: saturating log; below: quadratic strong "
             "push toward T). Use this as a 'satisfice-then-saturate' shaping. "
             "linear is identity (F̃ telescopes back to raw reward — used to train "
             "OptiDICE / FairDICE-paper baselines with no F shaping). "
             "cobbdouglas is Reward^ρ/Cost^(1−ρ) where the reshape already sets "
             "cost_t = 1 + ‖a_t‖². REQUIRES a CD reshape "
             "(walker2dcd / half_cheetahcd / walker2d_cd_safe, a reward/cost layout) "
             "and --pref_weights=[1, 0]. "
             "sigmoid = σ((R_d − τ_d)/s) per objective (goal attainment; --f_tau, --f_sigmoid_scale).",
    )
    parser.add_argument("--f_tau", type=float, nargs="+", default=None,
                        help="F=sigmoid: per-objective goal τ_d on the normalized cumulative return.")
    parser.add_argument("--f_sigmoid_scale", type=float, default=None,
                        help="F=sigmoid: width s of σ((R − τ)/s).")
    parser.add_argument("--g_iso_p", type=float, default=None,
                        help="G=isoelastic / isoelastic_ce: exponent p < 1 (0 = log/Nash, <0 more egalitarian).")
    parser.add_argument(
        "--power_p",
        type=float,
        default=None,  # required iff --utility_F_type == 'power_F' (validated below)
        help="Exponent p for power_F utility (F(x) = x^p). p>1 convex, 0<p<1 concave. "
             "REQUIRED when --utility_F_type=power_F; REJECTED for other F types.",
    )
    parser.add_argument(
        "--exp_f_scale",
        type=float,
        default=50.0,
        help="Divisor for exponential_F: F(x) = exp(x / exp_f_scale). Must match the env's "
             "racc magnitude (d4morl racc~300-400 → 50 default; MO-PointMaze racc~0-170 → ~10). "
             "Too large → exp is ~linear over the racc range (F-convexity inactive). "
             "Only affects utility_F_type=exponential_F.",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        nargs="+",
        default=None,
        help="(safe-CD linear 는 safety 채널마다 하나 — walker2d_cd_safe_pv 는 'S0문턱 S1문턱', 2026-11-05) "
             "Threshold T for piecewise_log_threshold utility. F̃(R) is shaped so "
             "the kink in log-space sits at R = T: above T it's the saturating "
             "log(R-T+1); below T it's a quadratic that grows fast (gradient ≈ T+1−R) "
             "to push the policy upward. REQUIRED when --utility_F_type=piecewise_log_threshold; "
             "REJECTED for other F types. Pick T near the dataset's reward-channel distribution "
             "(e.g., the 30th–50th percentile of trajectory returns you want to exceed).",
    )
    parser.add_argument(
        "--cobbdouglas_amp",
        type=float,
        default=1.0,
        help="Multiplicative amplification on dim 0 of CobbDouglas (Reward^ρ / Cost^(1-ρ) * amp). "
             "Only used when --utility_F_type=cobbdouglas. Default 1.0 (back-compat). "
             "Under G=linear this is mathematically equivalent to dividing β by amp; "
             "under G=piecewise_log it has no effect (log absorbs the constant).",
    )
    parser.add_argument(
        "--cobbdouglas_rho",
        type=float,
        default=0.5,
        help="Elasticity rho ∈ [0, 1] for cobbdouglas utility. "
             "F(R) = R_0^rho / R_1^(1 - rho),  R_1 = Σ_t (1 + ‖a_t‖²). "
             "rho=1 -> pure reward, rho=0 -> pure cost-minimization, rho=0.5 -> symmetric (default). "
             "Only used when --utility_F_type=cobbdouglas.",
    )
    parser.add_argument(
        "--cobbdouglas_form", type=str, default="total", choices=["total", "rate"],
        help="CD 를 F 에 넣는 형태 (2026-10-24 exp A, docs/CD_redesign_worksheet.md §3). "
             "total = CD(R) (기존). rate = n·CD(R·T/n) — 지금까지의 스텝당 평균 효율 × 경과 스텝. "
             "고정 길이면 같은 최적 정책이고 telescoping 보상이 시간에 고르게 퍼진다. "
             "조기 종료 에피소드는 남은 스텝의 효율을 잃는다. rate 는 --racc_encoding rate 필수.",
    )
    parser.add_argument(
        "--cobbdouglas_safety", type=str, default="plog", choices=list(CD_SAFETY_MODES),
        help="safe-CD 결합 방식 (2026-11-03, docs/safe_cd_definitions.md §4). plog = dim1 plog(S−th) (기존). "
             "linear = dim1 ΣS − th (논문 식, G=piecewise_log 와). product = F = CD_rate·S̄^κ (G=linear). "
             "inlog = F = plog([T·CD_rate, ΣS_k − th_k]) 궤적마다 log (G=linear, 2026-11-06 D6.28). "
             "violog = F = [plog(T·CD_rate), −log(1 + 위반량_k)] 문턱 없음 (G=linear, D6.30). "
             "linear·product 는 --cobbdouglas_form rate 와 3 채널 safe-CD reshape 가 필요하다.",
    )
    parser.add_argument(
        "--cobbdouglas_kappa", type=float, default=1.0,
        help="safety=product 의 safety 지수 κ (F = CD_rate·S̄^κ).",
    )
    parser.add_argument(
        "--cd_filter_kind", type=str, default="cd", choices=["cd"],
        help="상위 K%% 궤적 필터의 기준. 현재 \"cd\"(CD 점수 Reward^ρ/Cost^(1−ρ) 상위 K%%) "
             "하나뿐이다. \"corner\"(목적별 상위 (K/D)%% 합집합)는 정당화 불가한 post-hoc "
             "휴리스틱이라 2026-09-20 제거했다 — reward_transform.filter_top_k 주석 참조."
             "합집합 — AET 의 E[exp(·)] 가 볼록이라 최적해가 mixture 이고, per-traj 점수로 "
             "자르면 균형 궤적만 남아 극단이 깎이기 때문이다. cd=CD 점수 상위 K%% "
             "((reward,cost) 레이아웃 전용). k_pct=1.0 이면 어느 쪽이든 no-op 이다.")
    parser.add_argument(
        "--z_keeplist", type=str, default=None, metavar="NPY",
        help="에피소드 단위 잠재 z 의 **관측 라벨**. 이 keep-list 에 든 궤적은 z=1, 나머지는 0 "
             "이고, z 가 조건부 입력(racc 벡터 뒤)에 한 열로 붙는다. --traj_filter_path 와 "
             "함께 쓴다 (보통 union keep-list + 그 mode0 분할). 왜: AET 는 볼록 F 라 최적해가 "
             "혼합인데 racc 는 개시가 불가능(racc0=0)하고 정책이 쓰지도 않으며(실험 G), 혼합 "
             "성분은 head(t) 사이 정체성이 없다(실험 E). z 는 전 구간 상수이고 **관측값**이라 "
             "posterior collapse 가 원리적으로 불가능하다.")
    parser.add_argument(
        "--racc_encoding", type=str, default="ftilde", choices=["ftilde", "rate"],
        help="ν/μ/π 가 받는 racc **조건부 입력**의 형태. ftilde = F̃(R)=exp(R/50)−1 (기본, "
             "기존 동작). rate = R/(t+1) = 스텝당 평균 보상. 왜 필요한가: ftilde 는 전역 "
             "z-score 와 결합해 초반 t 의 모드 차이를 5e-4 로 압축한다 (2026-09-18 실측: "
             "racc 를 mode0↔mode1 로 바꿔도 t≤250 에서 행동 변화 0.0000). per-step 보상은 "
             "이 플래그와 무관하게 F̃ 텔레스코핑 그대로다.")
    parser.add_argument(
        "--bc_conditioning", type=str, default="state", choices=["state", "racc"],
        help="learner_type=bc 전용. state=π(a|s) (표준 BC, base MDP 의 Markov 정책). "
             "racc=π(a|s,racc) — **augmented MDP**, AETDICE 와 같은 정책 클래스·같은 all-H "
             "채점에 가중치만 w≡1. 목적이 return 에 비선형(convex MDP)이라 Markov 정책은 "
             "최적이 될 수 없으므로, perf 의 분모를 알고리즘과 같은 클래스에 두려면 racc 를 "
             "쓴다. 기본 state 는 2026-09-16 이전 런의 재현용이다.")
    parser.add_argument(
        "--mu_scaling_factor",
        type=float,
        default=1e4,
        help="Scale factor for conjugate_G (canonical default 1e4 — matches the paper's Table 4/5/6 runs).",
    )
    parser.add_argument( # for simplicity, either [1,0], [0,1] or [1,1]
        "--pref_weights",
        type=float,
        nargs="+",
        default=None,  # required iff --utility_G_type == 'linear' (validated below)
        help="Preference weights vector (length must equal reward_dim). linear G: fixed mu (REQUIRED). "
             "weighted_log / isoelastic / isoelastic_ce: weights inside G (optional, default 1; mu is learnable). "
             "REJECTED for piecewise_log.",
    )
    parser.add_argument(
        "--reward_scaling_factor",
        type=float,
        nargs="+",
        required=True,
        help="Per-objective reward scaling factor (length must equal reward_dim). "
             "[1,1] neutral, [2,1] obj1-favouring. Applied in Option A normalization inside load_d4morl_dataset.",
    )
    # ── IQL-only hyperparameters (consumed by IQL_learner; ignored by AET / FairDICE) ──
    parser.add_argument(
        "--expectile", type=float, default=0.7,
        help="IQL τ-expectile for V-loss. τ ∈ (0.5, 1); higher → V tracks upper tail of Q. "
             "Typical: 0.7 (locomotion), 0.9 (sparse-reward).",
    )
    parser.add_argument(
        "--temperature", type=float, default=3.0,
        help="IQL AWR temperature. Higher → sharper concentration on high-advantage actions. "
             "Typical: 3.0 (D4RL locomotion), 10.0 (antmaze).",
    )
    parser.add_argument(
        "--tau", type=float, default=0.005,
        help="IQL Polyak target-Q update rate. Typical: 0.005.",
    )
    parser.add_argument(
        "--awr_exp_clip", type=float, default=100.0,
        help="Upper clip on exp(temperature · advantage) to keep AWR weights bounded. "
             "Standard offline-RL practice.",
    )
    parser.add_argument("--env_name", type=str, default="walker2d", help="Environment name")
    parser.add_argument(
        "--num_objectives", type=int, default=2, help="Number of objectives (Walker2d: 2)"
    )
    parser.add_argument(
        "--learner_type",
        type=str,
        default="aet",
        choices=list(LEARNER_MODULES),
        help="Which learner to use. aet: state+racc-conditioned DICE dual (AET_learner). "
             "fairdice: state-only DICE dual (FairDICE_learner); racc ignored. "
             "iql: state+racc ESR baseline (Q + V + AWR; G-linear only, μ fixed to pref_weights). "
             "bc: filtered behavior cloning (state-only MLE on top-K%% CD-scoring trajectories; "
             "use with --cd_filter_rho and --cd_filter_k_pct).",
    )
    parser.add_argument(
        "--cd_filter_rho", type=float, default=None,
        help="Cobb-Douglas elasticity used to score per-trajectory CD for top-K%% filtering. "
             "REQUIRED when --learner_type=bc. OPTIONAL for other learners (aet/iql/fairdice) — "
             "if set, the buffer trains only on top-K%% trajectories by CD score under this rho. "
             "Pair with --cd_filter_k_pct.",
    )
    parser.add_argument(
        "--cd_filter_k_pct", type=float, default=None,
        help="Top fraction of trajectories to keep, by CD score (e.g. 0.1 = top 10%%). "
             "REQUIRED when --learner_type=bc; OPTIONAL for other learners.",
    )
    parser.add_argument(
        "--reward_reshape",
        type=str,
        default="none",
        choices=["none", "antcd", "walker2dcd",
                 "half_cheetahcd", "walker2d_cd_safe", "walker2d_comp",
                 "walker2d_cd_safe_vel", "walker2d_cd_safe_margin", "walker2d_cd_safe_sat",
                 "walker2d_cd_safe_pv", "walker2d_cd_safe_pok", "walker2d_cd_safe_pokv",
                 "walker2d_cd_safe_ha", "walker2d_cd_safe_h",
                 "walker2d_cd_safe_hm"],
        help="Optional per-step reward reshape applied at buffer-load time AND during "
             "rollout/eval. 'none': use the dataset's stored reward layout. "
             "'walker2dcd' / 'half_cheetahcd': cost-recovery reshape  cost := 5 − obj[1] "
             "for MO-Walker2d / MO-HalfCheetah respectively (obj[0] left unchanged); "
             "yields the (forward+alive, ‖a‖²) layout CobbDouglasUtility expects. "
             "When != 'none', config.reward_dim is overridden and --pref_weights / "
             "--reward_scaling_factor must match the reshape's output dim.",
    )
    parser.add_argument(
        "--wbc_weight", type=str, default="raw",
        choices=["raw", "norm", "clip99", "rank", "perstep", "binary", "timestrat", "traj", "action"],
        help="WBC 가중치 정형. raw=논문 그대로 · norm=배치평균 정규화 · clip99=p99 절단 후 "
             "정규화 · rank=양수의 순위로 치환 후 정규화(크기 버리고 순서만). 중요도 가중의 "
             "표준 분산 감소 — 사전 정당화된다. timestrat=자기 시점 w 를 시점 층(25 스텝) 안에서 "
             "평균 1 (per-timestep 채점 추출 전용, 2026-10-26). traj=궤적 평균 w 를 궤적 전체에 · "
             "action=w/ŵ(s,racc,t) 행동 비율만 (둘 다 --critic_from 필요, src/learners/wbc_state_weight.py).")
    parser.add_argument(
        "--nu_grad", type=str, default="true", choices=["true", "semi", "orth"],
        help="ν 기울기 모드 (2026-10-27, ODICE ablation). true=지금 그대로(true-gradient) · semi=ν(s′) 쪽 "
             "backward 기울기 제거 · orth=backward 를 forward 법평면에 투영해 --orth_eta 배로 더함 "
             "(Mao et al. ICLR 2024 Algorithm 1).")
    parser.add_argument("--reward_norm_scale", type=float, default=None,
                        help="주면 per-step 보상 정규화를 데이터셋 min–max 대신 raw/scale (shift 없음) 로 한다. "
                             "PEDA(D4MORL) 공식 코드의 실제 학습 설정은 1000 (2026-10-27 ablation). "
                             "CD reshape 와는 함께 쓸 수 없다 (CD 는 자체 scale-only).")
    parser.add_argument("--nu_target_tau", type=float, default=0.0,
                        help="ν target network EMA 계수 (ODICE 양방향 target, 원문 5e-3). 0 = 끔 (지금까지 동작).")
    parser.add_argument("--nu_linear_term", type=str, default="init", choices=["init", "data"],
                        help="쌍대 선형 항을 거는 상태. init = AETDICE 원래 (초기 상태). data = 원문 ODICE 처럼 "
                             "데이터의 모든 상태 (자기 시점 head) — 목적의 d0 가 d_D 로 바뀐다.")
    parser.add_argument("--orth_eta", type=float, default=1.0,
                        help="--nu_grad orth 의 투영 backward 기울기 계수 η (원문 격자 0.2–1.0).")
    parser.add_argument(
        "--critic_from", type=str, default=None,
        help="학습이 끝난 AET 런 디렉토리 (config.json + model/). 그 런의 ν/μ 를 불러와 **고정**하고 "
             "정책만 학습한다. ν/μ 손실은 정책을 읽지 않으므로 추출 방식끼리 같은 w 로 짝지어 비교된다. "
             "dual 설정(G/F/β/divergence/hidden/dataset …)이 다르면 시작 시 assert 로 멈춘다.")
    # --extraction diffusion 전용 (2026-10-22). 다른 추출에서 기본값이 아니면 _validate 가 막는다.
    parser.add_argument(
        "--diffusion_steps", type=int, default=20,
        help="extraction=diffusion: DDPM 스텝 수 K (cosine schedule). eval 비용이 K 에 비례.")
    parser.add_argument(
        "--diffusion_uncond_prob", type=float, default=0.0,
        help="extraction=diffusion: 학습 중 racc 를 null 토큰으로 떨굴 확률 (CFG 학습). "
             "0 = 순수 조건부 (Exp 1). CFG 를 쓰려면 0.1 정도.")
    parser.add_argument(
        "--diffusion_cfg_scale", type=float, default=1.0,
        help="extraction=diffusion: 샘플링 시 racc guidance 스케일 λ. 1 = 순수 조건부 "
             "(null pass 생략). λ>1 은 --diffusion_uncond_prob>0 으로 학습된 모델에만 의미가 있다.")
    parser.add_argument(
        "--traj_mixture", action="store_true",
        help="같은 궤적의 모든 transition 을 **같은 혼합 성분**으로만 채점해 성분 정체성을 "
             "궤적에 고정한다 (EM 식 하드 배정, traj_id 해시 — 라벨/수익 미사용). 성분이 "
             "하나의 정책이 되어 AET 가 요구하는 정책 수준 혼합을 표현한다.")
    parser.add_argument(
        "--global_mixture", action="store_true",
        help="혼합 가중을 상태 비의존 **전역 파라미터**로 둔다. 성분 k 가 국소 모드가 아니라 "
             "하나의 정책이 되어, AET 가 요구하는 **정책들의 혼합**(Zahavy et al. 2021)을 "
             "표현할 수 있다. 롤아웃은 --mixture_commit 과 함께 쓴다.")
    parser.add_argument(
        "--transition_filter_path", type=str, default=None,
        help="IDRL(arXiv:2504.13368) 식 transition 단위 keep 마스크 .npy (bool [n_transitions]). "
             "buffer 가 racc 계산·캐시 이후에 적용한다. --traj_filter_path 와 granularity 가 "
             "다르며 함께 쓸 수 있다.")
    parser.add_argument(
        "--traj_filter_path", type=str, default=None,
        help="Path to a .npy of trajectory-enumeration ids to KEEP (w-based dataset "
             "filtering, Exp3-03 coverage ablation). Built by src.ops.filter_trajectories_by_w. "
             "When set, the buffer trains only on those trajectories. None (default) = "
             "full dataset. Routes the run under ablation_study/wfilter_<name>/.",
    )
    parser.add_argument(
        "--resume_dir", type=str, default=None,
        help="Resume training from a checkpoint. Path to a previous save_dir containing "
             "model/ + resume_key.npy. save_dir is reused; all other args must match the "
             "original run (not enforced). Starts at iter = step // log_interval.",
    )
    return parser


def _validate(config, parser):
    """Cross-argument consistency checks.

    Anything that can't be expressed as a single argparse rule lives here.
    Failures raise via `parser.error(...)` so the user sees the argparse-style
    "<prog>: error: …" message rather than a deep traceback."""
    # --beta_init / --beta_final are required iff --beta_schedule != constant.
    if config.beta_schedule == "constant":
        if config.beta_init is not None or config.beta_final is not None:
            parser.error(
                "--beta_init / --beta_final are only valid when --beta_schedule != constant; "
                "for constant schedule, set --beta directly."
            )
    else:
        if config.beta_init is None or config.beta_final is None:
            parser.error(
                f"--beta_schedule={config.beta_schedule} requires both --beta_init and --beta_final"
            )
        if config.beta_init <= 0 or config.beta_final <= 0:
            parser.error("--beta_init and --beta_final must be > 0")
        if config.beta_warmup_steps < 0:
            parser.error("--beta_warmup_steps must be >= 0")
        if config.beta_warmup_steps >= config.total_train_steps:
            parser.error(
                f"--beta_warmup_steps ({config.beta_warmup_steps}) must be < "
                f"--total_train_steps ({config.total_train_steps})"
            )

    # --power_p is valid iff --utility_F_type == 'power_F'. Fail fast on misuse.
    if config.utility_F_type == "power_F" and config.power_p is None:
        parser.error("--power_p is required when --utility_F_type=power_F")
    if config.utility_F_type != "power_F" and config.power_p is not None:
        parser.error(
            f"--power_p is only valid when --utility_F_type=power_F "
            f"(got utility_F_type={config.utility_F_type!r})"
        )
    # --reward_threshold semantics:
    #   piecewise_log_threshold : REQUIRED (the kink is the whole point of this F).
    #   cobbdouglas             : OPTIONAL — turns the unused dim 1 of CD into
    #                             piecewise_log(Reward − T), so a 2-vec G aggregation
    #                             gives log(E[CD]) + log(E[plogT(Reward)]).
    #   anything else           : REJECTED.
    _F_ACCEPTING_THRESHOLD = {"piecewise_log_threshold", "cobbdouglas"}
    # nargs="+" (2026-11-05): 값이 하나면 float 로 되돌린다 — 기존 런과 config·이름·캐시 키가 같게.
    if config.reward_threshold is not None and len(config.reward_threshold) == 1:
        config.reward_threshold = float(config.reward_threshold[0])
    if isinstance(config.reward_threshold, list) and config.cobbdouglas_safety not in CD_SAFETY_THRESHOLDED:
        parser.error(f"--reward_threshold 를 여러 개 주는 것은 --cobbdouglas_safety {CD_SAFETY_THRESHOLDED} (채널마다 제약) 에서만")
    if config.utility_F_type == "piecewise_log_threshold" and config.reward_threshold is None:
        parser.error("--reward_threshold is required when --utility_F_type=piecewise_log_threshold")
    if (config.utility_F_type not in _F_ACCEPTING_THRESHOLD
            and config.reward_threshold is not None):
        parser.error(
            f"--reward_threshold is only valid when --utility_F_type ∈ "
            f"{sorted(_F_ACCEPTING_THRESHOLD)} (got utility_F_type={config.utility_F_type!r})"
        )

    # --pref_weights: linear G 에선 고정 μ (필수), 가중 concave G 에선 G 안의 가중치 (선택).
    from src.utility import WEIGHTED_G_TYPES
    if config.utility_G_type == "linear" and config.pref_weights is None:
        parser.error("--pref_weights is required when --utility_G_type=linear")
    if config.utility_G_type not in WEIGHTED_G_TYPES and config.pref_weights is not None:
        parser.error(
            f"--pref_weights is only valid when --utility_G_type ∈ {WEIGHTED_G_TYPES} "
            f"(got utility_G_type={config.utility_G_type!r})"
        )
    # 새 F/G 모수 (2026-10-25) — 쓰는 곳에서만 받고, 빠지면 즉시 실패 (조용한 기본값 금지).
    if config.utility_F_type == "sigmoid":
        if config.f_tau is None or config.f_sigmoid_scale is None or config.f_sigmoid_scale <= 0:
            parser.error("--utility_F_type=sigmoid needs --f_tau (per objective) and --f_sigmoid_scale > 0")
    elif config.f_tau is not None or config.f_sigmoid_scale is not None:
        parser.error("--f_tau / --f_sigmoid_scale are only valid with --utility_F_type=sigmoid")
    if config.utility_G_type in ("isoelastic", "isoelastic_ce"):
        if config.g_iso_p is None or not config.g_iso_p < 1.0:
            parser.error(f"--utility_G_type={config.utility_G_type} needs --g_iso_p < 1")
    elif config.g_iso_p is not None:
        parser.error("--g_iso_p is only valid with --utility_G_type ∈ {isoelastic, isoelastic_ce}")
    # isoelastic_ce 는 F̃ = exp(R/θ)−1 의 확실성 등가를 쓰므로 F 가 exponential_F 여야 한다.
    if config.utility_G_type == "isoelastic_ce" and config.utility_F_type != "exponential_F":
        parser.error("--utility_G_type=isoelastic_ce requires --utility_F_type=exponential_F "
                     "(its certainty equivalent θ·log(1+v) inverts F̃ = exp(R/θ)−1)")

    # IQL has no μ-update objective (no DICE dual), so learnable-μ G types are
    # incompatible. Fail-fast keeps the misconfiguration from silently training a
    # plog-G IQL with μ pinned at its softplus-init.
    if config.learner_type == "iql" and config.utility_G_type != "linear":
        parser.error(
            f"--learner_type=iql requires --utility_G_type=linear "
            f"(got {config.utility_G_type!r}; IQL has no μ-dual update)."
        )

    if config.critic_from is not None:
        if config.learner_type != "aet":
            parser.error("--critic_from is implemented for --learner_type aet only")
        for need in ("config.json", "model"):
            if not os.path.exists(os.path.join(config.critic_from, need)):
                parser.error(f"--critic_from {config.critic_from!r} has no {need}")

    if config.reward_norm_scale is not None and config.utility_F_type == "cobbdouglas":
        parser.error("--reward_norm_scale 는 CD 와 함께 쓸 수 없다 (CD 는 자체 scale-only 정규화).")
    # timestrat 은 자기 시점 w (w[b, t_b]) 를 정규화한다 — all-H 채점(original)과 traj_mixture 분기는
    # 자기 시점 w 를 쓰지 않으므로 거기서는 조용히 무시된다. 그래서 막는다.
    if config.wbc_weight in ("timestrat", "traj", "action") and (config.extraction == "original"
                                                                 or config.traj_mixture):
        parser.error(f"--wbc_weight {config.wbc_weight} needs a per-timestep extraction "
                     "(timecond / perstep_multihead / diffusion) without --traj_mixture")
    if config.wbc_weight in ("traj", "action") and not config.critic_from:
        parser.error(f"--wbc_weight {config.wbc_weight} needs --critic_from (weights are built once "
                     "from a frozen critic)")

    # Diffusion knobs only mean something under --extraction diffusion; a non-default value
    # elsewhere would be silently ignored, so reject it. MoG-only knobs are rejected the
    # other way round (they would be silently ignored by the denoiser).
    _diff_set = (config.diffusion_steps != 20 or config.diffusion_uncond_prob != 0.0
                 or config.diffusion_cfg_scale != 1.0)
    if config.extraction != "diffusion" and _diff_set:
        parser.error("--diffusion_* flags are only valid with --extraction diffusion")
    if config.extraction == "iproj":
        # log w 가 데이터 밖 행동에서 유한해야 한다 (CHI 의 relu 는 −∞). OptiDICE 도 ELU(soft-χ²) 를 쓴다.
        from src.divergence import FDivergence, LOG_W_DIVERGENCES
        if FDivergence[config.divergence] not in LOG_W_DIVERGENCES:
            parser.error(f"--extraction iproj needs --divergence in "
                         f"{[d.name for d in LOG_W_DIVERGENCES]} (log w must be finite), got {config.divergence}")
        if config.learner_type != "aet":
            parser.error("--extraction iproj is implemented for --learner_type aet only")
        if config.wbc_weight != "raw" or config.traj_mixture:
            parser.error("--extraction iproj does not weight data actions; "
                         "--wbc_weight / --traj_mixture do not apply")
    if config.extraction == "diffusion":
        if config.learner_type != "aet":
            parser.error("--extraction diffusion is implemented for --learner_type aet only")
        if config.traj_mixture or config.global_mixture:
            parser.error("--traj_mixture / --global_mixture are MoG-only; "
                         "not valid with --extraction diffusion")
        if not 0.0 <= config.diffusion_uncond_prob < 1.0:
            parser.error(f"--diffusion_uncond_prob must be in [0, 1) "
                         f"(got {config.diffusion_uncond_prob})")
        if config.diffusion_cfg_scale != 1.0 and config.diffusion_uncond_prob == 0.0:
            parser.error("--diffusion_cfg_scale != 1 needs a model trained with "
                         "--diffusion_uncond_prob > 0 (the null-token branch is untrained)")

    # CobbDouglas F is a coupled 2-obj utility (Reward^0.5/(1+Cost)^0.5). It only makes
    # sense paired with a CD reshape (which provides the (forward_velocity,
    # ‖a‖²) reward/cost layout that CobbDouglas's (reward, cost) channels expect). Fail-fast
    # prevents silently feeding a raw (obj0, obj1) layout into a formula that treats
    # dim 1 as a positive cost.
    from src.data.reward_transform import CD_RESHAPES as _CD_RESHAPES  # SSOT
    if (config.utility_F_type == "cobbdouglas"
            and config.reward_reshape not in _CD_RESHAPES):
        parser.error(
            f"--utility_F_type=cobbdouglas requires --reward_reshape ∈ "
            f"{sorted(_CD_RESHAPES)} (got {config.reward_reshape!r}; CD indexes R as "
            f"(reward, cost) and only those reshapes produce that layout — 그 reshape 가 "
            f"cost_t = 1 + ‖a_t‖² 까지 만든다)."
        )
    # rho ∈ [0, 1] is enforced inside CobbDouglasUtility too, but doing it here lets
    # the user fail-fast at argparse time instead of after the dataset is loaded.
    if not 0.0 <= config.cobbdouglas_rho <= 1.0:
        parser.error(
            f"--cobbdouglas_rho must be in [0, 1] (got {config.cobbdouglas_rho})."
        )
    # amp must be positive; only meaningful for cobbdouglas — REJECTED elsewhere
    # to avoid silent misuse on the elementwise Utility types.
    if config.cobbdouglas_amp <= 0:
        parser.error(f"--cobbdouglas_amp must be > 0 (got {config.cobbdouglas_amp}).")
    if config.utility_F_type != "cobbdouglas" and config.cobbdouglas_amp != 1.0:
        parser.error(
            f"--cobbdouglas_amp is only valid when --utility_F_type=cobbdouglas "
            f"(got utility_F_type={config.utility_F_type!r}, amp={config.cobbdouglas_amp})."
        )
    # rate 형태 CD 는 F 가 (R, 스텝 수) 의 함수다. 조건부 입력이 ftilde 면 eval 이 스텝 수 없이 F 를
    # 불러야 해서 정의가 안 되고, rate(R/t) 가 F 가 읽는 값 그대로라 그것만 허용한다.
    if config.cobbdouglas_form != "total":
        if config.utility_F_type != "cobbdouglas":
            parser.error(f"--cobbdouglas_form={config.cobbdouglas_form} 은 "
                         f"--utility_F_type=cobbdouglas 에서만 쓴다.")
        if config.racc_encoding != "rate":
            parser.error(f"--cobbdouglas_form=rate 는 --racc_encoding rate 가 필요하다 "
                         f"(got {config.racc_encoding!r}).")
    # safe-CD 결합 방식 (2026-11-03) — 조합 실수를 조용히 넘기지 않는다.
    if config.cobbdouglas_safety != "plog" or config.cobbdouglas_kappa != 1.0:
        from src.data.reward_transform import WALKER2D_CD_SAFE_RESHAPES
        if config.utility_F_type != "cobbdouglas" or config.cobbdouglas_form != "rate":
            parser.error("--cobbdouglas_safety linear/product 는 --utility_F_type cobbdouglas --cobbdouglas_form rate 에서만")
        if config.reward_reshape not in WALKER2D_CD_SAFE_RESHAPES:
            parser.error(f"--cobbdouglas_safety {config.cobbdouglas_safety} 는 safe-CD reshape "
                         f"{WALKER2D_CD_SAFE_RESHAPES} 가 필요하다 (got {config.reward_reshape!r})")
        if config.cobbdouglas_safety == "product" and config.utility_G_type != "linear":
            parser.error("--cobbdouglas_safety product 는 --utility_G_type linear (F = CD_rate·S̄^κ 가 곧 목적)")
        if config.cobbdouglas_safety == "linear" and config.utility_G_type != "piecewise_log":
            parser.error("--cobbdouglas_safety linear 는 --utility_G_type piecewise_log (논문 식의 바깥 log)")
        if config.cobbdouglas_safety in ("inlog", "violog") and config.utility_G_type != "linear":
            parser.error(f"--cobbdouglas_safety {config.cobbdouglas_safety} 는 --utility_G_type linear (log 는 이미 F 안에 있다)")
        if config.cobbdouglas_safety == "violog" and config.reward_threshold is not None:
            parser.error("--cobbdouglas_safety violog 는 문턱이 없다 (--reward_threshold 를 주지 말 것)")
        if config.cobbdouglas_kappa != 1.0 and config.cobbdouglas_safety != "product":
            parser.error("--cobbdouglas_kappa 는 --cobbdouglas_safety product 에서만")


def parse_and_validate() -> SimpleNamespace:
    """Single entrypoint for argument parsing. Returns a `SimpleNamespace` config
    with all CLI fields set + `continuous_action = True` baked in."""
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    # parse_known_args 는 모르는 플래그를 **조용히 버린다**. 오타나 파이프라인이 잘못 만든
    # 하이퍼파라미터가 소리 없이 사라져 기본값으로 학습되는 사고가 난다 (2026-09-14:
    # `--max_trajectories 300` 이 무시된 채 50k 궤적 전체로 돌았다 — 그 플래그는 CLI 에
    # 아예 없다). 동작은 그대로 두되(일부 런처가 여분 인자를 넘긴다) **크게 알린다**.
    if unknown:
        print(f"[cli] !! 인식하지 못한 인자 {unknown} — 무시하고 진행한다. "
              f"의도한 하이퍼파라미터라면 build_parser 에 없다는 뜻이다.", flush=True)
    config = SimpleNamespace(**vars(args))
    config.continuous_action = True
    _validate(config, parser)
    return config
