#!/bin/bash
# MO-Ant AETDICE — **정책 추출 두 줄기** 표준 런처. 이 두 개가 유일한 메인 경로다.
#
#   A) original   multi-head 구조 + racc 조건부만, all-H WBC. **논문 Table 4 를 정확히 재현**하는
#                 그 방식이다 (시간은 입력이 아니라 출력 슬롯).
#   B) timecond   raw remaining-time 을 policy 입력으로 넣고 WBC 를 자기 timestep 에서만 채점,
#                 head 는 single (multi-head 없음).
#
# 두 축(구조 × 채점)을 `--extraction` 하나로 묶었다 — 따로 두면 "multi-head + per-timestep"
# 같은 제3 조합이 생기고 어느 게 논문 설정인지 추적이 안 된다 (2026-09-07 실제 발생).
#
# 나머지 하이퍼파라미터는 헤드라인(`reprofix`)과 동일하게 고정한다:
#   β=1e-6 (BETA 로만 바꾼다 — β sweep 용, 2026-10-23), CHI, num_layers=4, hidden_dim=512,
#   batch_size=512, max_seq_len=500,
#   mu_scaling_factor=1e4, reward_scaling_factor=[1,1], total_train_steps=1e6,
#   final_eval_episodes=100.  clip 계열(chi clip/grad clip) 전부 OFF — pristine.
#
# 사용:
#   GPU=2 BRANCH=original SEEDS="1 2 3" bash scripts/run_branches_moant.sh
#   GPU=3 BRANCH=timecond SEEDS="1 2 3" bash scripts/run_branches_moant.sh
#   GPU=1 BRANCH=diffusion SEEDS="1 2 3 4 5" TAG_PREFIX=diffu0 bash scripts/run_branches_moant.sh
#   GPU=2 BRANCH=diffusion SEEDS="1 2 3 4 5" TAG_PREFIX=diffu10 \
#     EXTRA_ARGS="--diffusion_uncond_prob 0.1" bash scripts/run_branches_moant.sh
#   EXTRA_ARGS 는 diffusion 하이퍼파라미터 전용 통로다 (2026-10-22).
#   GPU=2 BRANCH=diffusion SEEDS=1 BETA=1e-3 TAG_PREFIX=diffu0b1e-3 SAVE=./results_diffusion \
#     bash scripts/run_branches_moant.sh
#   ★ BETA 를 바꾸면 TAG_PREFIX 에도 β 를 넣어라 — 로그는 logs/<TAG>.log 라 β 만 다르면 덮어쓴다.
#   Cobb-Douglas (2026-10-24): G=linear PREF="1 0" F=cobbdouglas RESHAPE=antcd CD_RHO=<ρ> 를 같이 준다.
#   GPU=2 BRANCH=diffusion SEEDS=1 BETA=1e-5 G=linear PREF="1 0" F=cobbdouglas RESHAPE=antcd \
#     CD_RHO=0.5 TAG_PREFIX=cddiffu_rho0.5 SAVE=./results_diffusion bash scripts/run_branches_moant.sh
#   ★ CD_RHO 를 바꾸면 TAG_PREFIX 에도 ρ 를 넣어라 (BETA 와 같은 이유).
#   Walker2d CD exp A (2026-10-24): ENV=walker2d DS=<walker pkl> RESHAPE=walker2dcd CD_FORM=rate RACC_ENC=rate
#   — 드라이버 scripts/run_cd_walker_expA.sh 가 이 조합만 넘긴다 (학습 런처는 여전히 이것 하나).
# 알고리즘(G/F)까지 바꾸는 6-조합 전체는 상위 드라이버 scripts/run_algo_grid_moant.sh 를 쓴다
# (이 스크립트가 유일한 학습 런처이고 드라이버는 G/F/PREF 만 넘긴다 — 중복 런처 금지).
# 채점: source scripts/aetenv.sh && python -m experiments.moant.canonical_table

set -u
cd "$(dirname "$0")/.."   # repo 루트 (clone 경로와 무관)
source scripts/aetenv.sh 2>/dev/null
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

GPU=${GPU:?GPU 를 지정해라 (예: GPU=0)}
BRANCH=${BRANCH:?BRANCH 를 지정해라: original | timecond | diffusion | iproj}
EXTRA_ARGS=${EXTRA_ARGS:-}      # diffusion 전용 (--diffusion_*). 다른 BRANCH 에선 cli 가 거부한다
SEEDS=${SEEDS:-"1 2 3"}
BETA=${BETA:-1e-6}              # 헤드라인 기본값. run 이름의 beta 토큰도 이 값을 따른다
G=${G:-piecewise_log}          # AET 축 기본: G=piecewise_log × F=exponential_F
F=${F:-exponential_F}
PREF=${PREF:-}                 # G=linear 일 때만 의미 있음 (ESR 은 "1.0 1.0", CD 는 "1 0")
RESHAPE=${RESHAPE:-none}       # CD 는 antcd 필수 (cli 가 검사한다)
CD_RHO=${CD_RHO:-}             # F=cobbdouglas 일 때만. 비우면 cli 기본값(0.5)
CD_FORM=${CD_FORM:-total}      # F=cobbdouglas 일 때만: total(기존) | rate (exp A, RACC_ENC=rate 필수)
CD_SAFETY=${CD_SAFETY:-}       # safe-CD 결합: plog(기본) | linear (논문 식, THRESH 필요) | product (F=CD_rate·S̄^κ) — 2026-11-03
CD_KAPPA=${CD_KAPPA:-}         # CD_SAFETY=product 의 κ
THRESH=${THRESH:-}             # --reward_threshold (safe-CD: raw 합 단위 S_th)
RSF=${RSF:-1 1}                # --reward_scaling_factor (reshape 채널 수만큼 — safe-CD 3 채널은 "1 1 1")
RACC_ENC=${RACC_ENC:-ftilde}   # 조건부 racc 입력: ftilde(기본) | rate
ENV=${ENV:-ant}                # Walker2d CD (exp A, 2026-10-24) 는 ENV=walker2d + DS=<walker pkl>
TAG_PREFIX=${TAG_PREFIX:-}     # run 이름 앞에 붙일 알고리즘 토큰 (예: aetdice / aetesr / aetser)
SAVE=${SAVE:-./results_branches}
# ESR-IQL 기준선 (2026-11-05, D6.24 — 사용자: "Run BC + ESR-IQL, 3 seeds" / "Small sweep, then 3 seeds"):
#   LEARNER=iql 이면 같은 런처로 IQL 을 학습한다 (별도 런처 금지 — SSOT). G 는 linear 여야 한다 (cli 검사).
#   IQL_TAU/IQL_TEMP = --expectile/--temperature, STEPS 기본 1e6 (논문 IQL 은 3e5 → STEPS=300000 으로 준다).
LEARNER=${LEARNER:-aet}         # aet | iql | bc
#   LEARNER=bc: racc 조건부 MoG BC, 필터 없음 (k_pct 1.0) — scripts/run_cd_walker_offline_expert.sh 의 BC 와 같은 인자.
#   BC 는 ρ 와 무관하게 학습되므로 seed 당 한 번만 돌리고 다른 ρ 는 저장된 returns 로 재채점한다.
IQL_TAU=${IQL_TAU:-}
IQL_TEMP=${IQL_TEMP:-}
STEPS=${STEPS:-1000000}
DS=${DS:-./environment/dataset/MO-Ant-v2/MO-Ant-v2_50000_expert_uniform.pkl}

case "$BRANCH" in
  original|timecond|diffusion|iproj) ;;
  *) echo "알 수 없는 BRANCH: $BRANCH (original | timecond | diffusion | iproj)"; exit 2 ;;
esac
[ -n "$PREF" ] && PREF_ARG="--pref_weights $PREF" || PREF_ARG=""
[ "$RESHAPE" != none ] && RESHAPE_ARG="--reward_reshape $RESHAPE" || RESHAPE_ARG=""
[ -n "$CD_RHO" ] && CD_ARG="--cobbdouglas_rho $CD_RHO" || CD_ARG=""
[ "$CD_FORM" != total ] && CD_ARG="$CD_ARG --cobbdouglas_form $CD_FORM"
[ -n "$CD_SAFETY" ] && CD_ARG="$CD_ARG --cobbdouglas_safety $CD_SAFETY"
[ -n "$CD_KAPPA" ] && CD_ARG="$CD_ARG --cobbdouglas_kappa $CD_KAPPA"
[ -n "$THRESH" ] && CD_ARG="$CD_ARG --reward_threshold $THRESH"
[ "$RACC_ENC" != ftilde ] && RACC_ARG="--racc_encoding $RACC_ENC" || RACC_ARG=""
IQL_ARG=""
[ -n "$IQL_TAU" ] && IQL_ARG="$IQL_ARG --expectile $IQL_TAU"
[ -n "$IQL_TEMP" ] && IQL_ARG="$IQL_ARG --temperature $IQL_TEMP"
case "$LEARNER" in aet|iql|bc) ;; *) echo "알 수 없는 LEARNER: $LEARNER (aet | iql | bc)"; exit 2 ;; esac
[ "$LEARNER" = bc ] && IQL_ARG="$IQL_ARG --bc_conditioning racc --cd_filter_rho 0.5 --cd_filter_k_pct 1.0"
[ "$LEARNER" = aet ] && [ -n "$IQL_ARG" ] && { echo "IQL_TAU/IQL_TEMP 는 LEARNER=iql 전용"; exit 2; }

mkdir -p logs "$SAVE"
echo "===== [$ENV/$BRANCH] G=$G F=$F${CD_RHO:+ ρ=$CD_RHO} cd_form=$CD_FORM racc=$RACC_ENC${RESHAPE:+ reshape=$RESHAPE} β=$BETA seeds='$SEEDS' GPU$GPU → $SAVE  $(date '+%F %T') ====="
for SEED in $SEEDS; do
  TAG=${TAG_PREFIX:+${TAG_PREFIX}_}${BRANCH}_s${SEED}
  echo "--- $TAG 시작 $(date '+%T') ---"
  CUDA_VISIBLE_DEVICES=$GPU python -m src.train.main \
    --env_name "$ENV" --learner_type "$LEARNER" --extraction "$BRANCH" \
    --utility_G_type "$G" --utility_F_type "$F" $PREF_ARG $RESHAPE_ARG $CD_ARG $RACC_ARG $IQL_ARG \
    --divergence CHI --beta "$BETA" --seed "$SEED" \
    --batch_size 512 --hidden_dim 512 --num_layers 4 \
    --total_train_steps "$STEPS" --log_interval 100000 --chunk_size 5000 \
    --eval_every_n_logs 2 --final_eval_episodes 100 \
    --max_seq_len 500 --mu_scaling_factor 1e4 --reward_scaling_factor $RSF \
    --dataset_path "$DS" --save_path "$SAVE" --tag "$TAG" $EXTRA_ARGS > "logs/${TAG}.log" 2>&1
  echo ">>> $TAG rc=$? $(date '+%T')"
done
echo "===== [$BRANCH] 완료 $(date '+%F %T') ====="
