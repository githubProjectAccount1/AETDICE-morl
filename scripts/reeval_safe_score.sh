#!/bin/bash
# 기준선 safe-CD 열 채우기 (2026-11-05, D6.24 — 사용자: "Re-evaluate in safety env").
# 입력 파일 한 줄 = run_dir. 학습 reshape 로 정책을 돌리고 SCORE_RESHAPE 채널 합을 eval/<OUT>_score.npy 에 저장.
# 동시 실행 PAR 개 (32 개 동시 렌더가 스레드를 고갈시킨 2026-11-04 사고 → 작게 유지).
# 사용: PAR=3 bash scripts/reeval_safe_score.sh logs/safe_reeval_runs.txt
cd "$(dirname "$0")/.."
source scripts/aetenv.sh >/dev/null 2>&1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
SCORE_RESHAPE=${SCORE_RESHAPE:-walker2d_cd_safe_pv}
OUT=${OUT:-safepv_100}
N=${N:-100}
PAR=${PAR:-3}
xargs -P "$PAR" -I{} sh -c '
  if [ -f "{}/eval/'"$OUT"'_score.npy" ]; then echo "skip {}"; exit 0; fi
  python -m src.eval.rollout --run_dir "{}" --num_episodes '"$N"' --score_reshape '"$SCORE_RESHAPE"' --out_name '"$OUT"' \
    > "{}/eval/'"$OUT"'.log" 2>&1; echo "rc=$? {}"' < "$1"
