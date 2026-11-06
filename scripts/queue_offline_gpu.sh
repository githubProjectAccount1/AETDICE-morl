#!/bin/bash
# 단일 GPU 메모리 큐 — 오프라인 학습 런을 GPU 에 ≥ MIN_FREE_MB 가 있을 때 하나씩 띄운다 (2026-11-03).
# 왜 하나로 합쳤나: 큐가 둘이면 같은 순간 같은 GPU 의 빈 메모리를 둘 다 보고 런을 띄워 OOM 이 난다.
#   (queue_cd_walker_offline_expert.sh 의 일반화 — 그 스크립트는 이미 돌고 있는 런들의 부모라 그대로 둔다.)
# 대기 파일 한 줄 = run_branches_moant.sh 에 줄 환경변수 할당들 (TAG_PREFIX 필수). 예:
#   TAG_PREFIX=cdexp_mog_rho0.05 CD_RHO=0.05 SEEDS=1
#   TAG_PREFIX=safecd_s1_prod_rho0.7 CD_RHO=0.7 SEEDS=2 RESHAPE=walker2d_cd_safe_vel CD_SAFETY=product RSF="1 1 1"
# 기본값 (줄에서 덮어쓸 수 있다) = Walker CD expert 설정 (D6.14): original MoG, β 1e-4, G linear [1,0], rate.
# 사용: setsid nohup bash scripts/queue_offline_gpu.sh > logs/offline_cd/queue_gpu.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
QUEUE=${QUEUE:-logs/offline_cd/queue_gpu.txt}
MIN_FREE_MB=${MIN_FREE_MB:-8000}
GAP_S=${GAP_S:-180}              # 새 런이 메모리를 다 잡을 때까지 다음 판단을 미룬다 (2 분으로는 부족했다: GPU 당 4 런)
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}
DEFAULTS='BRANCH=original BETA=1e-4 G=linear PREF="1 0" F=cobbdouglas RESHAPE=walker2dcd CD_FORM=rate RACC_ENC=rate ENV=walker2d DS=./data/MO-Walker2d-v2_50000_expert_uniform.pkl SAVE=./results_cd_walker_expert'
MAX_PER_GPU=${MAX_PER_GPU:-2}   # 2026-11-04 (사용자: "oom 날 수 있으면 gpu마다 job을 1개에서 2개만"): 빈 메모리만 보면 막 뜬 런이 아직 메모리를 안 잡아 GPU 하나에 4 런이 몰리고,
                                #   4 런은 24GB 에 안 들어가 OOM 후 메모리를 쥔 채 멈춘다 (11-03 01시 β 6 런, 15시 GPU3 4 런). 런 수로도 막는다.
free_gpu() { local uu; uu=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
             nvidia-smi --query-gpu=index,memory.free,uuid --format=csv,noheader,nounits |
             while IFS=', ' read -r i f u; do n=$(grep -c "$u" <<< "$uu")
               [ "$f" -gt "$MIN_FREE_MB" ] && [ "$n" -lt "$MAX_PER_GPU" ] && { echo "$i"; break; }; done; }
while [ -s "$QUEUE" ]; do
  G=$(free_gpu); [ -n "$G" ] || { sleep 60; continue; }
  LINE=$(head -1 "$QUEUE"); sed -i 1d "$QUEUE"
  TAGP=$(echo "$LINE" | grep -o 'TAG_PREFIX=[^ ]*' | cut -d= -f2)
  SEED=$(echo "$LINE" | grep -o 'SEEDS=[^ ]*' | cut -d= -f2)
  [ -n "$TAGP" ] && [ -n "$SEED" ] || { echo "[$(date '+%F %T')] 잘못된 줄 (TAG_PREFIX·SEEDS 필요): $LINE"; continue; }
  TAG=${TAGP}_original_s$SEED
  echo "[$(date '+%F %T')] start $TAG GPU$G | $LINE"
  ( eval "env $DEFAULTS GPU=$G $LINE bash scripts/run_branches_moant.sh" > "logs/offline_cd/$TAG.lane.log" 2>&1
    echo "[$(date '+%F %T')] done  $TAG $(grep -o 'rc=[0-9]*' "logs/offline_cd/$TAG.lane.log" | tail -1)" ) &
  sleep "$GAP_S"
done
wait
echo "[$(date '+%F %T')] queue empty"
