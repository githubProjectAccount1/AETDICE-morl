# AETDICE 실행 환경 — venv(aet) 활성화 + mujoco/osmesa 경로. (cd 없음: 현재 디렉토리 유지)
# 프로젝트 안 버전. 사용: `source scripts/aetenv.sh` (repo 루트에서).
# ★ venv activate(conda-pack 생성)는 $CONDA_PREFIX·$PS1 를 가드 없이 참조한다 → 호출 스크립트가
#   `set -u` 면 source 하는 순간 셸이 조용히 죽는다 (2026-09-07 실제 발생: run_branches_moant.sh 가
#   아무 출력 없이 즉시 종료). activate 는 우리 파일이 아니므로 여기서 nounset 을 잠시 끄고 복구한다.
case $- in *u*) _AETENV_HAD_U=1 ;; *) _AETENV_HAD_U=0 ;; esac
set +u
# repo 루트는 이 파일 위치에서 구한다 (clone 경로와 무관).
_AET_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# venv: 프로젝트 안 .aetenv (scripts/setup_aet_venv.sh) 가 있으면 그것, 없으면 ~/envs/aet. AET_VENV 로 덮어쓸 수 있다.
if [ -z "${AET_VENV:-}" ]; then
  if [ -d "$_AET_REPO/.aetenv" ]; then AET_VENV=$_AET_REPO/.aetenv; else AET_VENV=~/envs/aet; fi
fi
source "$AET_VENV/bin/activate"
export MUJOCO_PY_FORCE_CPU=1
# ${VAR:-} 로 감싼다: 호출 스크립트가 `set -u` 면 미설정 변수 참조가 셸 전체를 조용히 죽인다
# (2026-09-07 실제 발생 — run_branches_moant.sh 가 아무 출력 없이 즉시 종료).
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$HOME/miniconda3/envs/mesa21/lib:$AET_VENV/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline

# JAX 영구 compile cache — 같은 XLA 실행체 재컴파일 회피(두 번째 run부터 warm-start).
# 프로젝트 안(.jax_cache)에 둠. min_compile_time 1s → 사소한 컴파일은 캐시 제외(디스크 절약),
# 학습 scan 같은 무거운 컴파일만 캐시. 버그 fix 후(2026-09-02) fresh 생성.
export JAX_COMPILATION_CACHE_DIR=$_AET_REPO/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0

if [ "$_AETENV_HAD_U" = 1 ]; then set -u; fi
unset _AETENV_HAD_U
