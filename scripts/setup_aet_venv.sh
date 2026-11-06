#!/usr/bin/env bash
# venv `aet` (오프라인 학습·평가 + 원본 mujoco-py env) 를 프로젝트 안 .aetenv 로 만든다.
# 왜 별도 스크립트인가: bootstrap.sh 는 시스템 python 을 점검만 하고 venv 를 만들지 않는다.
#   시스템 python 이 3.12 · numpy 2 이면 mujoco-py 가 import 에서 죽는다.
# 스펙: transfer/env_specs/aet_pip_freeze.txt (python 3.9.23). conda 빌드 경로(file://) 줄은 pip 로 못 받으므로
#   빼고, 그중 실제로 필요한 gym 만 버전을 고정해 따로 받는다 (gym.envs.mujoco.mujoco_env 가 mujoco-py 를 쓰는 마지막 계열).
# 사용 (repo 루트): bash scripts/setup_aet_venv.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
VENV=$REPO/.aetenv
GYM_PIN="gym==0.22.0"
MUJOCO210_URL=https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

echo "== 1. 시스템 패키지 (mujoco-py 빌드 의존)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential patchelf libglew-dev libosmesa6-dev libgl1-mesa-dev libgl1 libglfw3 >/dev/null

echo "== 2. MuJoCo 210 바이너리 (mujoco-py 가 ~/.mujoco/mujoco210 을 기본 경로로 찾는다)"
if [ ! -d ~/.mujoco/mujoco210 ]; then
  mkdir -p ~/.mujoco "$REPO/.cache"
  wget -q "$MUJOCO210_URL" -O "$REPO/.cache/mujoco210.tar.gz"
  tar -xzf "$REPO/.cache/mujoco210.tar.gz" -C ~/.mujoco
fi

echo "== 3. venv $VENV (python 3.9)"
[ -d "$VENV" ] || uv venv -q --python 3.9 "$VENV"
REQ=$REPO/.cache/aet_requirements.txt
mkdir -p "$REPO/.cache"
grep -v ' @ file://' ${AET_FREEZE:-transfer/env_specs/aet_pip_freeze.txt} > "$REQ"
VIRTUAL_ENV=$VENV uv pip install -q -r "$REQ"
VIRTUAL_ENV=$VENV uv pip install -q "$GYM_PIN"

echo "== 4. mujoco-py 빌드 확인 (첫 import 에서 Cython 컴파일)"
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_PY_FORCE_CPU=1
"$VENV/bin/python" -c "import numpy, mujoco_py, gym, jax; print('numpy', numpy.__version__, '| gym', gym.__version__, '| jax', jax.__version__, jax.devices())"
"$VENV/bin/python" -c "
from environment.walker2d import Walker2dEnv
import numpy as np
e = Walker2dEnv(); o = e.reset(); o2, r, d, info = e.step(np.zeros(e.action_space.shape))
print('walker2d ok obs', o.shape, 'reward', r)"
echo "== 완료"
