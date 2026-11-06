#!/bin/bash
# Walker CD 정책의 영상 + 파노라마 (2026-11-04).
# 사용자: "실패하는 rollout들에 대해서 비디오 렌더링해줘 … 파노라마처럼 이미지 하나 길게" (FILTER=falls)
#         "CD utility의 잘 걸어가거나 뛰어가는 rho별로 파노라마도" (FILTER=completes)
#         "나중에 cd with safety 나오면 이것도 파노라마 만들거야" → 같은 스크립트에 safe-CD 런 목록을 주면 된다.
# 입력 파일 한 줄 = "<이름> <run_dir>". 런마다 조건에 맞는 에피소드 N 개 → mp4 1 개 + 에피소드별 파노라마 PNG.
#   ★ 에피소드끼리 위아래로 붙이지 않는다 (사용자 2026-11-04: "위 아래를 이렇게 붙이면 안돼고, 각각 쪼개서 줘").
# 사용: FILTER=completes OUT=results_viz/walks bash scripts/render_cd_panorama.sh <runs.txt>
cd "$(dirname "$0")/.."
source scripts/aetenv.sh >/dev/null 2>&1
export JAX_PLATFORMS=cpu MUJOCO_GL=egl XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
FILTER=${FILTER:?FILTER=falls|completes|all}
OUT=${OUT:?OUT=<출력 디렉터리>}
N=${N:-2}                       # 런당 에피소드 수
MAX_TRIES=${MAX_TRIES:-150}
WINDOW=${WINDOW:-$([ "$FILTER" = falls ] && echo 150 || echo 0)}   # 낙상은 마지막 150 스텝, 나머지는 에피소드 전체
PANELS=${PANELS:-$([ "$FILTER" = falls ] && echo 10 || echo 12)}
mkdir -p "$OUT"
while read NAME D; do
  ( python -m src.viz.render_video --run_dir "$D" --out_mp4 "$OUT/$NAME.mp4" --num_episodes $N \
      --episode_filter $FILTER --max_tries $MAX_TRIES --frame_stride 2 \
      --panorama_png "$OUT/$NAME" --pano_window $WINDOW --pano_panels $PANELS > "$OUT/$NAME.log" 2>&1
    echo "$NAME rc=$?" ) &
done < "$1"
wait
