#!/bin/bash
# Figure 5(a): 100 MO-Ant rollouts (top-down) of one AETDICE policy (convex F + concave G) drawn over
# the dataset coverage (light gray), with obj 0 (+x) / obj 1 (+y) axes.
#   1) roll out the policy 100 times and save the xy trajectories to an npz
#   2) rebuild the dataset xy trajectories from the rewards (src.viz.moant_dataset_xy, ~200 MB npz)
#   3) draw coverage -> trajectories -> axes and crop to the panel used in the paper
# Usage (repo root):
#   RUN=<AETDICE MO-Ant run dir> bash scripts/render_ant_coverage.sh
# Paper figure: RUN = the seed-5 run of the MO-Ant AETDICE command in README.md.
set -eu
cd "$(dirname "$0")/.."
source scripts/aetenv.sh
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 MUJOCO_GL=egl PYTHONPATH=$PWD
RUN=${RUN:?RUN=<MO-Ant AETDICE run dir>}
DS=${DS:-./data/MO-Ant-v2_50000_expert_uniform.pkl}
OUT=${OUT:-results_viz/ant}
ROLL=$OUT/rollouts100.npz
DS_XY=${DS_XY:-$OUT/moant_ds_xy.npz}
mkdir -p "$OUT"
ARGS="--spaghetti --num_episodes 100 --seed 0 --width 1100 --height 1100 --distance 165 \
  --lookat_x 60 --lookat_y 60 --trail_size 0.35 --trail_stride 2 --trail_alpha 1.0 \
  --spaghetti_endpoint_scale 3.0 --clean_floor --spaghetti_color_by heading --spaghetti_cmap turbo"
[ -f "$ROLL" ] || python -m src.viz.render_topdown --run_dir "$RUN" --out_mp4 "$OUT/_rollouts.png" \
  --spaghetti_save_npz "$ROLL" $ARGS
[ -f "$DS_XY" ] || python -m src.viz.moant_dataset_xy --pkl "$DS" --out "$DS_XY"
python -m src.viz.spaghetti_from_npz --npz "$ROLL" --out "$OUT/ant_coverage.png" \
  --box -8.345237791560677 128.3452377915607 --coverage_npz "$DS_XY" --obj_axes \
  --crop 40 1057 43 1034 --axis_len 112 --solid_color '#1f77b4'
