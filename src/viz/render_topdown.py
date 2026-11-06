"""Render a trained MO-Ant policy from a fixed top-down view, save as MP4.

Two camera modes:
  • cam_mode=fixed  (default) — camera at a stationary world point. The ant
    starts near (0, 0) and moves +x / +y / diagonally depending on the
    policy's preference direction. Use this for "where did the policy decide
    to go?" videos. Default lookat puts the ant near the bottom-left of the
    frame so there's room to walk up-and-to-the-right.
  • cam_mode=track  — camera follows the ant; ant always centered. Use this
    for gait analysis (the moving floor checker provides the speed cue).

The two MO-Ant objectives are vx_reward (x-axis velocity) and vy_reward
(y-axis velocity). Default --azimuth=0 maps world +x to image-right and
world +y to image-up, so:
    pref=[1.0, 0.0]   → ant runs toward the RIGHT
    pref=[0.0, 1.0]   → ant runs UP
    pref=[0.5, 0.5]   → ant runs UP-AND-RIGHT (diagonal)

Usage:
    # Default fixed cam, ant at bottom-left of frame, trail on
    python -m src.render_topdown \
        --run_dir <RUN_DIR> --out_mp4 ant.mp4

    # Wider view to show longer trajectory; iterate on --distance until happy
    python -m src.render_topdown \
        --run_dir <RUN_DIR> --out_mp4 ant.mp4 \
        --distance 50 --lookat_x 18 --lookat_y 14

    # Tracking camera (old behaviour) for gait detail
    python -m src.render_topdown \
        --run_dir <RUN_DIR> --out_mp4 ant_gait.mp4 \
        --cam_mode track --distance 6

The default `MUJOCO_GL=egl` uses GPU-accelerated headless rendering. If EGL
fails on your box, set `MUJOCO_GL=osmesa` (CPU software rasterizer; slower
but no GPU/X11 needed) before invoking the script.


python -m src.render_topdown --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0to1.0to1.0to1.0_musf1e4_antv5/G-plog_F-piecewise_log_beta1e-6_seed1_tfd4morl_ant_expert --out_mp4 20ant_x8y8.mp4 --width 720 --height 720 --distance 50 --lookat_x 20 --lookat_y 20 --trail_size 0.15 --seed 2


python -m src.render_topdown --run_dir ./results_Transformed/ant_amateur/rsf1.0to1.0_musf1e4_antcd/G-linear_w1.0to0.0_F-cobbdouglas_beta1e-4_seed1_T100_tfd4morl_ant_amateur --out_mp4 cd_t100.mp4 --width 720 --height 720 --distance 50 --lookat_x 20 --lookat_y 20 --trail_size 0.15 --seed 1



python -m src.render_topdown --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e4/G-plog_F-exp_F_beta1e-5_seed1_tfd4morl_ant_expert --out_mp4 20ant_x7y7.mp4 --width 720 --height 720 --distance 20 --lookat_x 7 --lookat_y 7 --trail_size 0.15

python -m src.render_topdown --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e4/G-plog_F-exp_F_beta1e-5_seed1_tfd4morl_ant_expert --out_mp4 20ant_x6y6.mp4 --width 720 --height 720 --distance 20 --lookat_x 6 --lookat_y 6 --trail_size 0.15


python -m src.render_topdown --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e4/G-plog_F-exp_F_beta1e-5_seed1_tfd4morl_ant_expert --out_mp4 25ant_x12y6.mp4 --width 720 --height 720 --distance 25 --lookat_x 12 --lookat_y 6


python -m src.render_topdown --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e4/G-plog_F-exp_F_beta1e-5_seed1_tfd4morl_ant_expert --out_mp4 ant_d150x70y70.mp4 --width 720 --height 720 --distance 150 --lookat_x 70 --lookat_y 70

python -m src.render_topdown --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e4/fairdice_G-plog_F-linear_beta1e-6_seed1_tfd4morl_ant_expert --out_mp4 ant_d150x60y60.mp4 --width 720 --height 720 --distance 150 --lookat_x 60 --lookat_y 60




 python -m src.render_topdown   --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e4/iql_G-linear_w0.0to1.0_F-exp_F_beta1e-6_seed1_tfd4morl_ant_expert   --out_mp4 spaghetti_animated_100ep.mp4   --width 1080 --height 1080 --fps 30   --distance 160 --lookat_x 60 --lookat_y 60   --spaghetti_animated --num_episodes 100   --animated_trail_length 100   --trail_size 0.20 --trail_alpha 0.7   --spaghetti_endpoint_scale 2.5   --spaghetti_cmap hsv



python -m src.render_topdown \
  --run_dir ./results_Transformed/ant_expert/rsf1.0to1.0_musf1e4/iql_G-linear_w0.0to1.0_F-exp_F_beta1e-6_seed1_tfd4morl_ant_expert \
  --out_mp4 spaghetti_animated_30ep_breadcrumbs.mp4 \
  --width 1080 --height 1080 --fps 30 \
  --distance 160 --lookat_x 60 --lookat_y 60 \
  --spaghetti_animated --num_episodes 30 \
  --trail_stride 3 --trail_size 0.20 --trail_alpha 0.7 \
  --spaghetti_endpoint_scale 2.5 --spaghetti_cmap hsv

Timelapse — 100 trajectories + top-down ant sprites every 50 steps → <out>.npz (2026-10-24):
python -m src.viz.render_topdown --run_dir <run> --out_mp4 video/dvm/timelapse_x.npz \
  --timelapse --num_episodes 100 --timelapse_every 50 --clean_floor
  (scripts/render_dvm_timelapse.sh; figure with world x/y axes: experiments.moant.compose_dvm_timelapse)

--num_episodes N               1–100; how many trails to overlay
--spaghetti_cmap NAME          'hsv' (current), 'viridis', 'plasma', 'tab10', 'rainbow', etc.
--spaghetti_endpoint_scale F   endpoint dot = trail_size × F (default 2.5)
--trail_size F                 trail dot radius
--trail_alpha F                trail dot transparency (0–1)
--trail_stride N               minimum stride; auto-coarsens further if budget hits
--width / --height             square recommended for spaghetti
--distance / --lookat_*        camera framing (probably need bigger distance for 100 ep)

"""
import argparse
import os
import sys

# mujoco_py needs /root/.mujoco/mujoco210/bin on LD_LIBRARY_PATH so that the
# Linux dynamic linker can dlopen libglewosmesa.so / libmujoco210.so at import
# time. Critical detail: the dynamic linker reads LD_LIBRARY_PATH at PROCESS
# STARTUP, not on every dlopen call — so just doing `os.environ[...] = ...`
# inside the already-running Python process is NOT enough; it'll satisfy any
# os.environ.get() check in mujoco_py but NOT influence the actual dlopen.
# So: set the env var, then re-exec the same Python with the corrected env so
# the linker re-reads it from scratch. One-shot, no infinite loop because the
# `if` guard short-circuits on the second pass.
#
# Also normalize MUJOCO_GL: setdefault doesn't help if it's already set to ''
# (empty string), which is how some shells leave it. Force `egl` for headless
# GPU rendering when the value is missing or empty.
_MUJOCO_LIB = "/root/.mujoco/mujoco210/bin"
if _MUJOCO_LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = _MUJOCO_LIB + (":" + _ld if _ld else "")
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# MuJoCo rendering backend — set BEFORE mujoco_py is imported (transitively
# pulled in by environment.ant). `egl` is GPU-accelerated headless; `osmesa`
# is a slower CPU fallback if EGL doesn't work; `glfw` opens a window (needs
# DISPLAY).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # eval/rollout is CPU-only anyway

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import imageio
import jax
import jax.numpy as jnp
import mujoco_py
import numpy as np

from environment.ant import AntEnv
from src.eval.rollout import load_config, load_model_cpu
from src.data.reward_transform import RESHAPE_STEP
from src.viz.timelapse import (apply_clean_floor, export_timelapse, final_heading_deg,
                               prepare_model_for_sprites)

# 정적 spaghetti 의 다중 패스 합성 (2026-10-24). MuJoCo scn.maxgeom=1000 에서 축·씬 geom 여유를 뺀 값.
_MARKER_PASS_BUDGET = 900
# 패스 이미지와 배경의 RGB 차이 합이 이보다 크면 marker 픽셀로 본다 (0–765 스케일).
_COMPOSITE_DIFF_THRESH = 24
# heading 색칠 범위 — MO-Ant 의 두 목적이 vx(0°) / vy(90°) 이다.
_HEADING_RANGE_DEG = (0.0, 90.0)


def _heading_unit(heading_deg):
    lo, hi = _HEADING_RANGE_DEG
    return float(np.clip((heading_deg - lo) / (hi - lo), 0.0, 1.0))


def _spaghetti_colors(trajectories, color_by, cmap_name, solid_color=None):
    """궤적별 RGB. index = 에피소드 순서(구 동작), heading = 최종 진행방향 atan2(dy, dx) — 혼합의 모드가
    색으로 갈린다. solid = 모든 궤적 한 색 (solid_color, matplotlib 색 지정; 2026-11-06 사용자 "gradation 색깔을
    trajectory rollout에서 빼봐. 단일 색갈로 맞춰보자")."""
    import matplotlib
    n = len(trajectories)
    if color_by == "solid":
        assert solid_color is not None, "color_by='solid' 는 solid_color 가 필요하다"
        return [tuple(matplotlib.colors.to_rgb(solid_color))] * n
    cmap = matplotlib.colormaps[cmap_name]
    if color_by == "index":
        ts = [i / max(1, n - 1) for i in range(n)]
    elif color_by == "heading":
        ts = [_heading_unit(final_heading_deg(traj)) for traj in trajectories]
    else:
        raise ValueError(f"unknown --spaghetti_color_by {color_by!r}")
    return [tuple(cmap(t)[:3]) for t in ts]


# Same dispatch table main.py uses. Imported here so the renderer
# pulls the right `init_train_state` / `get_model` for whichever learner trained
# the checkpoint we're loading. Without this, the renderer would always assume
# AET (state+racc) and silently corrupt-load fairdice/iql checkpoints.
# (2026-10-24) 예전엔 여기 로컬 표가 있어 "bc" 가 빠져 BC 체크포인트를 못 그렸다 → 공용 레지스트리.
from src.learners import LEARNER_MODULES, is_state_only_policy


def configure_topdown_camera(env, width, height, distance, elevation, azimuth,
                             lookat_x, lookat_y):
    """Lazily create the offscreen viewer and configure its FREE camera.
    Returns the viewer; caller controls it directly each frame.

    Why we pass camera_id=-1 to env.render():
      ant.xml defines `<camera name="track" mode="trackcom" pos="0 -3 0.3" .../>`,
      a behind-view camera that follows the center of mass. gym 0.21's
      MujocoEnv.render(mode='rgb_array') defaults to that camera if none is
      specified — which would override our free-camera settings. camera_id=-1
      tells mujoco_py to use the FREE camera (the `viewer.cam` we configure
      below), restoring the top-down view we want.
    """
    # First render call lazily creates the offscreen viewer.
    _ = env.render(mode="rgb_array", width=width, height=height, camera_id=-1)
    viewer = env._get_viewer("rgb_array")
    viewer.cam.distance  = distance
    viewer.cam.azimuth   = azimuth
    viewer.cam.elevation = elevation
    viewer.cam.lookat[0] = lookat_x
    viewer.cam.lookat[1] = lookat_y
    viewer.cam.lookat[2] = 0.0   # ground level
    return viewer


def _add_axis_markers(viewer, origin=(0.0, 0.0), arrow_length=5.0,
                      arrow_radius=0.12, label_axes=True):
    """Add a compass at the world origin: red arrow along +x, green arrow
    along +y, plus a labeled sphere at each arrow tip ("O" / "+X" / "+Y").

    Why we split arrows from labeled tip-spheres: a marker's label is rendered
    at its `pos`, so if the arrows and origin sphere all sit at (0,0), all the
    labels stack and you can't read them. Labels go on small spheres at the
    arrow ENDPOINTS, which spreads them apart visually.

    Arrow geoms in MuJoCo point along their local-z axis. We supply rotation
    matrices aligning local-z with world-x (R_x) / world-y (R_y), right-handed.
    """
    ox, oy = origin
    z = 0.06  # just above the floor so axes don't z-fight the ground plane

    # ─── Origin sphere (white, labeled "O") ──────────────────────────────
    viewer.add_marker(
        pos=np.array([ox, oy, z]),
        type=mujoco_py.const.GEOM_SPHERE,
        size=np.full(3, 0.18),
        rgba=np.array([1.0, 1.0, 1.0, 1.0]),
        label="O" if label_axes else "",
    )

    # ─── +X arrow (red) ──────────────────────────────────────────────────
    R_x = np.array([
        [ 0, 0, 1],
        [ 0, 1, 0],
        [-1, 0, 0],
    ], dtype=np.float64)
    viewer.add_marker(
        pos=np.array([ox, oy, z]),
        type=mujoco_py.const.GEOM_ARROW,
        size=np.array([arrow_radius, arrow_radius, arrow_length]),
        rgba=np.array([1.0, 0.1, 0.1, 1.0]),
        label="",
        mat=R_x,
    )
    # Invisible label anchor at the +X arrow's TIP. MuJoCo renders the label
    # at the geom's pos; an arrow's pos is its base, so labeling the arrow
    # directly stacks all three labels at the origin. We hang the "+X" label
    # off a tiny transparent sphere placed at the tip — alpha=0 hides the
    # geom but the label still draws at the tip's screen position.
    if label_axes:
        viewer.add_marker(
            pos=np.array([ox + arrow_length, oy, z]),
            type=mujoco_py.const.GEOM_SPHERE,
            size=np.full(3, 0.001),
            rgba=np.array([0.0, 0.0, 0.0, 0.0]),
            label="+X",
        )

    # ─── +Y arrow (green) ────────────────────────────────────────────────
    R_y = np.array([
        [1,  0, 0],
        [0,  0, 1],
        [0, -1, 0],
    ], dtype=np.float64)
    viewer.add_marker(
        pos=np.array([ox, oy, z]),
        type=mujoco_py.const.GEOM_ARROW,
        size=np.array([arrow_radius, arrow_radius, arrow_length]),
        rgba=np.array([0.1, 0.9, 0.1, 1.0]),
        label="",
        mat=R_y,
    )
    if label_axes:
        viewer.add_marker(
            pos=np.array([ox, oy + arrow_length, z]),
            type=mujoco_py.const.GEOM_SPHERE,
            size=np.full(3, 0.001),
            rgba=np.array([0.0, 0.0, 0.0, 0.0]),
            label="+Y",
    )


def spaghetti_frame(env, viewer, trajectories, *, width, height, color_by, cmap_name, trail_size,
                    trail_stride, trail_alpha, endpoint_scale, show_axes, axis_length, axis_label,
                    reset_seed, underlay=None, solid_color=None):
    """궤적 목록 → MuJoCo topdown 정지 이미지 (--spaghetti 의 렌더 부분, 2026-10-27 함수로 분리).

    학습 체크포인트(main 의 --spaghetti)와 rollout npz 에서 다시 그리는 경우(src.viz.spaghetti_from_npz,
    예: 원본 env 오라클 mixture)가 같은 코드를 쓰도록 한 곳에 둔다.
    예전에는 scn.maxgeom(1000) 하나에 다 넣으려고 stride 를 키웠다 — 100 궤적이면 궤적당 ~8 점이라 선이
    아니라 점선이 됐다 (2026-10-24). 이제 trail_stride 그대로 전부 만들고 _MARKER_PASS_BUDGET 개씩 나눠
    여러 번 렌더한 뒤 **배경과 다른 픽셀만** 합성한다. floor 가 균일색(--clean_floor 권장)이면 diff 마스크가
    깨끗하다. endpoint 는 마지막 패스라 맨 위.

    underlay (2026-11-06, Fig 5a 데이터셋 coverage): [H, W] 또는 [H, W, 3] 곱셈 계수(0–1). 궤적 패스보다 **먼저**
    배경 프레임에 곱해진다 → 회색 영역 위에 궤적이 얹힌다 (궤적 픽셀은 패스 이미지에서 그대로 복사)."""
    colors = _spaghetti_colors(trajectories, color_by, cmap_name, solid_color)
    trail_markers, end_markers = [], []
    size_arr = np.full(3, trail_size, dtype=np.float64)
    endpoint_sz = np.full(3, trail_size * endpoint_scale, dtype=np.float64)
    for traj, (r, g, b) in zip(trajectories, colors):
        trail_rgba = np.array([r, g, b, trail_alpha], dtype=np.float64)
        for x, y in traj[::trail_stride]:
            trail_markers.append((np.array([x, y, 0.05]), size_arr, trail_rgba))
        ex, ey = traj[-1]
        end_markers.append((np.array([ex, ey, 0.10]), endpoint_sz,
                            np.array([r, g, b, 1.0], dtype=np.float64)))
    passes = [trail_markers[i:i + _MARKER_PASS_BUDGET]
              for i in range(0, len(trail_markers), _MARKER_PASS_BUDGET)]
    passes += [end_markers[i:i + _MARKER_PASS_BUDGET]
               for i in range(0, len(end_markers), _MARKER_PASS_BUDGET)]
    print(f"  {len(trail_markers)} trail + {len(end_markers)} endpoint markers "
          f"(stride {trail_stride}) → {len(passes)} render passes")

    # 배경 = reset_seed 로 reset 한 장면 (ant 1 마리가 원점 근처에 선다) + 축.
    env.seed(reset_seed)
    env.reset()

    def _render_with(markers, with_axes):
        viewer._markers.clear()
        if with_axes:
            _add_axis_markers(viewer, origin=(0.0, 0.0),
                              arrow_length=axis_length, label_axes=axis_label)
        for pos, size, rgba in markers:
            viewer.add_marker(pos=pos, type=mujoco_py.const.GEOM_SPHERE, size=size,
                              rgba=rgba, label="")
        return np.asarray(env.render(mode="rgb_array", width=width,
                                     height=height, camera_id=-1)).copy()

    background = _render_with([], with_axes=False)
    frame = _render_with([], with_axes=show_axes)
    if underlay is not None:
        u = np.asarray(underlay, dtype=np.float64)
        u = u[..., None] if u.ndim == 2 else u
        assert u.shape[:2] == frame.shape[:2], (u.shape, frame.shape)
        frame = np.clip(np.rint(frame * u), 0, 255).astype(frame.dtype)
    for k, pm in enumerate(passes):
        img = _render_with(pm, with_axes=False)
        assert img.shape == background.shape, (img.shape, background.shape)
        mask = np.abs(img.astype(np.int16) - background.astype(np.int16)).sum(-1) \
            > _COMPOSITE_DIFF_THRESH
        frame[mask] = img[mask]
        print(f"    pass {k + 1}/{len(passes)}: {len(pm)} markers, {int(mask.sum())} px")
    return frame


def render_frame(env, viewer, width, height, cam_mode, ant_xy,
                 trail_xys=None, trail_size=0.07,
                 trail_color=(1.0, 0.2, 0.2, 0.7),
                 show_axes=True, axis_length=3.0, axis_label=True):
    """Render one frame. In `track` mode, updates lookat to follow the ant.
    If trail_xys is provided, draws a sphere marker at each past position.
    If show_axes is True, draws a compass at the world origin with a red +x
    arrow and green +y arrow (red=X is the standard MuJoCo / robotics convention)."""
    if cam_mode == "track":
        viewer.cam.lookat[0] = float(ant_xy[0])
        viewer.cam.lookat[1] = float(ant_xy[1])

    # mujoco_py 2.1's MjRenderContextOffscreen does NOT clear `_markers` after
    # render. Without this manual reset, adding the cumulative trail every
    # frame queues markers quadratically and overflows scn.maxgeom (1000).
    # `_markers` is a readonly Cython attribute → mutate the list in place.
    viewer._markers.clear()

    if show_axes:
        _add_axis_markers(viewer, origin=(0.0, 0.0),
                          arrow_length=axis_length, label_axes=axis_label)

    if trail_xys:
        size_arr = np.full(3, trail_size, dtype=np.float64)
        rgba_arr = np.asarray(trail_color, dtype=np.float64)
        for x, y in trail_xys:
            viewer.add_marker(
                pos=np.array([float(x), float(y), 0.05]),
                type=mujoco_py.const.GEOM_SPHERE,
                size=size_arr,
                rgba=rgba_arr,
                label="",
            )

    return env.render(mode="rgb_array", width=width, height=height, camera_id=-1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run_dir",   required=True, type=str,
                   help="Path to a trained-policy run dir (contains config.json + model/).")
    p.add_argument("--out_mp4",   default="ant_topdown.mp4",
                   help="Output MP4 path.")
    p.add_argument("--num_steps", type=int,   default=None,
                   help="Max steps PER EPISODE (early-terminates on `done`). "
                        "Default: pick from the loaded run's cfg.max_seq_len so "
                        "T=100 policies render 100-step rollouts and T=500 policies "
                        "render 500-step rollouts. Override only when you want "
                        "to deliberately mismatch (e.g., render a T=100 policy "
                        "out to 500 steps to see what it does past its training horizon).")
    p.add_argument("--num_episodes", type=int, default=1,
                   help="How many episodes to render back-to-back into one MP4. "
                        "Each episode uses a different seed (--seed + episode_idx) so "
                        "trajectories differ. Trail resets between episodes. "
                        "Default 1 = single trajectory; set to e.g. 10 to see how the "
                        "policy varies across rollouts. evaluate.py uses 100, "
                        "so --num_episodes 100 reproduces all eval trajectories.")
    p.add_argument("--seed",      type=int,   default=0,
                   help="Base PRNG seed. Episode i uses seed + i.")
    p.add_argument("--width",     type=int,   default=640)
    p.add_argument("--height",    type=int,   default=480)
    p.add_argument("--fps",       type=int,   default=30)

    # Camera knobs
    p.add_argument("--cam_mode",  choices=("fixed", "track"), default="fixed",
                   help="fixed: stationary camera, ant moves through frame. "
                        "track: camera follows ant (gait analysis).")
    p.add_argument("--distance",  type=float, default=30.0,
                   help="Camera distance from lookat. Larger = wider field of view.")
    p.add_argument("--elevation", type=float, default=-89.0,
                   help="Camera elevation in degrees. -90 is straight down (singular).")
    p.add_argument("--azimuth",   type=float, default=90.0,
                   help="Camera azimuth in degrees. 90 (default) = world +x is image-right, "
                        "+y is image-up — canonical left-to-right reading orientation. "
                        "0 rotates 90° (world +x ends up image-UP, +y image-LEFT) so the "
                        "ant appears to run upward instead of rightward; rarely what you want.")
    p.add_argument("--lookat_x",  type=float, default=12.0,
                   help="World x-coord of camera target (image center). Default puts "
                        "ant start (0, 0) toward the bottom-left of the frame.")
    p.add_argument("--lookat_y",  type=float, default=9.0,
                   help="World y-coord of camera target (image center).")

    # Axis-compass knobs
    p.add_argument("--clean_floor", action="store_true",
                   help="Disable the black-and-white checker floor texture and replace "
                        "it with a soft uniform color. Highly recommended for wide "
                        "top-down views (e.g. --distance 100+) where the checker pattern "
                        "becomes a visually distracting grid against trail markers.")
    p.add_argument("--floor_color", type=float, nargs=3, default=[0.93, 0.93, 0.93],
                   metavar=("R", "G", "B"),
                   help="Floor RGB when --clean_floor is set. Default: (0.93, 0.93, 0.93) "
                        "— very light grey. Try (0.97, 0.94, 0.86) for warm beige, "
                        "(0.85, 0.90, 0.95) for cool blue-grey.")
    p.add_argument("--show_axes", default=True, action=argparse.BooleanOptionalAction,
                   help="Draw red +x arrow and green +y arrow at world origin.")
    p.add_argument("--axis_length", type=float, default=5.0,
                   help="Length of the +x / +y arrow markers in world units. "
                        "Tip spheres labeled '+X' / '+Y' sit at the ends.")
    p.add_argument("--axis_label", default=True, action=argparse.BooleanOptionalAction,
                   help="Show '+X' / '+Y' / 'O' text labels on the axis markers.")

    # Trail knobs
    p.add_argument("--trail", default=True, action=argparse.BooleanOptionalAction,
                   help="Draw breadcrumb sphere markers at past torso positions.")
    p.add_argument("--trail_stride", type=int, default=3,
                   help="Add a marker every N steps (1 = every step). Lower = denser trail.")
    p.add_argument("--trail_size",   type=float, default=0.15,
                   help="Sphere radius in world units.")
    p.add_argument("--trail_alpha",  type=float, default=0.85,
                   help="Trail color alpha.")

    # Spaghetti-plot knobs
    p.add_argument("--spaghetti_animated", action="store_true",
                   help="Like --spaghetti, but writes an MP4 where all N ants move "
                        "simultaneously through time, each as a colored sphere leaving a "
                        "PERSISTENT breadcrumb trail behind it (markers accumulate, never "
                        "fade). Honors --trail / --trail_stride / --trail_size / "
                        "--trail_alpha just like the single-ant mode. Trail stride is "
                        "auto-coarsened so total per-frame markers stay under MuJoCo's "
                        "1000-geom limit at end-of-video.")
    p.add_argument("--spaghetti", action="store_true",
                   help="Instead of an MP4, simulate --num_episodes rollouts and "
                        "save a single PNG with all trails overlaid (each trajectory "
                        "in a different color; brighter dot at each endpoint). Faster "
                        "than rendering full video — only the final frame is rasterized.")
    p.add_argument("--spaghetti_endpoint_scale", type=float, default=2.5,
                   help="Endpoint marker radius = trail_size × this. Bigger = more "
                        "obvious where each rollout ended. Default 2.5.")
    p.add_argument("--spaghetti_color_by", choices=("index", "heading"), default="index",
                   help="Static --spaghetti only: color trajectories by episode index (old "
                        "behaviour) or by final heading atan2(dy, dx) mapped over 0–90 deg "
                        "(vx-only → vy-only), which makes mixture modes visible.")
    p.add_argument("--spaghetti_save_npz", type=str, default=None,
                   help="Static --spaghetti only: also save the rollout torso xy as an npz "
                        "(xy [N,T,2] NaN-padded, lengths [N]) — the format src.viz.spaghetti_from_npz "
                        "reads, so the same rollouts can be redrawn (e.g. with a dataset-coverage layer) "
                        "without re-simulating (2026-11-06).")
    p.add_argument("--spaghetti_cmap", type=str, default="hsv",
                   help="matplotlib colormap name for per-trajectory colors. "
                        "'hsv' for max distinguishability, 'viridis' / 'plasma' for "
                        "a single hue gradient (calmer, harder to track individual lines).")
    # Timelapse knobs (2026-10-24)
    p.add_argument("--timelapse", action="store_true",
                   help="Simulate --num_episodes rollouts and, every --timelapse_every steps, render the "
                        "ant from straight above (world +x = image-right, +y = image-up) as an RGBA "
                        "sprite. Writes <out>.npz (trajectories + sprites in world metres); the figure "
                        "with x/y axes is drawn by experiments.moant.compose_dvm_timelapse. Ignores the "
                        "camera flags (fixed top-down sprite camera, src.viz.timelapse). Needs --clean_floor.")
    p.add_argument("--timelapse_every", type=int, default=50,
                   help="Steps between sprites (dt = 0.05 s per step).")
    p.add_argument("--timelapse_px_per_m", type=float, default=32.0,
                   help="Sprite resolution [px per m].")
    p.add_argument("--timelapse_tile", type=int, default=128,
                   help="Square sprite size [px]; must cover the ant (legs reach ~1.2 m).")
    args = p.parse_args()

    # ---- load policy + config ---------------------------------------------
    cfg = load_config(args.run_dir)
    # Default num_steps to the policy's own training horizon. The policy was
    # trained with a specific time-embedding T_max = cfg.max_seq_len, so rolling
    # out to a different length feeds out-of-distribution time-indices to the
    # network. Resolve here (post-cfg load, pre-rollout) so a T=100 policy gets
    # a 100-step render unless the user explicitly overrides --num_steps.
    if args.num_steps is None:
        args.num_steps = int(getattr(cfg, "max_seq_len", 500))
        print(f"--num_steps auto-set to cfg.max_seq_len={args.num_steps}")
    learner_type = getattr(cfg, "learner_type", "aet")  # legacy aet runs predate the flag
    if learner_type not in LEARNER_MODULES:
        raise ValueError(
            f"Unknown learner_type={learner_type!r} in config; "
            f"expected one of {list(LEARNER_MODULES)}"
        )
    _learner = LEARNER_MODULES[learner_type]
    train_state = load_model_cpu(
        os.path.abspath(os.path.join(args.run_dir, "model")),
        cfg,
        _learner.init_train_state,
    )
    policy, _, _ = _learner.get_model(train_state.policy_state)
    policy.eval()

    utility_F = cfg.utility_F
    u0 = np.asarray(
        utility_F(jnp.zeros((1, cfg.reward_dim), dtype=jnp.float32), keep_dims=True)
    ).squeeze(0)

    # Single dispatch point for "give me the policy distribution at this step".
    # Mirrors evaluate.py:select_action — fairdice is state-only,
    # aet / iql see (state, racc_Ftilde). Defining the closure here lets the
    # rollout loops below stay agnostic to which learner trained the checkpoint.
    _state_only = is_state_only_policy(cfg)   # evaluate.py 와 같은 판정 (SSOT)

    def pi_dist(s_b, raccs_raw, t):
        time_b = jnp.array([t])
        if _state_only:
            return policy(s_b, timesteps=time_b)
        racc_Ftilde_b = utility_F(jnp.asarray(raccs_raw)[None, :], keep_dims=True) - u0
        return policy(s_b, racc_Ftilde_b, timesteps=time_b)

    # 행동 선택을 jit 한다 (2026-10-24). 예전에는 매 env step 마다 정책을 eager 로 불러서
    # extraction=diffusion(스텝당 denoiser 20 회)은 24 에피소드 렌더에 1 시간이 넘었다.
    # evaluate.py 의 select_action 과 같은 식이다: sample → clip → action_scale.
    @jax.jit
    def select_action(s_b, raccs_raw, t, key):
        dist = pi_dist(s_b, raccs_raw, t)
        return jnp.clip(jnp.squeeze(dist.sample(seed=key)), -1.0, 1.0) * action_scale

    # Per-step reward-reshape closure. The policy (and racc bookkeeping below) was
    # trained in the buffer's reshaped space — for antv5 that's 5-dim, but the env
    # still returns the original 2-dim info['obj']. Apply the same reshape the
    # buffer used at load time so racc_raw accumulates in the same space the
    # policy expects, matching evaluate.py exactly.
    reward_reshape = getattr(cfg, "reward_reshape", "none")
    if reward_reshape != "none":
        if reward_reshape not in RESHAPE_STEP:
            raise ValueError(f"Unknown reward_reshape={reward_reshape!r} in cfg")
        _reshape_step = RESHAPE_STEP[reward_reshape]

        # Reshapes that need next_obs (e.g. walker2d_cd_safe, which reads
        # height/angle for the safety bowl) accept it as the third arg; others
        # ignore the kwarg. Always forward — the dispatch handles compatibility.
        def reshape_obj(action, info_obj, next_obs=None):
            return _reshape_step(action, info_obj, next_obs)
    else:
        def reshape_obj(action, info_obj, next_obs=None):
            return np.asarray(info_obj, dtype=np.float32)

    # ---- env + rendering --------------------------------------------------
    env = AntEnv()
    env.seed(args.seed)
    env.reset()       # seed first so the offscreen viewer is created against the seeded state

    # Optional clean floor: replace the default black-and-white checker (MatPlane
    # → texplane) with a flat uniform color. Setting `geom_matid = -1` detaches
    # the floor from any material so the renderer falls back to its `geom_rgba`.
    # This is a render-time-only override on the model — physics is unaffected
    # because contact/dynamics don't read material visuals. Must happen AFTER
    # env.reset() (which builds the sim) but BEFORE the first render call (which
    # caches the scene).
    if args.clean_floor:
        apply_clean_floor(env, args.floor_color)
        print(f"  Clean floor: matid disabled, rgb={tuple(round(c, 2) for c in args.floor_color)}")

    if args.timelapse:   # 첫 render(컨텍스트 생성) 전에 — 이유는 timelapse.SHADOW_EXTENT_M 주석
        prepare_model_for_sprites(env)

    viewer = configure_topdown_camera(
        env, args.width, args.height,
        args.distance, args.elevation, args.azimuth,
        args.lookat_x, args.lookat_y,
    )

    # Normalization constants — match evaluate.py exactly so the
    # policy sees the same conditioning distribution it was trained on.
    reward_min     = np.asarray(cfg.reward_min, dtype=np.float32)
    reward_max     = np.asarray(cfg.reward_max, dtype=np.float32)
    reward_span    = reward_max - reward_min
    reward_scaling = np.asarray(cfg.reward_scaling_factor, dtype=np.float32)
    state_mean     = jnp.array(cfg.state_mean)
    state_std      = jnp.array(cfg.state_std)
    action_scale   = jnp.asarray(
        getattr(cfg, "action_scale", np.ones(cfg.action_dim, dtype=np.float32)),
        dtype=jnp.float32,
    )

    frames      = []
    trail_color = (1.0, 0.25, 0.25, args.trail_alpha)

    def simulate_episode(ep_seed, record_states=False):
        """렌더 없이 한 에피소드를 굴린다 — spaghetti 두 모드와 timelapse 가 같은 rollout 을 쓰도록 한 곳에
        둔다 (같은 ep_seed → 같은 궤적; 정적 spaghetti 그림과 timelapse 가 같은 에피소드를 가리킨다).
        반환: (스텝 전 torso xy 목록, 스텝 전 MjSimState 목록 또는 None, raw 목적 합 info['obj'])."""
        env.seed(ep_seed)
        obs = env.reset()
        key = jax.random.PRNGKey(ep_seed)
        raccs_raw = np.zeros((cfg.reward_dim,), dtype=np.float32)
        raw_return = None
        traj, states = [], ([] if record_states else None)
        for t in range(args.num_steps):
            traj.append(tuple(env.get_body_com("torso")[:2].tolist()))
            if record_states:
                states.append(env.sim.get_state())   # qpos/qvel 사본
            key, subkey = jax.random.split(key)
            s_b = ((jnp.asarray(obs, dtype=jnp.float32) - state_mean) / state_std)[None, :]
            a = select_action(s_b, jnp.asarray(raccs_raw), jnp.int32(t), subkey)
            obs, _, done, info = env.step(np.asarray(a))
            raw_r = reshape_obj(np.asarray(a), info["obj"], obs)   # post-step obs = next_obs
            raccs_raw += reward_scaling * (raw_r - reward_min) / reward_span
            obj = np.asarray(info["obj"], dtype=np.float64)
            raw_return = obj.copy() if raw_return is None else raw_return + obj
            if done:
                break
        return traj, states, raw_return

    # ============================================================
    # Spaghetti ANIMATED mode: simulate N episodes silently, render
    # an MP4 where all N ants move simultaneously, each leaving a
    # PERSISTENT breadcrumb trail behind it (markers accumulate over
    # time, never fade out).
    # ============================================================
    if args.spaghetti_animated:
        import matplotlib.cm as _cm

        # 1) Pre-simulate all episodes (no per-step rendering — fast).
        trajectories = []
        print(f"Spaghetti animated: simulating {args.num_episodes} episodes "
              f"(up to {args.num_steps} steps each) ...")
        for ep_idx in range(args.num_episodes):
            ep_seed = args.seed + ep_idx
            traj, _, _ = simulate_episode(ep_seed)
            trajectories.append(traj)
            ex, ey = traj[-1]
            print(f"  Ep {ep_idx + 1:>3}/{args.num_episodes} (seed={ep_seed}): "
                  f"{len(traj):>3} steps, end (x, y) = ({ex:7.2f}, {ey:7.2f})")

        # 2) Per-ant color (same hue assignment as the static --spaghetti mode,
        #    so a static PNG and the animated MP4 of the same run match colors).
        N = len(trajectories)
        max_T = max(len(t) for t in trajectories) if trajectories else 0
        cmap = _cm.get_cmap(args.spaghetti_cmap)
        head_colors = []
        for ep_idx in range(N):
            t = ep_idx / max(1, N - 1) if N > 1 else 0.0
            r, g, b, _ = cmap(t)
            head_colors.append((r, g, b))

        # 3) Auto-coarsen breadcrumb stride so per-frame marker count stays under
        #    MuJoCo's 1000-geom budget at END-OF-VIDEO. At frame_t, each ant has
        #    ceil((cur_t + 1) / stride) breadcrumbs + 1 head. Worst case is
        #    frame_t = max_T-1 with all N ants alive: total ≈ N + N × max_T/stride.
        #    Solve for stride: stride ≥ N × max_T / (BUDGET − N), with BUDGET ≈ 950
        #    (leaving headroom for axes / scene geoms).
        BUDGET = 950
        if N >= BUDGET:
            raise ValueError(
                f"--num_episodes={N} ≥ {BUDGET}: can't fit even N head markers in MuJoCo's "
                f"1000-geom limit. Reduce --num_episodes."
            )
        min_stride = max(1, int(np.ceil(N * max_T / (BUDGET - N))))
        stride = max(args.trail_stride, min_stride)
        n_breadcrumbs_end = (max_T + stride - 1) // stride       # per-ant, at last frame
        total_at_end = N + N * n_breadcrumbs_end                 # heads + breadcrumbs
        if stride > args.trail_stride:
            print(f"  Auto-coarsened breadcrumb stride: {stride} (requested {args.trail_stride}); "
                  f"keeps {total_at_end} markers ≤ {BUDGET} at last frame "
                  f"(N={N}, max_T={max_T})")
        else:
            print(f"  Breadcrumb stride: {stride} → {n_breadcrumbs_end} breadcrumbs/ant at "
                  f"last frame, {total_at_end} markers total")

        head_size  = np.full(3, args.trail_size * args.spaghetti_endpoint_scale,
                             dtype=np.float64)
        trail_size = np.full(3, args.trail_size, dtype=np.float64)

        # 4) Render frame-by-frame. At frame_t, each ant's marker stack is:
        #     - persistent breadcrumbs at every traj[i][k] for k ∈ {0, stride, 2·stride, ..., cur_t}
        #     - one head marker (bigger, opaque) at traj[i][cur_t]
        #    Markers persist visually only because we re-add them every frame after
        #    viewer._markers.clear() — MuJoCo doesn't keep marker state across frames.
        #    Honors --trail (default true): if disabled, only the head sphere is shown.
        print(f"  Rendering {max_T} frames at {args.fps} fps "
              f"(~{max_T / args.fps:.1f}s of video)...")
        frames = []
        for frame_t in range(max_T):
            viewer._markers.clear()
            if args.show_axes:
                _add_axis_markers(viewer, origin=(0.0, 0.0),
                                  arrow_length=args.axis_length,
                                  label_axes=args.axis_label)
            for ep_idx, traj in enumerate(trajectories):
                # If this trajectory ended early, freeze the ant at its last position
                # for the rest of the video instead of removing it abruptly.
                cur_t = min(frame_t, len(traj) - 1)
                if cur_t < 0:
                    continue
                r, g, b = head_colors[ep_idx]
                trail_rgba = np.array([r, g, b, args.trail_alpha], dtype=np.float64)
                head_rgba  = np.array([r, g, b, 1.0],              dtype=np.float64)
                # Persistent breadcrumbs: every `stride`-th past position up to (but
                # not including) cur_t. The head marker covers the current position.
                if args.trail:
                    for ti in range(0, cur_t, stride):
                        bx, by = traj[ti]
                        viewer.add_marker(
                            pos=np.array([bx, by, 0.05]),
                            type=mujoco_py.const.GEOM_SPHERE,
                            size=trail_size,
                            rgba=trail_rgba,
                            label="",
                        )
                # Head: current position, fully opaque, slightly larger.
                hx, hy = traj[cur_t]
                viewer.add_marker(
                    pos=np.array([hx, hy, 0.10]),
                    type=mujoco_py.const.GEOM_SPHERE,
                    size=head_size,
                    rgba=head_rgba,
                    label="",
                )
            frame = env.render(mode="rgb_array",
                               width=args.width, height=args.height, camera_id=-1)
            frames.append(frame)
            if (frame_t + 1) % 100 == 0:
                print(f"    rendered {frame_t + 1}/{max_T} frames")

        # 5) Save MP4.
        out_path = args.out_mp4
        if out_path.lower().endswith(".png"):
            out_path = out_path[:-4] + ".mp4"
        elif not out_path.lower().endswith(".mp4"):
            out_path = out_path + ".mp4"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        imageio.mimwrite(out_path, frames, fps=args.fps,
                         macro_block_size=1, codec="libx264")
        print(f"\nSaved spaghetti animated MP4: {out_path}  "
              f"({N} ants, {max_T} frames @ {args.fps} fps, "
              f"{args.width}×{args.height})")
        return

    # ============================================================
    # Spaghetti mode: simulate N episodes silently, render ONE image
    # ============================================================
    if args.spaghetti:

        # 1) Simulate all episodes WITHOUT per-step rendering.
        trajectories = []  # one list of (x, y) per episode
        print(f"Spaghetti: simulating {args.num_episodes} episodes "
              f"(up to {args.num_steps} steps each) without per-step rendering ...")
        for ep_idx in range(args.num_episodes):
            ep_seed = args.seed + ep_idx
            traj, _, _ = simulate_episode(ep_seed)
            trajectories.append(traj)
            ex, ey = traj[-1]
            print(f"  Ep {ep_idx + 1:>3}/{args.num_episodes} (seed={ep_seed}): "
                  f"{len(traj):>3} steps, end (x, y) = ({ex:7.2f}, {ey:7.2f})")

        if args.spaghetti_save_npz:
            L = np.array([len(t) for t in trajectories], dtype=np.int64)
            xy = np.full((len(trajectories), int(L.max()), 2), np.nan)
            for i, t in enumerate(trajectories):
                xy[i, :L[i]] = np.asarray(t)
            os.makedirs(os.path.dirname(os.path.abspath(args.spaghetti_save_npz)) or ".", exist_ok=True)
            np.savez(args.spaghetti_save_npz, xy=xy, lengths=L,
                     seeds=args.seed + np.arange(len(trajectories)))
            print(f"  Saved rollout xy: {args.spaghetti_save_npz}")

        frame = spaghetti_frame(
            env, viewer, trajectories, width=args.width, height=args.height,
            color_by=args.spaghetti_color_by, cmap_name=args.spaghetti_cmap,
            trail_size=args.trail_size, trail_stride=args.trail_stride, trail_alpha=args.trail_alpha,
            endpoint_scale=args.spaghetti_endpoint_scale, show_axes=args.show_axes,
            axis_length=args.axis_length, axis_label=args.axis_label, reset_seed=args.seed)

        # 3) Save as PNG (derive path from --out_mp4).
        out_path = args.out_mp4
        if out_path.lower().endswith(".mp4"):
            out_path = out_path[:-4] + ".png"
        elif not out_path.lower().endswith(".png"):
            out_path = out_path + ".png"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        imageio.imwrite(out_path, frame)
        print(f"\nSaved spaghetti PNG: {out_path}  "
              f"({args.num_episodes} trajectories, {args.width}×{args.height})")
        return

    # ============================================================
    # Timelapse mode (2026-10-24): 위에서 본 ant 스프라이트 + 궤적 → npz (그림은 compose_dvm_timelapse)
    # ============================================================
    if args.timelapse:
        if not args.clean_floor:
            raise ValueError("--timelapse needs --clean_floor: the sprite mask is a diff against a "
                             "floor-only tile, and the checker texture shifts with every lookat.")
        episodes = (simulate_episode(args.seed + i, record_states=True)
                    for i in range(args.num_episodes))
        export_timelapse(env, viewer, episodes, args.num_episodes, args.num_steps,
                         os.path.splitext(args.out_mp4)[0] + ".npz", every=args.timelapse_every,
                         tile=args.timelapse_tile, ppm=args.timelapse_px_per_m,
                         extra=dict(run_dir=np.str_(args.run_dir), seed=np.int32(args.seed)))
        return

    # ============================================================
    # Multi-episode video mode (default)
    # ============================================================
    print(f"Recording {args.num_episodes} episode(s) × up to {args.num_steps} steps "
          f"from {args.run_dir} ...")
    for ep_idx in range(args.num_episodes):
        ep_seed = args.seed + ep_idx
        env.seed(ep_seed)               # different env stochasticity per episode
        obs = env.reset()
        key = jax.random.PRNGKey(ep_seed)   # different action-sampling rng per episode

        # Per-episode state — racc and trail reset every episode.
        raccs_raw   = np.zeros((cfg.reward_dim,), dtype=np.float32)
        raw_returns = np.zeros((cfg.reward_dim,), dtype=np.float64)
        trail_xys   = []
        ep_frames   = 0
        done        = False

        for t in range(args.num_steps):
            torso_xy = env.get_body_com("torso")[:2]
            if args.trail and t % args.trail_stride == 0:
                trail_xys.append((float(torso_xy[0]), float(torso_xy[1])))

            # Render BEFORE stepping so frame[0] of each episode is its initial state.
            frames.append(render_frame(
                env, viewer, args.width, args.height,
                args.cam_mode, torso_xy,
                trail_xys=trail_xys if args.trail else None,
                trail_size=args.trail_size,
                trail_color=trail_color,
                show_axes=args.show_axes,
                axis_length=args.axis_length,
                axis_label=args.axis_label,
            ))
            ep_frames += 1

            key, subkey = jax.random.split(key)
            s_b  = ((jnp.asarray(obs, dtype=jnp.float32) - state_mean) / state_std)[None, :]
            a    = select_action(s_b, jnp.asarray(raccs_raw), jnp.int32(t), subkey)

            obs, _, done, info = env.step(np.asarray(a))
            # Post-step `obs` IS next_obs from the env's perspective; pass it as
            # the third arg so reshapes that need it (walker2d_cd_safe) work.
            raw_r        = reshape_obj(np.asarray(a), info["obj"], obs)
            raccs_raw   += reward_scaling * (raw_r - reward_min) / reward_span
            raw_returns += raw_r
            if done:
                break

        # Per-episode summary — handy for picking next --distance / --lookat.
        final_xy = env.get_body_com("torso")[:2]
        early   = " (terminated early)" if done else ""
        print(f"  Ep {ep_idx + 1:>3}/{args.num_episodes} (seed={ep_seed}): "
              f"{ep_frames} frames{early}, "
              f"end (x, y) = ({final_xy[0]:7.2f}, {final_xy[1]:7.2f}), "
              f"raw_returns = [{raw_returns[0]:8.2f}, {raw_returns[1]:8.2f}]")

    # ---- write MP4 --------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out_mp4)) or ".", exist_ok=True)
    imageio.mimwrite(
        args.out_mp4, frames, fps=args.fps, codec="libx264", quality=8,
    )
    print(f"\nSaved MP4: {args.out_mp4}  "
          f"({len(frames)} total frames across {args.num_episodes} episode(s) @ {args.fps} fps)")


if __name__ == "__main__":
    main()
