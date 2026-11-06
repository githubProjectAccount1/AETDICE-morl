"""rollout npz(timelapse 산출물)의 궤적으로 MuJoCo topdown spaghetti PNG 를 그린다 (2026-10-27).

왜: 원본 env 미세조정 오라클은 학습 체크포인트(--run_dir)가 아니라 정책 mixture 라 render_topdown --spaghetti 로
굴릴 수 없다. 대신 그 오라클의 rollout 은 timelapse npz(xy, lengths)에 이미 있다. 같은 이유로 다른 방법들도
timelapse npz 에서 그리면 **모든 패널이 같은 rollout 파일·같은 카메라**를 쓴다 (다시 굴리지 않는다).
렌더 자체는 src.viz.render_topdown.spaghetti_frame (SSOT — --spaghetti 와 같은 코드).
render_topdown --spaghetti --spaghetti_save_npz 가 남긴 npz 도 같은 형식(xy, lengths)이다.

사용: source scripts/aetenv.sh && JAX_PLATFORMS=cpu \
      python -m src.viz.spaghetti_from_npz --npz video/dvm/timelapse_median/diffusion_s1.npz \
             --out video/dvm/spaghetti100w_diffusion_s1.png --box -6 152

축 맞춤: src.viz.multiant_video 와 같은 규칙 — 바로 위 카메라, lookat = 상자 중심, distance = 반폭/tan(fovy/2) 이라
지면 상자 [x0, x1]² 가 PNG 를 정확히 채운다 (합성 그림이 이 extent 로 미터 눈금을 단다).
→ 픽셀 (row, col) ↔ 월드 (x, y): col = (x - x0)/(x1 - x0)·W, row = (x1 - y)/(x1 - x0)·H (world_to_px).

Fig 5a 확장 (2026-11-06, 사용자 요청 "어느 방향이 objective 몇 번인지 + dataset coverage 를 연한 회색으로 먼저"):
  --coverage_npz  데이터셋 경로(src.viz.moant_dataset_xy 산출)의 점유 영역을 연한 회색으로 바닥에 먼저 칠하고
                  (spaghetti_frame underlay) 그 위에 rollout 궤적을 얹는다.
  --obj_axes      원점에서 +x("obj 0") / +y("obj 1") 화살표 (matplotlib, 위 픽셀↔월드 식으로 정확히 배치).
  --crop          최종 PNG 를 픽셀 [top:bottom, left:right] 로 자른다 (논문 패널과 같은 잘라내기 재현용).
"""
import argparse
import os

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy import ndimage

from src.viz.multiant_video import topdown_camera_for_box
from src.viz.render_topdown import AntEnv, configure_topdown_camera, spaghetti_frame
from src.viz.timelapse import apply_clean_floor

# render_dvm_spaghetti.sh 와 같은 marker 설정 (그 그림과 같은 모양이 되도록).
_TRAIL_SIZE, _TRAIL_STRIDE, _TRAIL_ALPHA, _ENDPOINT_SCALE = 0.35, 2, 1.0, 3.0
_FLOOR_RGB = (0.93, 0.93, 0.93)          # = render_topdown --floor_color 기본값

# 데이터셋 coverage 층. 점유 = 픽셀에 떨어진 데이터셋 경로 점 수 ≥ _COV_MIN_HITS (경로 몇 개만 지나는 외톨이
# 픽셀은 영역에서 뺀다), 그 뒤 closing·구멍 메우기로 연속 영역, 가우시안으로 경계만 부드럽게.
_COV_MIN_HITS = 5
_COV_CLOSE_PX = 4
_COV_EDGE_SIGMA_PX = 2.0
_COV_SHADE = 0.85                        # 바닥 RGB 에 곱하는 계수 (흰 바닥 255 → 연한 회색 ~217)

# obj 축 화살표 (월드 미터 / 픽셀 / pt).
_AXIS_LEN_M = 22.0
_AXIS_RGB = (0.12, 0.12, 0.12)
_AXIS_LW_PT = 4.0
_AXIS_HEAD_PT = 26
_AXIS_FONT_PT = 30
_AXIS_LABEL_GAP_M = 1.5
# "obj 1" 은 +y 화살표 끝 위, 왼쪽 끝을 원점보다 약간 왼쪽에 맞춘다 (논문 crop 은 원점 왼쪽 여백이 ~3 m 뿐이라
# 가운데 정렬하면 잘린다). 궤적과 겹치는 부분이 읽히도록 글자에 흰 테두리.
_AXIS_Y_LABEL_DX_M = -2.5
_AXIS_TEXT_HALO_PT = 4.0
_DPI = 100
# 2026-11-06 사용자: "x랑 y축 (obj 1이랑 obj0이 좀 더 길게 나와있는게 좋을것같은데. 마치 좌표축처럼" / "격자도 있으면
#   좋을듯. 아주 얇은 실선으로 격자 무늬 (너무 촘촘하지 않게)". 긴 축은 라벨을 화살표 끝 안쪽에 둔다 (crop 아래·왼쪽
#   여백이 ~3 m 라 바깥에 두면 잘린다). 격자는 underlay 에 곱해 넣어 궤적 아래에 깔린다.
_GRID_SHADE = 0.80                       # 격자선 픽셀에 곱하는 계수 (회색 위에서도 보이게)
_GRID_LINE_PX = 1                        # 격자선 두께 (픽셀)
_LONG_LABEL_INSET_M = 1.5


def world_to_px(x, y, box, size):
    """box 카메라 화면의 연속 픽셀 좌표 (col, row) — 픽셀 중심이 +0.5."""
    x0, x1 = box
    s = size / (x1 - x0)
    return (np.asarray(x) - x0) * s, (x1 - np.asarray(y)) * s


def coverage_mask(xy, lengths, box, size):
    """[N, T, 2] 데이터셋 경로 → [size, size] 0–1 점유 마스크 (경계만 부드러움)."""
    assert xy.ndim == 3 and xy.shape[2] == 2 and xy.shape[0] == len(lengths), xy.shape
    valid = np.arange(xy.shape[1])[None, :] < np.asarray(lengths)[:, None]
    pts = xy[valid]
    col, row = world_to_px(pts[:, 0], pts[:, 1], box, size)
    c, r = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
    inside = (c >= 0) & (c < size) & (r >= 0) & (r < size)
    hits = np.bincount(r[inside] * size + c[inside], minlength=size * size).reshape(size, size)
    occ = hits >= _COV_MIN_HITS
    disk = np.hypot(*np.mgrid[-_COV_CLOSE_PX:_COV_CLOSE_PX + 1, -_COV_CLOSE_PX:_COV_CLOSE_PX + 1]) \
        <= _COV_CLOSE_PX
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, structure=disk))
    print(f"  coverage: {int(inside.sum())} dataset points in view, {int(occ.sum())} px occupied")
    return np.clip(ndimage.gaussian_filter(occ.astype(np.float64), _COV_EDGE_SIGMA_PX), 0.0, 1.0)


def grid_underlay(box, size, spacing_m):
    """[size, size] 곱셈 층 — 월드 좌표 spacing_m 간격 (0 은 축이 그리므로 제외) 얇은 격자선."""
    g = np.ones((size, size))
    x0, x1 = box
    for v in np.arange(spacing_m, x1, spacing_m):
        c, r = world_to_px(v, v, box, size)
        ci, ri = int(np.floor(c)), int(np.floor(r))
        if 0 <= ci < size:
            g[:, ci:ci + _GRID_LINE_PX] = _GRID_SHADE
        if 0 <= ri < size:
            g[ri:ri + _GRID_LINE_PX, :] = _GRID_SHADE
    return g


def draw_obj_axes(frame, box, axis_len_m=_AXIS_LEN_M):
    """원점에서 +x = obj 0, +y = obj 1 화살표를 frame 위에 그린다 (축 제외 픽셀은 그대로).
    axis_len_m 이 기본값보다 길면 '좌표축' 모양 — 라벨을 화살표 끝 안쪽 (축 위·오른쪽) 에 둔다."""
    H, W = frame.shape[:2]
    assert H == W, frame.shape
    x0, x1 = box
    fig = plt.figure(figsize=(W / _DPI, H / _DPI), dpi=_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame, extent=(x0, x1, x0, x1), interpolation="none")
    ax.set_xlim(x0, x1)
    ax.set_ylim(x0, x1)
    ax.axis("off")
    arrow = dict(arrowstyle=f"-|>,head_length={_AXIS_HEAD_PT / 40:.2f},head_width={_AXIS_HEAD_PT / 80:.2f}",
                 color=_AXIS_RGB, lw=_AXIS_LW_PT, shrinkA=0, shrinkB=0)
    text = dict(fontsize=_AXIS_FONT_PT, color=_AXIS_RGB, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=_AXIS_TEXT_HALO_PT, foreground="white")])
    L = float(axis_len_m)
    ax.annotate("", xy=(L, 0.0), xytext=(0.0, 0.0), arrowprops=arrow)
    ax.annotate("", xy=(0.0, L), xytext=(0.0, 0.0), arrowprops=arrow)
    if L <= _AXIS_LEN_M:
        ax.text(L + _AXIS_LABEL_GAP_M, 0.0, "obj 0", ha="left", va="center", **text)
        ax.text(_AXIS_Y_LABEL_DX_M, L + _AXIS_LABEL_GAP_M, "obj 1", ha="left", va="bottom", **text)
    else:
        ax.text(L, _LONG_LABEL_INSET_M, "obj 0", ha="right", va="bottom", **text)
        ax.text(_LONG_LABEL_INSET_M, L, "obj 1", ha="left", va="top", **text)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    assert out.shape == frame.shape, (out.shape, frame.shape)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=float, nargs=2, required=True, help="x, y 공통 범위 [m] (정사각)")
    ap.add_argument("--size", type=int, default=1100)
    ap.add_argument("--coverage_npz", default=None,
                    help="데이터셋 경로 npz (xy, lengths; src.viz.moant_dataset_xy) — 연한 회색 점유 영역 층")
    ap.add_argument("--coverage_max_len", type=int, default=None,
                    help="데이터셋 경로를 앞 N 점으로 자른다 (rollout 지평선과 맞출 때)")
    ap.add_argument("--obj_axes", action="store_true", help="원점에 obj 0 (+x) / obj 1 (+y) 화살표")
    ap.add_argument("--solid_color", default=None,
                    help="궤적을 한 색으로 (matplotlib 색, 예 '#1f77b4'); 없으면 진행방향 turbo 그라데이션")
    ap.add_argument("--axis_len", type=float, default=_AXIS_LEN_M, help="obj 축 화살표 길이 [m]")
    ap.add_argument("--grid", type=float, default=None, help="격자 간격 [m] (없으면 격자 없음)")
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
                    help="최종 PNG 픽셀 잘라내기 [top:bottom, left:right]")
    args = ap.parse_args()

    z = np.load(args.npz)
    xy, lengths = z["xy"], z["lengths"]
    trajectories = [[tuple(p) for p in xy[i, :int(lengths[i])].tolist()] for i in range(len(lengths))]
    assert all(len(t) > 0 for t in trajectories)

    underlay = None
    if args.coverage_npz:
        c = np.load(args.coverage_npz)
        cl = c["lengths"] if args.coverage_max_len is None \
            else np.minimum(c["lengths"], args.coverage_max_len)
        cov = coverage_mask(c["xy"], cl, args.box, args.size)
        underlay = 1.0 - (1.0 - _COV_SHADE) * cov
    if args.grid:
        g = grid_underlay(args.box, args.size, args.grid)
        underlay = g if underlay is None else underlay * g

    env = AntEnv()
    env.seed(0)
    env.reset()
    apply_clean_floor(env, _FLOOR_RGB)
    center, distance, elevation, azimuth = topdown_camera_for_box(
        args.box, float(env.sim.model.vis.global_.fovy))
    viewer = configure_topdown_camera(env, args.size, args.size, distance, elevation, azimuth,
                                      center, center)
    frame = spaghetti_frame(env, viewer, trajectories, width=args.size, height=args.size,
                            color_by="solid" if args.solid_color else "heading", cmap_name="turbo",
                            solid_color=args.solid_color, trail_size=_TRAIL_SIZE,
                            trail_stride=_TRAIL_STRIDE, trail_alpha=_TRAIL_ALPHA,
                            endpoint_scale=_ENDPOINT_SCALE, show_axes=False, axis_length=5.0,
                            axis_label=True, reset_seed=0, underlay=underlay)
    if args.obj_axes:
        frame = draw_obj_axes(frame, args.box, args.axis_len)
    if args.crop:
        t, b, l, r = args.crop
        frame = frame[t:b, l:r]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    imageio.imwrite(args.out, frame)
    print(f"Saved {args.out}  ({len(trajectories)} trajectories, {args.size}px, shape {frame.shape})")


if __name__ == "__main__":
    main()
