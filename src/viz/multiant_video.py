"""여러 에피소드의 Ant 를 한 장면에 동시에 + 궤적 꼬리 — 바로 위에서 본 영상 (2026-10-27 사용자 요청).

사용자 (2026-10-27): "비디오는 탑 다운 뷰로, 버블 느낌으로 하니까 잘 안 보여 — 그냥 원래 ant 처럼 하고 배경도
그냥 놔둬. axis 만 잘 맞춰서 각 알고리즘별로 다 따로 비디오. 3 개 한꺼번에 붙이지 마."
이어서: "궤적은 보이도록, trailing 같은 이펙트는 있어야 할 듯. 100 개 궤적 동시에. mujoco 환경 그대로, 1 개씩."

방법: ant.xml 의 torso 몸체를 N 번 복제한 **렌더 전용 모델**을 만들고(이름에 _i 접미사, actuator 제거), 저장된
rollout 의 스텝별 qpos(src.viz.timelapse.export_timelapse 가 npz 에 남김)를 ant 마다 그대로 넣어 mj_forward 후
렌더한다. 물리를 다시 굴리지 않으므로 그림·표의 rollout 과 **같은 에피소드**다. ant 몸체·바닥(checker)·조명은
MuJoCo 기본 그대로이고 marker(구) 는 쓰지 않는다. 100 마리 = geom 1301 개라 mujoco-py 의 고정 maxgeom(1000) 을
넘는다 → 렌더 컨텍스트의 장면을 더 크게 다시 만든다 (_MAXGEOM).

궤적 꼬리: MuJoCo 프레임 위에 matplotlib 선으로 그린다 (marker 로 그리면 geom 이 수만 개). 층 순서는
바닥 → 궤적 → ant 다. ant 가 없는 바닥 프레임(_HIDE_XY 로 치운 상태)을 한 번 렌더해 두고, 매 프레임 그와 다른 픽셀을
ant 층(RGBA)으로 떼어 궤적 위에 다시 얹는다 (render_topdown.spaghetti_frame 의 배경 diff 합성과 같은 생각).
색 = 그 에피소드의 최종 heading (turbo, 0° vx → 90° vy; 그림들과 같은 규칙) — 끝 방향을 미리 알려 주는 색이다.
지난 경로 전체는 얇고 반투명하게, 최근 _TAIL_STEPS 스텝은 굵게(회색 테두리로 checker 흑·백 칸 모두에서 보이게)
뒤로 갈수록 옅어지게 그린다.

축 맞춤: 카메라는 바로 위(azimuth 90 → 화면 오른쪽 = 월드 +x, 화면 위 = 월드 +y), lookat = 상자 중심,
distance = 반폭 / tan(fovy/2) — 지면(z=0)의 정사각 상자 [x0, x1]×[y0, y1] 이 화면을 정확히 채운다. 그 화면을
matplotlib 축(같은 extent, 미터 눈금)에 그대로 붙인다. 모든 영상이 같은 --box 를 쓰면 축이 같다.
렌더는 CPU(mujoco-py offscreen) — 2048 px 에서 프레임당 ~10 s.

사용: source scripts/aetenv.sh && JAX_PLATFORMS=cpu python -m src.viz.multiant_video \
      --npz video/dvm/timelapse_median/diffusion_s1.npz --out video/dvm/topdown_trails_diffusion_s1.mp4 \
      --title "Diffusion . seed 1" --box -6 152
"""
import argparse
import copy
import os
import xml.etree.ElementTree as ET

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from src.viz.render_topdown import _COMPOSITE_DIFF_THRESH, _heading_unit, mujoco_py   # 재실행 처리도 그쪽 (SSOT)

_ANT_XML = os.path.join(os.path.dirname(__file__), "..", "..", "environment", "assets", "ant.xml")
_NQ_ANT = 15                     # free joint 7 + hinge 8
_AZIMUTH, _ELEVATION = 90.0, -89.99
_FPS = 30
_TICK_M = 20.0
_SHADOWSIZE = 8192
# 그림자 맵은 stat.extent 와 light 위치로 덮는 범위가 정해진다 — 상자 전체를 덮도록 키운다.
_SHADOW_PAD = 1.2
_MAXGEOM = 4000                  # ant 1 마리 13 geom → 100 마리 1300 + 바닥
_HIDE_XY = 1000.0                # 바닥만 렌더할 때 ant 를 치울 곳 (바닥 반폭 300 m 밖, 화면 밖)
_CMAP = "turbo"
_TAIL_STEPS = 60                 # 굵게 그리는 최근 구간 (= 3 s; dt 0.05)
_PATH_LW, _PATH_ALPHA = 0.9, 0.45
_TAIL_LW, _TAIL_HALO_LW = 2.4, 4.2
_TAIL_ALPHA_MIN = 0.35           # 꼬리의 가장 오래된 끝 alpha (가장 최근 = 1)
_HALO_RGBA = (0.45, 0.45, 0.45, 0.85)
_DPI = 100
_MARGIN_PX = dict(left=110, right=40, bottom=90, top=110)


def multi_ant_xml(n):
    """ant.xml 의 torso 몸체를 n 번 복제한 렌더 전용 MJCF 문자열. 액추에이터·추적 카메라는 뺀다."""
    root = ET.parse(os.path.abspath(_ANT_XML)).getroot()
    world = root.find("worldbody")
    torso = world.find("body[@name='torso']")
    world.remove(torso)
    for cam in torso.findall("camera"):
        torso.remove(cam)
    for act in root.findall("actuator"):
        root.remove(act)
    # 렌더 전용: 저장된 qpos 를 넣고 forward 만 한다 → 접촉·관절 한계 제약은 필요 없고, 출발점에 겹친 100 마리의
    # 접촉이 njmax(500) 를 넘겨 forward 가 실패한다 (2026-10-27).
    ET.SubElement(root.find("option"), "flag", contact="disable", limit="disable")
    for i in range(n):
        t = copy.deepcopy(torso)
        for el in t.iter():
            if "name" in el.attrib:
                el.attrib["name"] = f"{el.attrib['name']}_{i}"
        world.append(t)
    return ET.tostring(root, encoding="unicode")


def topdown_camera_for_box(box, fovy_deg):
    """지면 정사각 상자 [x0, x1]² 가 정사각 화면을 정확히 채우는 바로 위 카메라 → (center, distance, elev, azim).
    spaghetti_from_npz 도 같은 함수를 쓴다 (두 산출물의 축 규칙 SSOT)."""
    x0, x1 = box
    assert x1 > x0, box
    half = (x1 - x0) / 2.0
    return (x0 + x1) / 2.0, half / np.tan(np.radians(fovy_deg) / 2.0), _ELEVATION, _AZIMUTH


def load_rollouts(npz, n):
    """(qpos [n,T,15], lengths [n], 최종 heading [n] deg) — 앞 n 에피소드."""
    z = np.load(npz)
    assert "qpos" in z.files, f"{npz} 에 qpos 가 없다 — timelapse 를 다시 뽑아라 (2026-10-27 이후 export)"
    assert z["qpos"].shape[0] >= n, (z["qpos"].shape, n)
    q, L, h = z["qpos"][:n], z["lengths"][:n].astype(int), z["headings_deg"][:n]
    assert q.shape[2] == _NQ_ANT and L.min() >= 1, (q.shape, L.min())
    return q, L, h


def _trail_segments(xy, L, t):
    """시각 t 까지의 선분 [(K,2,2)] 과 각 선분의 (에피소드, 나이) — 일찍 끝난 에피소드는 끝에서 멈춘다."""
    segs, owner, age = [], [], []
    for i in range(len(L)):
        end = min(t, int(L[i]) - 1)
        if end < 1:
            continue
        p = xy[i, :end + 1]
        segs.append(np.stack([p[:-1], p[1:]], axis=1))
        owner.append(np.full(end, i))
        age.append(end - 1 - np.arange(end))          # 가장 최근 선분 = 0
    if not segs:
        return np.zeros((0, 2, 2)), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(segs), np.concatenate(owner), np.concatenate(age)


def _make_renderer(n, box, px):
    model = mujoco_py.load_model_from_xml(multi_ant_xml(n))
    assert model.nq == n * _NQ_ANT, (model.nq, n)
    model.vis.quality.shadowsize = _SHADOWSIZE
    center, distance, elevation, azimuth = topdown_camera_for_box(box, float(model.vis.global_.fovy))
    model.stat.extent = (box[1] - box[0]) / 2.0 * _SHADOW_PAD
    model.light_pos[:, :2] = center
    sim = mujoco_py.MjSim(model)
    ctx = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
    mujoco_py.cymj._mjv_makeScene(model, ctx.scn, _MAXGEOM)
    assert ctx.scn.maxgeom == _MAXGEOM and model.ngeom < _MAXGEOM, (ctx.scn.maxgeom, model.ngeom)
    ctx.cam.lookat[:] = (center, center, 0.0)
    ctx.cam.azimuth, ctx.cam.elevation, ctx.cam.distance = azimuth, elevation, distance

    def render(qpos_flat):
        sim.data.qpos[:] = qpos_flat
        sim.forward()
        ctx.render(px, px, camera_id=-1)
        assert ctx.scn.ngeom == model.ngeom, (ctx.scn.ngeom, model.ngeom)   # geom 잘림 없음
        return np.asarray(ctx.read_pixels(px, px, depth=False))[::-1].copy()   # OpenGL 은 아래부터

    return render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--n_ants", type=int, default=100)
    ap.add_argument("--box", type=float, nargs=2, required=True, help="x, y 공통 범위 [m] (정사각)")
    ap.add_argument("--px", type=int, default=2048, help="렌더 화면(축 안쪽) 한 변 픽셀")
    ap.add_argument("--max_frames", type=int, default=None, help="스모크용")
    ap.add_argument("--still_at", type=int, default=None, help="그 스텝 한 장만 --out(.png)으로 (점검용)")
    args = ap.parse_args()

    q, L, heads = load_rollouts(args.npz, args.n_ants)
    N = q.shape[0]
    T_end = int(L.max()) if args.max_frames is None else min(int(L.max()), args.max_frames)
    x0, x1 = args.box
    P = args.px
    render = _make_renderer(N, args.box, P)

    hidden = q[:, 0].astype(np.float64)
    hidden[:, :2] = _HIDE_XY
    floor = render(hidden.reshape(-1))
    cmap = matplotlib.colormaps[_CMAP]
    rgb = np.array([cmap(_heading_unit(h))[:3] for h in heads])
    xy = q[..., :2].astype(np.float64)

    m = _MARGIN_PX
    W, H = P + m["left"] + m["right"], P + m["top"] + m["bottom"]
    fig = plt.figure(figsize=(W / _DPI, H / _DPI), dpi=_DPI)
    ax = fig.add_axes([m["left"] / W, m["bottom"] / H, P / W, P / H])
    ext = [x0, x1, x0, x1]
    im_floor = ax.imshow(np.zeros((P, P, 3), np.uint8), extent=ext, origin="upper", interpolation="none",
                         zorder=0)
    path_lc = LineCollection([], linewidths=_PATH_LW, zorder=1, capstyle="round")
    halo_lc = LineCollection([], linewidths=_TAIL_HALO_LW, colors=[_HALO_RGBA], zorder=2, capstyle="round")
    tail_lc = LineCollection([], linewidths=_TAIL_LW, zorder=3, capstyle="round")
    for lc in (path_lc, halo_lc, tail_lc):
        ax.add_collection(lc)
    im_ants = ax.imshow(np.zeros((P, P, 4), np.uint8), extent=ext, origin="upper", interpolation="none",
                        zorder=4)
    ticks = np.arange(np.ceil(x0 / _TICK_M) * _TICK_M, x1 + 1e-6, _TICK_M)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlim(x0, x1); ax.set_ylim(x0, x1)
    ax.set_xlabel("x [m]   (objective 1: vx)", fontsize=13)
    ax.set_ylabel("y [m]   (objective 2: vy)", fontsize=13)
    ax.tick_params(labelsize=11)
    fig.text(m["left"] / W, 1 - 35 / H, args.title, fontsize=15, fontweight="bold", ha="left", va="center")
    step_txt = fig.text(1 - m["right"] / W, 1 - 80 / H, "", fontsize=13, ha="right", va="center",
                        family="monospace")
    fig.text(m["left"] / W, 1 - 80 / H,
             f"{N} episodes (seeds 0-{N - 1}) at once, top-down . trail colour = final heading "
             f"(0° vx -> 90° vy, turbo) . bold tail = last {_TAIL_STEPS} steps",
             fontsize=11, ha="left", va="center", color="#555b66")

    # 프레임을 메모리에 쌓지 않고 바로 인코딩 (2198×2248×3 × 500 ≈ 7 GB/영상이라 병렬이면 RAM 을 다 쓴다).
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    still = args.still_at is not None
    writer = None if still else imageio.get_writer(args.out, fps=_FPS, codec="libx264", quality=8,
                                                    macro_block_size=1)
    shape = None
    for t in ([args.still_at] if still else range(T_end)):
        ti = np.minimum(t, L - 1)
        img = render(q[np.arange(N), ti].astype(np.float64).reshape(-1))
        ant = np.abs(img.astype(np.int16) - floor.astype(np.int16)).sum(-1) > _COMPOSITE_DIFF_THRESH
        im_floor.set_data(np.where(ant[..., None], floor, img))
        im_ants.set_data(np.dstack([img, (ant * 255).astype(np.uint8)]))

        segs, owner, age = _trail_segments(xy, L, t)
        old = age >= _TAIL_STEPS
        path_lc.set_segments(segs[old])
        path_lc.set_color(np.column_stack([rgb[owner[old]], np.full(old.sum(), _PATH_ALPHA)]))
        tail = ~old
        fade = _TAIL_ALPHA_MIN + (1.0 - _TAIL_ALPHA_MIN) * (1.0 - age[tail] / _TAIL_STEPS)
        halo_lc.set_segments(segs[tail])
        tail_lc.set_segments(segs[tail])
        tail_lc.set_color(np.column_stack([rgb[owner[tail]], fade]))

        step_txt.set_text(f"t = {t:3d} / {T_end - 1}")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frame = frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2].copy()   # yuv420p 는 짝수 크기만
        assert shape is None or frame.shape == shape, (frame.shape, shape)
        shape = frame.shape
        if still:
            imageio.imwrite(args.out, frame)
            print(f"Saved still t={t} {args.out}")
            return
        writer.append_data(frame)
        if (t + 1) % 50 == 0:
            print(f"  frame {t + 1}/{T_end}", flush=True)
    plt.close(fig)
    writer.close()
    print(f"Saved {args.out}  ({T_end} frames @ {_FPS} fps, {shape[1]}×{shape[0]})")


if __name__ == "__main__":
    main()
