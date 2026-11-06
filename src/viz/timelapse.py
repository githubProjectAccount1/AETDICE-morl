"""Ant 타임랩스 스프라이트 — 위에서 본 ant 를 일정 스텝마다 RGBA 로 떠서 월드 좌표 그대로 npz 에 담는다.

두 곳이 같은 코드를 쓰도록 여기 둔다 (2026-10-24 분리):
  · src.viz.render_topdown --timelapse            — 오프라인 학습 체크포인트 (AET/MoG/diffusion/BC)
  · experiments.parity.fw_oracle_timelapse        — mjx 로 학습한 FW 오라클 mixture
그림(월드 x/y 축, 100 궤적 + 스프라이트)은 experiments.moant.compose_dvm_timelapse 가 그린다.

이 모듈은 mujoco_py 를 직접 import 하지 않는다 — 호출자가 만든 env(AntEnv)·viewer 만 받는다.
"""
import os

import numpy as np

# 스프라이트의 soft alpha: 배경과의 RGB 차이 합(0–765)이 LO 이하면 0, HI 이상이면 1.
# 하드 임계값을 쓰면 ant 외곽 anti-aliasing 픽셀이 계단처럼 잘린다.
ALPHA_LO, ALPHA_HI = 8.0, 48.0
# 배경 스프라이트의 lookat 지면점 — ant 가 화면에 안 들어오게 (궤적은 1사분면, floor 반폭 300 m 안).
BG_LOOKAT = (-150.0, -150.0)
# 그림자 맵 반폭 = stat.extent × shadowclip 이 렌더 컨텍스트 생성 때 고정된다. ant.xml 은 extent=60 →
# ±60 m 를 1024 px 로 덮어 그림자가 ~8 px/m 점박이가 되고 ±60 m 밖엔 그림자가 없다. 스프라이트는 ant
# 주변만 그리므로 반폭을 줄이고 해상도를 올린 뒤, 타일마다 light 를 ant 위로 옮긴다 (render_tile).
SHADOW_EXTENT_M, SHADOWSIZE = 4.0, 4096
# 스프라이트 카메라: 바로 위에서(az 90 → image-right = 월드 +x, image-up = 월드 +y) 좁은 fovy 로 원근을
# 거의 없앤다 — fovy 45° 면 지면 위 ~0.55 m 의 몸통이 다리보다 ~13% 크게 찍힌다 (10° 면 ~2%).
# 방향·축척 확인 (2026-10-24): 월드 +x 1.5 m marker → 오른쪽 47.8 px, +y → 위 48.8 px (32 px/m).
AZIMUTH, ELEVATION, FOVY_DEG = 90.0, -89.9, 10.0


def final_heading_deg(traj):
    """시작점→끝점 진행방향 atan2(dy, dx) [deg]."""
    (x0, y0), (x1, y1) = traj[0], traj[-1]
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def apply_clean_floor(env, rgb):
    """바닥 checker 텍스처를 떼고 균일색으로 — 스프라이트 마스크가 바닥 diff 라서 필수다.
    렌더 전용 모델 수정이라 물리에는 영향이 없다 (contact/dynamics 는 material 을 안 읽는다)."""
    floor_id = env.sim.model.geom_name2id("floor")
    env.sim.model.geom_matid[floor_id] = -1
    env.sim.model.geom_rgba[floor_id] = np.array([*rgb, 1.0], dtype=np.float32)


def prepare_model_for_sprites(env):
    """**첫 render(컨텍스트 생성) 전에** 불러야 한다 — 그림자 맵 크기가 그때 고정된다."""
    env.sim.model.stat.extent = SHADOW_EXTENT_M
    env.sim.model.vis.quality.shadowsize = SHADOWSIZE


def timelapse_steps(n_steps, every):
    """스프라이트를 뜰 스텝: 0, every, 2·every, … 에 마지막 스텝을 더한다."""
    ts = list(range(0, n_steps, every))
    if ts[-1] != n_steps - 1:
        ts.append(n_steps - 1)
    return ts


def render_tile(env, viewer, state, look_xy, size):
    """저장한 MjSimState 로 되돌리고 지면점 look_xy 를 화면 중심에 둔 정사각 타일 (float32 RGB).
    directional light 의 xy 를 look_xy 로 옮긴다 — 그 위치가 그림자 맵 중심이다."""
    env.sim.model.light_pos[:, :2] = np.asarray(look_xy, dtype=np.float64)[None, :2]
    env.sim.set_state(state)
    env.sim.forward()
    viewer.cam.lookat[0] = float(look_xy[0])
    viewer.cam.lookat[1] = float(look_xy[1])
    viewer.cam.lookat[2] = 0.0
    img = env.render(mode="rgb_array", width=size, height=size, camera_id=-1)
    return np.asarray(img, dtype=np.float32).copy()


def sprite_rgba(tile, bg):
    """타일 − 배경 → RGBA uint8: ant(+ 그림자)만 불투명. 투명 픽셀의 RGB 는 0 (npz 압축이 잘 되게)."""
    assert tile.shape == bg.shape and tile.shape[-1] == 3, (tile.shape, bg.shape)
    alpha = np.clip((np.abs(tile - bg).sum(-1) - ALPHA_LO) / (ALPHA_HI - ALPHA_LO), 0.0, 1.0)
    rgb = np.where(alpha[..., None] > 0.0, tile, 0.0)
    return np.concatenate([rgb, 255.0 * alpha[..., None]], -1).round().clip(0, 255).astype(np.uint8)


def export_timelapse(env, viewer, episodes, n_episodes, num_steps, out_path, *, every, tile, ppm,
                     extra=None):
    """episodes: (traj_xy 목록, MjSimState 목록, raw_return) 를 n_episodes 번 내주는 iterable.
    스프라이트 카메라를 맞추고, 에피소드마다 바로 스프라이트를 떠서(상태를 다 들고 있지 않게) npz 로 저장."""
    assert every >= 1, every
    S, T, N = int(tile), int(num_steps), int(n_episodes)
    env.sim.model.vis.global_.fovy = FOVY_DEG
    viewer._markers.clear()
    viewer.cam.azimuth, viewer.cam.elevation = AZIMUTH, ELEVATION
    viewer.cam.distance = (S / 2) / (ppm * np.tan(np.radians(FOVY_DEG) / 2))   # 지면에서 ppm
    ghost_steps = np.asarray(timelapse_steps(T, every), dtype=np.int32)
    K = len(ghost_steps)
    xy = np.full((N, T, 2), np.nan, dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int32)
    ghost_xy = np.full((N, K, 2), np.nan, dtype=np.float32)       # 일찍 끝난 에피소드는 NaN
    sprites = np.zeros((N, K, S, S, 4), dtype=np.uint8)
    # 스텝별 qpos — 같은 rollout 으로 top-down 영상(src.viz.multiant_video)을 다시 굴리지 않고 만들기 위해
    # 저장한다 (2026-10-27). 일찍 끝난 에피소드는 NaN.
    nq = int(env.sim.model.nq)
    qpos = np.full((N, T, nq), np.nan, dtype=np.float32)
    raw_returns, bg, i = [], None, -1
    print(f"Timelapse: {N} episodes × {K} sprites ({S}px, {ppm:g} px/m) ...")
    for i, (traj, states, raw_return) in enumerate(episodes):
        assert i < N and len(states) == len(traj) <= T, (i, N, len(states), len(traj), T)
        if bg is None:
            bg = render_tile(env, viewer, states[0], BG_LOOKAT, S)
        L = len(traj)
        xy[i, :L], lengths[i] = np.asarray(traj, dtype=np.float32), L
        qpos[i, :L] = np.stack([np.asarray(st.qpos, dtype=np.float32) for st in states])
        raw_returns.append(np.asarray(raw_return, np.float32))
        for k, t in enumerate(ghost_steps):
            if t < L:
                sprites[i, k] = sprite_rgba(render_tile(env, viewer, states[t], traj[t], S), bg)
                ghost_xy[i, k] = traj[t]
        print(f"  Ep {i + 1:>3}/{N}: {L} steps, end ({traj[-1][0]:6.1f}, {traj[-1][1]:6.1f}), "
              f"heading {final_heading_deg(traj):5.1f}°", flush=True)
    assert i == N - 1, (i, N)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(
        out_path, xy=xy, lengths=lengths, ghost_steps=ghost_steps, ghost_xy=ghost_xy,
        sprites=sprites, sprite_half_m=np.float32(S / 2 / ppm), qpos=qpos,
        raw_returns=np.stack(raw_returns),
        headings_deg=np.asarray([final_heading_deg(xy[j, :lengths[j]]) for j in range(N)], np.float32),
        **(extra or {}))
    print(f"Saved {out_path}")
