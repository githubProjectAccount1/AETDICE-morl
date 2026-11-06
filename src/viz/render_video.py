"""Render a trained policy as an MP4 — any env (walker2d / half_cheetah / ant / etc.).

Sibling of render_topdown.py but simpler: uses each env's native MuJoCo camera
(side view for walker2d/half_cheetah, default for the others) instead of the
hand-built top-down rig. No trails — the camera moves with the body, so
breadcrumbs don't make sense across the moving frame.

Usage:
    python -m src.render_video \
        --run_dir ./results_Transformed/walker2d_amateur/.../G-linear_w1.0to0.0_F-cobbdouglas_rho0.1_beta1e-6_seed1_tfd4morl_walker2d_amateur \
        --out_mp4 walker_rho0.1_beta1e-6.mp4 \
        --num_episodes 3 --seed 0

Defaults pulled from the trained config:
    --num_steps  -> cfg.max_seq_len
    env class    -> dispatch on cfg.env_name
    reshape      -> cfg.reward_reshape (so rollouts match training-time bookkeeping)




# rho=0.1, β=1e-4
/venv/aet/bin/python -m src.render_video \
  --run_dir ./results_Transformed/walker2d_amateur/rsf1.0to1.0_musf1e4_walker2dcd/G-linear_w1.0to0.0_F-cobbdouglas_rho0.1_beta1e-4_seed1_tfd4morl_walker2d_amateur \
  --out_mp4 video/walker_cd/walker_rho0.1_beta1e-4.mp4 \
  --num_episodes 1 --seed 0


# rho=0.5, β=1e-4
/venv/aet/bin/python -m src.render_video \
  --run_dir ./results_Transformed/walker2d_amateur/rsf1.0to1.0_musf1e4_walker2dcd/G-linear_w1.0to0.0_F-cobbdouglas_beta1e-4_seed1_tfd4morl_walker2d_amateur \
  --out_mp4 video/walker_cd/walker_rho0.5_beta1e-4.mp4 \
  --num_episodes 1 --seed 1



# rho=0.9, β=1e-6
/venv/aet/bin/python -m src.render_video \
  --run_dir ./results_Transformed/walker2d_amateur/rsf1.0to1.0_musf1e4_walker2dcd/G-linear_w1.0to0.0_F-cobbdouglas_rho0.9_beta1e-6_seed1_tfd4morl_walker2d_amateur \
  --out_mp4 video/walker_cd/walker_rho0.9_beta1e-6.mp4 \
  --num_episodes 1 --seed 2
"""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp

# Repo-root injection so plain-script invocation works too.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.eval.rollout import (
    load_config, load_model_cpu, ENV_MAP, LEARNER_MODULES,
)
from src.data.reward_transform import RESHAPE_STEP


def _try_imports_for_writer():
    """Prefer imageio (more reliably available); fall back to mediapy."""
    try:
        import imageio.v2 as imageio
        return ("imageio", imageio)
    except ImportError:
        pass
    try:
        import mediapy
        return ("mediapy", mediapy)
    except ImportError:
        pass
    raise ImportError(
        "Need either `imageio` (with imageio-ffmpeg) or `mediapy` to write MP4. "
        "Install with: pip install imageio imageio-ffmpeg"
    )


_EPISODE_FILTERS = ("all", "falls", "completes")


def _keep_episode(episode_filter, fell):
    if episode_filter == "falls":
        return fell
    if episode_filter == "completes":
        return not fell
    return True


def _save_panorama(frames, steps, xpos, path, n_panels, window, crop, title=""):
    """프레임을 시간순으로 n_panels 장 골라 (가운데 crop 비율만) 가로로 이어 붙인 PNG.
    패널 위에 스텝 t 와 몸통 x 위치를 적는다 — 카메라가 몸통을 따라가 화면만으로는 전진 거리가 안 보이기 때문."""
    from PIL import Image, ImageDraw
    assert len(frames) == len(steps) == len(xpos) and frames
    idx = np.arange(len(frames))
    if window > 0:
        idx = idx[np.asarray(steps) >= steps[-1] - window]
    pick = idx[np.unique(np.linspace(0, len(idx) - 1, min(n_panels, len(idx))).round().astype(int))]
    H, W = frames[0].shape[:2]
    cw = max(1, int(round(W * crop)))
    x0 = (W - cw) // 2
    head, sep = 28, 4
    canvas = Image.new("RGB", (len(pick) * cw + (len(pick) - 1) * sep, H + head), "white")
    draw = ImageDraw.Draw(canvas)
    for j, i in enumerate(pick):
        x = j * (cw + sep)
        canvas.paste(Image.fromarray(frames[i][:, x0:x0 + cw]), (x, head))
        draw.text((x + 6, 8), f"t={steps[i]}  x={xpos[i]:.1f}m", fill="black")
    if title:
        draw.text((canvas.width - 8 * len(title) - 6, 8), title, fill="black")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    canvas.save(path)
    print(f"  panorama {len(pick)} panels -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Trained-policy run dir.")
    p.add_argument("--out_mp4", default="rollout.mp4")
    p.add_argument("--num_episodes", type=int, default=1,
                   help="How many episodes back-to-back. Each uses seed=base_seed+ep_idx.")
    p.add_argument("--num_steps", type=int, default=None,
                   help="Cap per episode. Default: cfg.max_seq_len from the run.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=60,
                   help="Video frame rate. With --realtime, auto-set to round(1/env.dt) "
                        "so 1 sim-second = 1 video-second (walker2d→125, ant/half_cheetah→20).")
    p.add_argument("--realtime", action="store_true",
                   help="Override --fps so playback matches simulated time (= round(1/env.dt)). "
                        "Walker2d videos are otherwise 4× slow-mo at the default 30 fps.")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Render every Nth env step instead of every step. Useful with the "
                        "default 30 fps to keep videos short without slow-mo: e.g. for walker2d "
                        "use --frame_stride 4 (skips 3 of every 4 frames, ~real-time at 30 fps).")
    p.add_argument("--camera_id", type=int, default=-1,
                   help="MuJoCo camera id; -1 = the env's free-cam (default per env).")
    # 2026-11-04 사용자: "실패하는 rollout들에 대해서 비디오 렌더링해줘. 넘어지는거 보고싶어.
    #   그리고 파노라마처럼, 비디오 프레임 이어 붙여서 이미지 하나 길게 만들어줘."
    #   (2026-11-04 추가) "CD utility의 잘 걸어가거나 뛰어가는 rho별로 파노라마도 만들어줘" → completes.
    p.add_argument("--episode_filter", choices=_EPISODE_FILTERS, default="all",
                   help="falls = 조기 종료(낙상, steps < num_steps) 에피소드만, completes = 끝까지 간 에피소드만. "
                        "all 이 아니면 seed 를 seed, seed+1, … 로 늘려 가며 --num_episodes 개를 모을 때까지 "
                        "(최대 --max_tries 에피소드) 굴린다.")
    p.add_argument("--max_tries", type=int, default=200)
    p.add_argument("--panorama_png", default=None,
                   help="에피소드마다 프레임을 가로로 이어 붙인 PNG 를 이 경로 접두사로 저장 (_ep{seed}.png).")
    p.add_argument("--pano_panels", type=int, default=10)
    p.add_argument("--pano_window", type=int, default=0,
                   help="파노라마에 쓸 마지막 스텝 수 (0 = 에피소드 전체). 낙상 과정만 보려면 ~150.")
    p.add_argument("--pano_crop", type=float, default=0.5,
                   help="각 패널에서 가운데 가로 비율만 남긴다 (카메라가 몸통을 따라가므로 가운데에 있다).")
    args = p.parse_args()

    cfg = load_config(args.run_dir)
    learner_type = getattr(cfg, "learner_type", "aet")
    if learner_type not in LEARNER_MODULES:
        raise ValueError(f"Unknown learner_type={learner_type!r} in config")
    if cfg.env_name not in ENV_MAP:
        raise ValueError(f"Unknown env_name={cfg.env_name!r} in config; "
                         f"expected one of {list(ENV_MAP)}")

    # Default num_steps to the training horizon. T=100 policy gets a 100-step rollout.
    if args.num_steps is None:
        args.num_steps = int(getattr(cfg, "max_seq_len", 500))
        print(f"--num_steps defaulted to cfg.max_seq_len={args.num_steps}")

    _learner = LEARNER_MODULES[learner_type]
    train_state = load_model_cpu(
        os.path.abspath(os.path.join(args.run_dir, "model")),
        cfg, _learner.init_train_state,
    )
    policy, _, _ = _learner.get_model(train_state.policy_state)
    policy.eval()

    # F̃ shift — same telescoping as the buffer; eval-side select_action mirror.
    utility_F  = cfg.utility_F
    state_mean = jnp.array(cfg.state_mean)
    state_std  = jnp.array(cfg.state_std)
    action_scale = jnp.asarray(
        getattr(cfg, "action_scale", np.ones(cfg.action_dim, dtype=np.float32)),
        dtype=jnp.float32,
    )
    # 조건부 racc 인코딩·state-only 판정은 src/eval/evaluate.py select_action 과 같아야 한다
    # (2026-11-02: rate 인코딩 런에서 F 를 스텝 없이 불러 죽었다 — rate 는 R/max(t,1), F 를 거치지 않는다).
    racc_encoding = getattr(cfg, "racc_encoding", "ftilde")
    from src.learners import is_state_only_policy
    state_only = is_state_only_policy(cfg)
    u0 = (utility_F(jnp.zeros((1, cfg.reward_dim), dtype=jnp.float32), keep_dims=True)
          if racc_encoding == "ftilde" else None)

    reward_min     = np.asarray(cfg.reward_min, dtype=np.float32)
    reward_max     = np.asarray(cfg.reward_max, dtype=np.float32)
    reward_span    = reward_max - reward_min
    reward_scaling = np.asarray(cfg.reward_scaling_factor, dtype=np.float32)

    reward_reshape = getattr(cfg, "reward_reshape", "none")
    reshape_step = RESHAPE_STEP[reward_reshape] if reward_reshape != "none" else None

    @jax.jit
    def select_action(state, racc_raw, t, rng):
        sb = (jnp.expand_dims(state, 0) - state_mean) / state_std
        if state_only:
            dist = policy(sb, timesteps=jnp.array([t]))
        else:
            rb = jnp.expand_dims(racc_raw, 0)
            rb_F = (rb / jnp.maximum(jnp.asarray(t, jnp.float32), 1.0) if racc_encoding == "rate"
                    else utility_F(rb, keep_dims=True) - u0)
            dist = policy(sb, rb_F, timesteps=jnp.array([t]))
        a = dist.sample(seed=rng)
        return jnp.clip(jnp.squeeze(a), -1.0, 1.0) * action_scale

    # Build env. Each Mujoco env's default render() works at user-supplied size.
    env = ENV_MAP[cfg.env_name]()

    # Resolve --realtime / --frame_stride against env.dt so playback timing is
    # explicit. env.dt is the wall-clock sim-time per env.step (= mujoco timestep
    # × frame_skip). A frame is rendered every `frame_stride` env steps, then
    # played at `fps`. Real-time playback requires fps == frame_stride / env.dt.
    if args.realtime:
        args.fps = max(1, round(args.frame_stride / env.dt))
        print(f"--realtime: fps auto-set to {args.fps} "
              f"(env.dt={env.dt:.4f}s, frame_stride={args.frame_stride})")
    elif args.frame_stride != 1:
        print(f"frame_stride={args.frame_stride}: rendering every {args.frame_stride}th env step "
              f"(playback rate = {args.fps * env.dt / args.frame_stride:.2f}× real-time)")

    writer_kind, writer_mod = _try_imports_for_writer()
    print(f"Recording {args.num_episodes} episode(s) × ≤{args.num_steps} steps "
          f"of {cfg.env_name} from {args.run_dir} ...")

    frames = []
    kept = 0
    filtering = args.episode_filter != "all"
    n_try = args.max_tries if filtering else args.num_episodes
    for ep in range(n_try):
        if kept >= args.num_episodes:
            break
        n_tried = ep + 1
        ep_frames = []
        ep_steps = []
        ep_x = []
        env.seed(args.seed + ep)
        state = env.reset()
        racc_raw = np.zeros((cfg.reward_dim,), dtype=np.float32)
        key = jax.random.PRNGKey(args.seed + ep)

        ep_obj0 = 0.0; ep_obj1 = 0.0; steps = 0; done = False
        while not done and steps < args.num_steps:
            key, sub = jax.random.split(key)
            action = np.asarray(select_action(
                jnp.asarray(state, dtype=jnp.float32),
                jnp.asarray(racc_raw, dtype=jnp.float32),
                steps, sub,
            ))
            next_obs, _, done, info = env.step(action)
            env_obj = np.asarray(info["obj"], dtype=np.float32)
            raw_r = reshape_step(action, env_obj, next_obs) if reshape_step is not None else env_obj
            norm_r = reward_scaling * (raw_r - reward_min) / reward_span
            racc_raw += norm_r
            ep_obj0 += float(raw_r[0])
            if raw_r.shape[0] > 1: ep_obj1 += float(raw_r[1])

            # render frame — only every frame_stride'th env step, plus the very
            # first one (steps==0 here since we incremented after the action).
            if steps % args.frame_stride == 0:
                try:
                    frame = env.render(mode="rgb_array",
                                       width=args.width, height=args.height,
                                       camera_id=None if args.camera_id < 0 else args.camera_id)
                except TypeError:
                    # Older mujoco_py signature without size kwargs — env defaults.
                    frame = env.render(mode="rgb_array")
                if frame is not None:
                    ep_frames.append(np.asarray(frame))
                    ep_steps.append(steps)
                    ep_x.append(float(env.sim.data.qpos[0]))
            state = next_obs
            steps += 1

        fell = steps < args.num_steps
        if not _keep_episode(args.episode_filter, fell):
            continue
        kept += 1
        print(f"  Ep seed={args.seed + ep}  steps={steps}{' (낙상)' if fell else ''}  "
              f"obj0={ep_obj0:.2f}  obj1={ep_obj1:.2f}")
        frames.extend(ep_frames)
        if args.panorama_png:
            _save_panorama(ep_frames, ep_steps, ep_x, f"{args.panorama_png}_ep{args.seed + ep}.png",
                           args.pano_panels, args.pano_window, args.pano_crop,
                           title=f"seed {args.seed + ep}: {steps} steps" + (" (fell)" if fell else ""))
    if filtering:
        print(f"{args.episode_filter} {kept}개 / {n_tried} 에피소드 시도")
        if kept == 0:
            print(f"{args.episode_filter} 에피소드가 없다 — 영상을 만들지 않는다")
            return

    # Save MP4
    out_dir = os.path.dirname(os.path.abspath(args.out_mp4))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if writer_kind == "imageio":
        writer_mod.mimsave(args.out_mp4, frames, fps=args.fps, codec="libx264", quality=8)
    else:
        writer_mod.write_video(args.out_mp4, np.stack(frames, axis=0), fps=args.fps)
    print(f"Saved {len(frames)} frames @ {args.fps}fps -> {args.out_mp4}")


if __name__ == "__main__":
    main()
