"""Per-job training entrypoint.

This module is intentionally a thin orchestrator. The phases below map 1:1 to
the helpers in `src/train/{cli,setup,loop}.py`; the heavy lifting (argparse,
env build, buffer load, wandb init, train loop, checkpoint write) all lives
in those modules. Read top-to-bottom to see the full pipeline:

    1. parse + validate args  →  cli.parse_and_validate()
    2. select learner module  →  setup.select_learner()
    3. build env + reshape    →  setup.build_env() / apply_reward_reshape()
    4. assert dim consistency →  setup.assert_dim_consistency()
    5. cd_filter + utility_F  →  setup.build_cd_filter() / build_utility_F()
    6. resolve save_dir       →  setup.resolve_save_dir()
    7. load data + buffer     →  setup.build_buffer()
    8. β schedule + cfg dump  →  setup.attach_beta_schedule() / maybe_dump_config()
    9. init state + wandb     →  setup.init_run_state() / init_wandb()
   10. train loop             →  loop.train_loop()
   11. finalize artifacts     →  loop.finalize()

For the original Korean design notes on utility G / utility F / scaling that
motivated this pipeline, see `docs/utility_design.md`.
"""
import os

# JAX env tweaks must be set BEFORE jax is imported transitively (which happens
# inside the setup module). Keep this block at the very top.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from src.train.cli import parse_and_validate
from src.train.loop import finalize, train_loop
from src.train.setup import (
    apply_reward_reshape,
    assert_dim_consistency,
    attach_beta_schedule,
    attach_wbc_weights,
    build_buffer,
    build_cd_filter,
    build_env,
    build_utility_F,
    init_run_state,
    init_wandb,
    maybe_dump_config,
    resolve_reward_bounds,
    resolve_save_dir,
    select_learner,
)


def main():
    config = parse_and_validate()
    learner = select_learner(config)

    env = build_env(config)
    apply_reward_reshape(config)
    print(f"State dim: {config.state_dim}, Action dim: {config.action_dim}, "
          f"Reward dim: {config.reward_dim} (reshape={config.reward_reshape})")

    assert_dim_consistency(config)

    cd_filter = build_cd_filter(config)
    # CD 의 정규화 상수 (0, T) 를 utility_F 생성 전에 config 에 채운다 (no-op for 그 외 설정).
    resolve_reward_bounds(config, cd_filter)
    build_utility_F(config)

    run_name, save_dir = resolve_save_dir(config)

    buffer = build_buffer(config, cd_filter)
    attach_beta_schedule(config)
    maybe_dump_config(config, save_dir)

    train_state, key, start_iter = init_run_state(config, save_dir, learner)
    attach_wbc_weights(config, buffer, train_state, learner)
    init_wandb(config, save_dir, run_name)

    train_state, eval_info = train_loop(
        config, env, buffer, train_state, key, save_dir, run_name,
        start_iter, learner,
    )
    finalize(config, buffer, train_state, eval_info, save_dir, learner)


if __name__ == "__main__":
    main()
