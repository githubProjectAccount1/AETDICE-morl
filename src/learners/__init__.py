"""Learner registry — one module per method.

Public dispatch table: `LEARNER_MODULES` maps `--learner_type` CLI choice to the
corresponding learner module. Each module exposes the same surface:
    init_train_state, train_step, get_model,
    save_model, load_model, save_checkpoint, load_checkpoint

Adding a new learner = one new module + one row in `LEARNER_MODULES`.
No if-tree edits anywhere else.
"""
from . import AET_learner, FairDICE_learner, IQL_learner, BC_learner

LEARNER_MODULES = {
    "aet":      AET_learner,      # state+racc DICE dual; racc-conditioned ν
    "fairdice": FairDICE_learner, # state-only DICE dual; racc ignored
    "iql":      IQL_learner,      # state+racc ESR baseline (Q + V + AWR; μ fixed)
    "bc":       BC_learner,       # filtered behavior cloning on top-K% CD trajs
}


def is_state_only_policy(config) -> bool:
    """정책이 racc 를 **받지 않는가** — evaluate / render 가 정책 호출 형태를 고르는 단일 기준.

    분기 기준은 learner 이름이 아니라 정책 입력이다 (2026-09-16): BC 는 --bc_conditioning=racc 로도
    돌기 때문에 learner_type 만 보면 racc-BC 를 state-only 로 호출하게 된다. learner 별 판정은 그
    learner 모듈이 소유한다. 옛 config.json(learner_type 없음)은 aet 로 되짚는다.
    """
    learner_type = getattr(config, "learner_type", "aet")
    if learner_type == "bc":
        return BC_learner.bc_conditioning(config) == "state"
    return learner_type == "fairdice"

