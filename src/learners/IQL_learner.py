"""IQL (Implicit Q-Learning) baseline adapted to the src pipeline.

ESR baseline: maximizes E[F(R)] under a fixed preference scalarization. Compared
to AET / FairDICE, there's no DICE dual — IQL uses three losses:

  1. V-loss (expectile regression) — V(s, racc) is fit to the τ-expectile of
     Q_target(s, a, racc) under the data action distribution. With τ in (0.5, 1),
     V approximates the "best in-distribution Q" without ever querying Q at an
     out-of-distribution action.
  2. Q-loss (Bellman, V-bootstrap) — Q(s, a, racc) regressed to r·μ + V(s', racc'),
     where μ is fixed to `pref_weights` (G-linear only) and r is the F̃-telescoped
     per-step reward from Buffer (so trajectory sum = F̃(R_H) = F(R_H) − F(0)).
     A Polyak-averaged target Q computes Q_target_selected used by the V-loss.
  3. Policy-loss (AWR) — π imitates dataset actions, reweighted by
     exp(temperature · A) where A = Q_target − V is detached and clipped.

Conventions kept identical to AET/FairDICE_learner:
  - Continuous actions, MoG policy via `ContinuousPolicy(state, racc)`.
  - Per-timestep value heads with last column masked to 0 (terminal absorbing).
  - μ via `MuNetworkTransformed`; fail-fast at argparse if utility_G_type is
    learnable-μ (plog / plog_one) — IQL has no μ-update objective.
  - save_checkpoint / load_checkpoint surface mirrors the other learners so the
    sweep runner's auto-resume works without per-learner branches.
"""
from collections import namedtuple
import jax
import jax.numpy as jnp
from flax import nnx
from src.nets.policy import MuNetworkTransformed, ContinuousPolicy
from src.nets.critic import Critic, QCritic
from src.learners import _common
from src.learners._common import (
    NetworkState, Model, get_model, make_cosine_adam_tx,
    save_model, save_checkpoint,
)


# IQL keeps its OWN 5-field TrainState (adds q_state / v_state); NetworkState /
# Model / get_model are the shared variants from _common (get_model's None-guard
# handles IQL's None target_params on the v / policy / mu slots).
TrainState = namedtuple(
    "TrainState",
    ["policy_state", "q_state", "v_state", "mu_state", "step"],
)


def init_train_state(config) -> TrainState:
    rngs = nnx.Rngs(config.seed)

    # See comment in AET_learner.init_train_state — racc_dim sizes
    # the network's racc input from F̃-output dim, falling back to reward_dim
    # when they coincide (pointwise F).
    racc_dim = getattr(config, "racc_dim", config.reward_dim)
    policy = ContinuousPolicy(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        reward_dim=config.reward_dim,
        racc_dim=racc_dim,
        racc_mean=config.racc_mean,
        racc_std=config.racc_std,
        hidden_dims=config.hidden_dims[:-1],
        horizon=config.max_seq_len,
        layer_norm=config.layer_norm,
        rngs=rngs,
    )

    q_network = QCritic(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        reward_dim=config.reward_dim,
        racc_dim=racc_dim,
        racc_mean=config.racc_mean,
        racc_std=config.racc_std,
        hidden_dims=config.hidden_dims,
        horizon=config.max_seq_len,
        layer_norm=config.layer_norm,
        rngs=rngs,
    )

    v_network = Critic(
        state_dim=config.state_dim,
        reward_dim=config.reward_dim,
        racc_dim=racc_dim,
        racc_mean=config.racc_mean,
        racc_std=config.racc_std,
        hidden_dims=config.hidden_dims,
        horizon=config.max_seq_len,
        layer_norm=config.layer_norm,
        rngs=rngs,
    )

    # Un-clipped Adam chains (grad_clip removed 2026-09-02).
    policy_tx = make_cosine_adam_tx(config.policy_lr, config.total_train_steps)
    q_tx = make_cosine_adam_tx(config.critic_lr, config.total_train_steps)
    v_tx = make_cosine_adam_tx(config.critic_lr, config.total_train_steps)

    # mu is fixed for IQL (G-linear only enforced at argparse); the optimizer is built
    # for surface parity with AET/FairDICE TrainState but never receives gradient
    # updates because mu has no learnable params under G-linear.
    mu_tx = make_cosine_adam_tx(config.mu_lr, config.total_train_steps)

    policy_optim = nnx.Optimizer(policy, policy_tx)
    (policy_gd, policy_state) = nnx.split((policy, policy_optim))

    q_optim = nnx.Optimizer(q_network, q_tx)
    (q_gd, q_state) = nnx.split((q_network, q_optim))

    v_optim = nnx.Optimizer(v_network, v_tx)
    (v_gd, v_state) = nnx.split((v_network, v_optim))

    mu = MuNetworkTransformed(config)
    mu_optim = nnx.Optimizer(mu, mu_tx)
    (mu_gd, mu_state) = nnx.split((mu, mu_optim))

    q_target = q_state.filter(nnx.Param)
    policy_target = policy_state.filter(nnx.Param)
    v_target = v_state.filter(nnx.Param)
    mu_target = mu_state.filter(nnx.Param)

    return TrainState(
        policy_state=NetworkState(policy_gd, policy_state, policy_target),
        q_state=NetworkState(q_gd, q_state, q_target),
        v_state=NetworkState(v_gd, v_state, v_target),
        mu_state=NetworkState(mu_gd, mu_state, mu_target),
        step=jnp.array(0),
    )


def _expectile_loss(diff, expectile):
    """Asymmetric MSE: residuals where Q > V get weighted by `expectile`,
    residuals where Q < V by (1 - expectile). With expectile > 0.5, V learns
    the upper tail of Q-given-state under the data distribution."""
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return jnp.mean(weight * jnp.square(diff))


def train_step(config, train_state: TrainState, batch, key: jax.random.PRNGKey):
    """One IQL update step. Order matters: V → Q → policy.

    V is fit first (uses Q_target), then Q is fit using the just-updated V's
    bootstrap target, then the policy is extracted via AWR with both Q_target
    and V detached. The Polyak target Q is updated last.
    """
    step = train_state.step

    rewards = batch.rewards            # [B, RewardDim]   F̃-telescoped from Buffer
    states = batch.states              # [B, StateDim]
    next_states = batch.next_states
    actions = batch.actions            # [B, ActionDim]
    raccs = batch.raccs                # [B, RewardDim]   F̃-transformed in Buffer
    next_raccs = batch.next_raccs

    policy, policy_optim, _ = get_model(train_state.policy_state)
    q_net, q_optim, q_target_net = get_model(train_state.q_state)
    v_net, v_optim, _ = get_model(train_state.v_state)
    mu_network, mu_optim, _ = get_model(train_state.mu_state)

    # μ is fixed (G-linear, validated at argparse). r_scalar broadcasts against
    # the [B, H] residual matrices below.
    mu = mu_network()
    r_scalar = (rewards @ mu).reshape(-1, 1)  # [B, 1]

    # Q_target_selected: target Q at the dataset's (s, a). Used by V-loss as
    # regression target and by policy-loss as part of the advantage. Detached
    # everywhere — the target network is only updated via Polyak below.
    q_target_full = q_target_net(states, actions, raccs, timesteps=None)  # [B, H+1]
    q_target_selected = jax.lax.stop_gradient(q_target_full[:, :-1])      # [B, H]

    # ── V loss: expectile regression of V toward Q_target_selected ─────────
    def v_loss_fn(v_net):
        v_full = v_net(states, raccs, timesteps=None)  # [B, H+1]
        v_pred = v_full[:, :-1]                         # [B, H]; terminal column is mask·0
        diff = q_target_selected - v_pred
        return _expectile_loss(diff, config.expectile), v_pred

    (v_loss, v_pred), v_grads = nnx.value_and_grad(v_loss_fn, has_aux=True)(v_net)
    v_optim.update(v_grads)

    # Re-evaluate V with updated params for Q's bootstrap. nnx.Optimizer.update
    # mutates the wrapped module in place, so v_net now holds the post-update params.
    next_v_full = v_net(next_states, next_raccs, timesteps=None)  # [B, H+1]
    next_v = next_v_full[:, 1:]                                    # [B, H]; aligned with V_t
    target_q = jax.lax.stop_gradient(r_scalar + next_v)            # [B, H]

    # ── Q loss: MSE Bellman with V-bootstrap (no max, no OOD action query) ─
    def q_loss_fn(q_net):
        q_full = q_net(states, actions, raccs, timesteps=None)  # [B, H+1]
        q_pred = q_full[:, :-1]                                  # [B, H]
        return jnp.mean(jnp.square(q_pred - target_q)), q_pred

    (q_loss, q_pred), q_grads = nnx.value_and_grad(q_loss_fn, has_aux=True)(q_net)
    q_optim.update(q_grads)

    # ── Policy loss: AWR with detached advantage ───────────────────────────
    # Advantage uses the (pre-update, target-network) Q so the policy weights
    # are stationary within this step. exp_a is clipped to keep the regression
    # bounded — standard IQL practice (offline RL exp-weights blow up otherwise).
    adv = q_target_selected - jax.lax.stop_gradient(v_pred)  # [B, H]
    exp_a = jnp.minimum(jnp.exp(adv * config.temperature), config.awr_exp_clip)
    exp_a_flat = exp_a.flatten()  # [B*H]

    def policy_loss_fn(policy):
        dist = policy(states, raccs)                      # batch_shape=[B, H]
        actions_broad = jnp.repeat(
            batch.actions[:, None, :], config.max_seq_len, axis=1
        )                                                 # [B, H, A]
        log_probs = dist.log_prob(actions_broad)          # [B, H]
        log_probs_flat = log_probs.flatten()              # [B*H]
        return -jnp.mean(exp_a_flat * log_probs_flat), log_probs

    (policy_loss, _), policy_grads = nnx.value_and_grad(policy_loss_fn, has_aux=True)(policy)
    policy_optim.update(policy_grads)

    # ── Polyak update target Q ─────────────────────────────────────────────
    q_state_ = nnx.state((q_net, q_optim))
    new_q_params = q_state_.filter(nnx.Param)
    tau = config.tau
    new_q_target = jax.tree_map(
        lambda t, s: (1.0 - tau) * t + tau * s,
        train_state.q_state.target_params,
        new_q_params,
    )

    policy_state_ = nnx.state((policy, policy_optim))
    v_state_ = nnx.state((v_net, v_optim))
    mu_state_ = nnx.state((mu_network, mu_optim))

    train_state = TrainState(
        policy_state=train_state.policy_state._replace(state=policy_state_),
        q_state=train_state.q_state._replace(state=q_state_, target_params=new_q_target),
        v_state=train_state.v_state._replace(state=v_state_),
        mu_state=train_state.mu_state._replace(state=mu_state_),
        step=step + 1,
    )

    return train_state, {
        "policy_loss": policy_loss,
        "q_loss": q_loss,
        "v_loss": v_loss,
        "mu": mu,
        "debug_q_mean": jnp.mean(q_target_selected),
        "debug_v_mean": jnp.mean(v_pred),
        "debug_adv_mean": jnp.mean(adv),
        "debug_exp_a_mean": jnp.mean(exp_a_flat),
    }


# save_model / save_checkpoint imported from _common. load_model / load_checkpoint
# are thin wrappers binding this module's init_train_state as the restore template.
def load_model(path: str, config) -> TrainState:
    return _common.load_model(path, config, init_train_state)


def load_checkpoint(save_dir: str, config):
    return _common.load_checkpoint(save_dir, config, init_train_state)
