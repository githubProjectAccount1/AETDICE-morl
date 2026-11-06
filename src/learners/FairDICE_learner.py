"""FairDICE learner adapted to the src pipeline.

Same Bellman / Fenchel structure as `AET_learner.py`, but the
critic and policy are state-only (`FairDICECritic`, `FairDICEContinuousPolicy`)
— no racc conditioning. The augmented per-step reward (F̃-telescoped, scalarized
via mu) is supplied by the Transformed Buffer, so FairDICE-style state-only
training optimizes the same dual `E[G(F̃(R))]` objective.

Conventions (kept identical to AET_learner):
  - Outer utility G ∈ {"linear", "piecewise_log"};
    conjugate via `conjugate_G(mu, utility_G_type, mu_scaling_factor)`.
  - Inner utility F is absorbed into `batch.rewards` upstream (Buffer); it
    never appears in this learner.
  - `config.fair_alpha` is unused.
"""
import jax
import jax.numpy as jnp
from flax import nnx
from src.nets.policy import MuNetworkTransformed, FairDICEContinuousPolicy
from src.nets.critic import FairDICECritic
from src.divergence import f, FDivergence, f_derivative_inverse
from src.utility import conjugate_G, g_kwargs_from_config
from src.learners import _common
from src.learners._common import (
    NetworkState, TrainState, Model,
    _mu_is_learnable, get_model, make_cosine_adam_tx,
    save_model, save_checkpoint,
)


def init_train_state(config) -> TrainState:
    rngs = nnx.Rngs(config.seed)

    policy = FairDICEContinuousPolicy(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dims=config.hidden_dims[:-1],
        horizon=config.max_seq_len,
        layer_norm=config.layer_norm,
        rngs=rngs,
    )

    nu = FairDICECritic(
        state_dim=config.state_dim,
        hidden_dims=config.hidden_dims,
        horizon=config.max_seq_len,
        layer_norm=config.layer_norm,
        nu_output_transform=getattr(config, "nu_output_transform", "none"),
        rngs=rngs,
    )

    # Un-clipped Adam chains (grad_clip removed 2026-09-02).
    policy_tx = make_cosine_adam_tx(config.policy_lr, config.total_train_steps)
    nu_tx = make_cosine_adam_tx(config.critic_lr, config.total_train_steps)
    mu_tx = make_cosine_adam_tx(config.mu_lr, config.total_train_steps)

    policy_optim = nnx.Optimizer(policy, policy_tx)
    (policy_gd, policy_state) = nnx.split((policy, policy_optim))

    nu_optim = nnx.Optimizer(nu, nu_tx)
    (nu_gd, nu_state) = nnx.split((nu, nu_optim))

    mu = MuNetworkTransformed(config)
    mu_optim = nnx.Optimizer(mu, mu_tx)
    (mu_gd, mu_state) = nnx.split((mu, mu_optim))

    nu_target = nu_state.filter(nnx.Param)
    policy_target = policy_state.filter(nnx.Param)
    mu_target = mu_state.filter(nnx.Param)

    return TrainState(
        policy_state=NetworkState(policy_gd, policy_state, policy_target),
        nu_state=NetworkState(nu_gd, nu_state, nu_target),
        mu_state=NetworkState(mu_gd, mu_state, mu_target),
        step=jnp.array(0),
    )


def train_step(config, train_state: TrainState, batch, key: jax.random.PRNGKey):
    key, subkey = jax.random.split(key)
    step = train_state.step
    # β is schedule-driven; constant schedule degrades to config.beta unchanged.
    beta = config.beta_schedule_fn(step)
    f_divergence = FDivergence[config.divergence]
    # w_max only matters for f_divergence=CLIP_CHI; default ∞ for other divergences.
    w_max = float(getattr(config, "w_max", float("inf")))
    # Standardize Bellman residual by batch σ (stop-grad) when --standardize_e.
    standardize_e = bool(getattr(config, "standardize_e", False))

    # batch.rewards is the F̃-telescoped per-step augmented reward (built in Buffer).
    # Summed over an episode it telescopes to F̃(R_H) = F(R_H) - F(0), so dotting with
    # mu and running standard FairDICE finite-horizon optimizes E[G(F̃(R))] in the dual.
    rewards = batch.rewards      # [B, RewardDim]
    states = batch.states        # [B, StateDim]
    next_states = batch.next_states
    init_states = batch.init_states
    # batch.raccs / batch.next_raccs / batch.init_raccs are intentionally ignored:
    # FairDICECritic and FairDICEContinuousPolicy are state-only.

    policy, policy_optim, _ = get_model(train_state.policy_state)
    nu_network, nu_optim, _ = get_model(train_state.nu_state)
    mu_network, mu_optim, _ = get_model(train_state.mu_state)

    def nu_loss_fn(nu_network, mu_network):
        nu_values = nu_network(states)
        next_nu_values = nu_network(next_states)

        nu_t = nu_values[:, :-1]        # [B, 0:H]
        nu_tp1 = next_nu_values[:, 1:]  # [B, 1:H+1]

        init_nu = nu_network(init_states)[:, 0]  # Value at t=0

        mu = mu_network()

        r_scalar = (rewards @ mu).reshape(-1, 1)
        e_matrix = r_scalar + nu_tp1 - nu_t
        e_flat_raw = e_matrix.flatten()
        # When standardize_e is on, normalize e by batch σ (stop-grad).
        if standardize_e:
            sigma_e = jax.lax.stop_gradient(jnp.std(e_flat_raw) + 1e-8)
            e_flat = e_flat_raw / sigma_e
        else:
            sigma_e = jnp.float32(1.0)
            e_flat = e_flat_raw

        w_flat = jax.nn.relu(f_derivative_inverse(e_flat / beta, f_divergence, w_max=w_max))

        # Split the inner Lagrangian term so wandb can show reg vs Bellman pull
        # separately. Mathematically: loss_2 = mean(w·e) − β·mean(f(w)).
        f_w_flat     = f(w_flat, f_divergence)
        bellman_term = jnp.mean(w_flat * e_flat)
        reg_term     = beta * jnp.mean(f_w_flat)

        loss_1 = jnp.mean(init_nu)
        loss_2 = bellman_term - reg_term
        loss_3 = conjugate_G(mu, config.utility_G_type, config.mu_scaling_factor,
                             **g_kwargs_from_config(config))

        total_loss = loss_1 + loss_2 + loss_3

        # Distributional summaries of the critic outputs (mirrors AET_learner):
        # exclude the last column because critic.py:30-32 hard-masks it to zero (terminal
        # convention, not a network prediction). Matches the slice the loss uses for nu_t.
        nu_predict = nu_values[:, :-1]
        nu_mean = jnp.mean(nu_predict)
        nu_std = jnp.std(nu_predict)

        # Diagnostics mirroring AET_learner (see comments there).
        w_max_obs        = jnp.max(w_flat)
        w_p99            = jnp.quantile(w_flat, 0.99)
        w_frac_gt1       = jnp.mean((w_flat > 1.0).astype(jnp.float32))
        w_frac_sat       = jnp.mean((w_flat >= w_max * 0.999).astype(jnp.float32))
        ess              = (jnp.sum(w_flat) ** 2) / (w_flat.size * (jnp.sum(w_flat ** 2) + 1e-12))
        reg_over_bellman = reg_term / (jnp.abs(bellman_term) + 1e-8)
        dbg = {
            "debug_bellman_term":     bellman_term,
            "debug_reg_term":         reg_term,
            "debug_reg_over_bellman": reg_over_bellman,
            "debug_f_w_mean":         jnp.mean(f_w_flat),
            "debug_w_max":            w_max_obs,
            "debug_w_p99":            w_p99,
            "debug_w_frac_gt1":       w_frac_gt1,
            "debug_w_frac_sat":       w_frac_sat,
            "debug_ess":              ess,
        }

        return total_loss, (loss_1, loss_2, loss_3, w_flat, e_flat_raw, sigma_e, mu, nu_mean, nu_std, dbg)

    has_aux = True
    argnums = (0, 1) if _mu_is_learnable(config) else 0

    grad_fn = nnx.value_and_grad(nu_loss_fn, argnums=argnums, has_aux=has_aux)
    (nu_loss, (loss_1, loss_2, loss_3, w_flat, e_flat_raw, sigma_e, mu, nu_mean, nu_std, dbg)), grads = grad_fn(nu_network, mu_network)

    if _mu_is_learnable(config):
        nu_grads, mu_grads = grads
        mu_optim.update(mu_grads)
    else:
        nu_grads = grads

    nu_optim.update(nu_grads)

    nu_state_ = nnx.state((nu_network, nu_optim))
    mu_state_ = nnx.state((mu_network, mu_optim))

    def policy_loss_fn(policy):
        # dist: Independent(Normal([B,H,A]), 1) → batch_shape=[B,H], event_shape=[A]
        dist = policy(states)

        # Broadcast actions [B, A] → [B, H, A]
        actions_broad = jnp.repeat(batch.actions[:, None, :], config.max_seq_len, axis=1)
        log_probs = dist.log_prob(actions_broad)   # [B, H]
        log_probs_flat = log_probs.flatten()        # [B*H]

        mu = mu_network()
        r_scalar = (rewards @ mu).reshape(-1, 1)
        nu_values = nu_network(states)
        next_nu_values = nu_network(next_states)
        nu_t = nu_values[:, :-1]
        nu_tp1 = next_nu_values[:, 1:]
        e_matrix = r_scalar + nu_tp1 - nu_t
        e_flat_raw = e_matrix.flatten()
        # standardize_e: normalize e by batch σ (stop-grad), matching nu_loss_fn so
        # the policy's w uses the same residual scale as the critic update.
        # (Structure mirrors nu_loss_fn; nu additionally binds sigma_e for logging.)
        if standardize_e:
            e_flat = e_flat_raw / jax.lax.stop_gradient(jnp.std(e_flat_raw) + 1e-8)
        else:
            e_flat = e_flat_raw
        w_flat = jax.nn.relu(f_derivative_inverse(e_flat / beta, f_divergence, w_max=w_max))
        w_detached = jax.lax.stop_gradient(w_flat)
        policy_loss = -jnp.mean(w_detached * log_probs_flat)

        return policy_loss, log_probs

    (policy_loss, log_probs), policy_grads = nnx.value_and_grad(policy_loss_fn, has_aux=True)(
        policy
    )
    policy_optim.update(policy_grads)
    policy_state_ = nnx.state((policy, policy_optim))

    train_state = train_state._replace(
        policy_state=train_state.policy_state._replace(state=policy_state_),
        nu_state=train_state.nu_state._replace(state=nu_state_),
        mu_state=train_state.mu_state._replace(state=mu_state_),
        step=step + 1,
    )

    return train_state, {
        "policy_loss": policy_loss,
        "nu_loss": nu_loss,
        "mu": mu,
        "debug_w_mean": jnp.mean(w_flat),
        "debug_e_mean": jnp.mean(e_flat_raw),
        "debug_sigma_e": sigma_e,    # 1.0 when --standardize_e off; running batch σ otherwise
        "debug_nu_mean": nu_mean,
        "debug_nu_std": nu_std,
        "debug_beta_current": beta,  # schedule-driven; constant for --beta_schedule=constant
        "loss_1": loss_1,
        "loss_2": loss_2,
        "loss_3": loss_3,
        **dbg,
    }


# save_model / save_checkpoint imported from _common. load_model / load_checkpoint
# are thin wrappers binding this module's init_train_state as the restore template.
def load_model(path: str, config) -> TrainState:
    return _common.load_model(path, config, init_train_state)


def load_checkpoint(save_dir: str, config):
    return _common.load_checkpoint(save_dir, config, init_train_state)
