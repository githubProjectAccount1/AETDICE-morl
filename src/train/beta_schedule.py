"""β-decay schedules for the DICE Bellman dual.

`make_beta_schedule(config)` returns a JAX-compatible callable `step → β` that
gets attached to `config.beta_schedule_fn` once at startup. AET / FairDICE
`train_step` reads `β = config.beta_schedule_fn(step)` instead of the constant
`config.beta`. Inside the JIT'd graph, β becomes a function of `step`.

Constant case is special-cased so existing configs (only `--beta` set) keep
working unchanged: the closure always returns the same value.

Direction convention: nothing is enforced — `β_init > β_final` is "decay" (start
loose, end tight, the natural DICE direction); `β_init < β_final` is "anneal".
"""
import jax.numpy as jnp


_SCHEDULES = ("constant", "linear", "exp", "cosine")


def make_beta_schedule(config):
    """Build the schedule callable from `config` knobs.

    Reads:
      config.beta_schedule        ∈ {constant, linear, exp, cosine}; default "constant"
      config.beta                 used for constant
      config.beta_init            initial β for non-constant schedules
      config.beta_final           final β for non-constant schedules
      config.beta_warmup_steps    hold β_init for this many steps before decay (default 0)
      config.total_train_steps    decay denominator
    """
    schedule = getattr(config, "beta_schedule", "constant")
    if schedule not in _SCHEDULES:
        raise ValueError(
            f"Unknown beta_schedule={schedule!r}; expected one of {_SCHEDULES}"
        )

    if schedule == "constant":
        beta_const = float(config.beta)
        # Wrap as jnp.float32 so the type matches the non-constant branches and
        # downstream `e_flat / beta` / `beta * f(...)` ops never trip a dtype mismatch.
        return lambda step: jnp.float32(beta_const)

    init = float(config.beta_init)
    final = float(config.beta_final)
    warmup = int(getattr(config, "beta_warmup_steps", 0))
    total = int(config.total_train_steps)
    decay_span = max(1, total - warmup)

    def _t(step):
        # Fraction of the decay window completed, ∈ [0, 1].
        # During warmup (step < warmup), t=0 so β stays at β_init.
        return jnp.clip((step.astype(jnp.float32) - warmup) / decay_span, 0.0, 1.0)

    if schedule == "linear":
        def fn(step):
            return jnp.float32(init) + jnp.float32(final - init) * _t(step)
        return fn

    if schedule == "exp":
        # Geometric interpolation: log-linear from init to final. Matches the
        # log-spaced β grid we use in sweeps (1e-4, 1e-5, 1e-6, …) — moving by
        # equal log-distances per training step.
        log_init = jnp.log(jnp.float32(init))
        log_final = jnp.log(jnp.float32(final))
        def fn(step):
            return jnp.exp(log_init + (log_final - log_init) * _t(step))
        return fn

    if schedule == "cosine":
        def fn(step):
            return jnp.float32(final) + 0.5 * jnp.float32(init - final) * (
                1.0 + jnp.cos(jnp.pi * _t(step))
            )
        return fn

    # Unreachable due to early validation, but keeps mypy/static-analysis happy.
    raise ValueError(f"Unhandled beta_schedule={schedule!r}")
