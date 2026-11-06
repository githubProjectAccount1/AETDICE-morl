from flax import nnx
from src.nets.policy import MLP, RaccEncoder
import jax.numpy as jnp
from src.utils import symlog, symexp


def _apply_nu_transform(x, kind):
    """Apply optional reparameterization to ν network output.
    'none': identity (default — back-compat).
    'symexp': ν = symexp(raw); network effectively learns symlog(true_ν).
              DreamerV3-style; helps when ν has wide dynamic range and small β
              would otherwise make e/β explode (because raw_ν predictions are
              imprecise at extreme magnitudes)."""
    if kind == "symexp":
        return symexp(x)
    return x


def _make_mask(horizon):
    """마지막 column(=terminal) 을 0 으로 마스크. Critic/FairDICECritic/QCritic 공유."""
    return jnp.ones((horizon + 1,)).at[-1].set(0.0)


class QCritic(nnx.Module):
    """Per-timestep Q(s, a, racc) for continuous actions.

    Mirrors `Critic`'s shape (state + racc → [B, H+1] with last column masked to 0)
    but additionally concatenates the continuous action vector into the input. Used
    by the IQL learner where the Bellman target is r·μ + V(s'); Q never has to be
    queried at an OOD action because V handles the max.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        reward_dim,
        racc_mean,
        racc_std,
        horizon,
        hidden_dims=[256, 256],
        layer_norm: bool = False,
        racc_dim=None,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.horizon = horizon
        self.racc_encoder = RaccEncoder(racc_mean, racc_std)
        # state + continuous action + racc embedding (RaccEncoder is identity-shape).
        # racc_dim = F̃-output dim; defaults to reward_dim for pointwise F where the
        # two coincide. CobbDouglas always emits F̃-dim=2 regardless of reward_dim,
        # so safe_cd (reward_dim=3) must pass racc_dim=2 explicitly.
        if racc_dim is None:
            racc_dim = reward_dim
        input_dim = state_dim + action_dim + racc_dim
        self.backbone = MLP(
            din=input_dim,
            hidden_dims=hidden_dims,
            dout=horizon + 1,
            rngs=rngs,
            layer_norm=layer_norm,
        )
        self.mask = _make_mask(self.horizon)

    def __call__(self, states, actions, raccs, timesteps=None):
        racc_embedding = self.racc_encoder(raccs)
        fused_feature = jnp.concatenate([states, actions, racc_embedding], axis=-1)
        out = self.backbone(fused_feature) * self.mask
        if timesteps is not None:
            t = timesteps.astype(jnp.int32).reshape(-1, 1)
            return jnp.take_along_axis(out, t, axis=1).squeeze(1)
        return out


# TODO: change the name of the class from FairDICECritic to MOMDPCritic
class FairDICECritic(nnx.Module):
    """Critic for FairDICE finite-horizon (SER) — state-only input, no racc conditioning."""

    def __init__(
        self,
        state_dim,
        horizon,
        hidden_dims=[256, 256],
        layer_norm: bool = False,
        nu_output_transform: str = "none",
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.horizon = horizon
        self.nu_output_transform = nu_output_transform
        self.backbone = MLP(din=state_dim, hidden_dims=hidden_dims, dout=horizon + 1,
                            rngs=rngs, layer_norm=layer_norm)
        self.mask = _make_mask(horizon)

    def __call__(self, states, timesteps=None):
        feature = _apply_nu_transform(self.backbone(states), self.nu_output_transform)
        out = feature * self.mask

        if timesteps is not None:
            t = timesteps.astype(jnp.int32).reshape(-1, 1)
            return jnp.take_along_axis(out, t, axis=1).squeeze(1)
        return out

# TODO: change class name to augmentedMOMDPcritic
# TODO: 1.state encoder and racc encoder, 2.backbone -> head, (horizon+1)
class Critic(nnx.Module):
    """AETDICE ν critic: V(s, racc) → [B, H+1] (타임스텝당 값 1개, 마지막 column 은 0 마스크).

    backbone(state ⊕ racc_norm) 이 `[B, H+1]` 을 wide Linear head 로 바로 낸다 — 각 column 이
    독립 출력이라 네트워크가 column 0 을 나머지와 무관하게 움직일 수 있다. racc 는 z-score
    정규화만 거친 raw 값이다 (`RaccEncoder`).
    """

    def __init__(
        self,
        state_dim,
        reward_dim,
        racc_mean,
        racc_std,
        horizon,
        hidden_dims=[256, 256],
        layer_norm: bool = False,
        racc_dim=None,
        nu_output_transform: str = "none",
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.horizon = horizon
        # 'none' (default — back-compat) or 'symexp' to reparameterize ν as
        # symexp(raw); see _apply_nu_transform for rationale.
        self.nu_output_transform = nu_output_transform

        self.racc_encoder = RaccEncoder(racc_mean, racc_std)

        # racc_dim = F̃-output dim flowing into the network. Defaults to reward_dim
        # for pointwise F where they coincide; set explicitly for CobbDouglas
        # (F̃-dim=2 regardless of raw reward_dim).
        if racc_dim is None:
            racc_dim = reward_dim
        input_dim = state_dim + racc_dim

        self.backbone = MLP(din=input_dim, hidden_dims=hidden_dims, dout=horizon + 1,
                            rngs=rngs, layer_norm=layer_norm)
        self.mask = _make_mask(horizon)

    def __call__(self, states, raccs, timesteps=None):
        racc_embedding = self.racc_encoder(raccs)
        fused_feature = jnp.concatenate([states, racc_embedding], axis=-1)
        feature = _apply_nu_transform(self.backbone(fused_feature), self.nu_output_transform)

        out = feature * self.mask

        if timesteps is not None:
            t = timesteps.astype(jnp.int32).reshape(-1, 1)
            val = jnp.take_along_axis(out, t, axis=1).squeeze(1)
            return val
        else:
            return out
