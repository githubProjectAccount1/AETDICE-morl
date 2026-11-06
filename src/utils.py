import jax.numpy as jnp


def symlog(x):
    """Symmetric log: sign(x) · log(1 + |x|). Compresses wide dynamic range to
    bounded-ish output while preserving sign. Inverse of symexp."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


_SYMEXP_CLIP = 40.0   # symexp(40) ≈ 2.4e17; well under float32 max ~3.4e38 with margin


def symexp(x):
    """Symmetric exp: sign(x) · (exp(|x|) − 1). Inverse of symlog. Used to
    reparameterize a value/critic head so the NETWORK learns symlog(true_value),
    extending its effective dynamic range without forcing it to predict
    extreme magnitudes directly. DreamerV3-style. At init, raw≈0 → output≈0
    (vs plain exp, which gives exp(0)=1). Stable across small/large value
    regimes — useful for DICE critics where ν must be predicted precisely
    enough that e/β doesn't blow up at small β.

    NaN guard: hard-clip the raw input at ±40 before exp. Without this,
    raw output > ~88 overflows float32 → inf → NaN once it touches the loss.
    Clipping at 40 caps symexp output at ~2.4e17, comfortably under float32 max."""
    x = jnp.clip(x, -_SYMEXP_CLIP, _SYMEXP_CLIP)
    return jnp.sign(x) * (jnp.expm1(jnp.abs(x)))


def normalization(x, mean, std):
    x = (x - mean) / (std + 1e-8)
    return x


def min_max_normalization(x, min, max, eps=1e-8):
    x = (x - min) / (max - min + eps)
    return x
