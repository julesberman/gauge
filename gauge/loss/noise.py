"""Noise schedule utilities for score-model losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_sigma_schedule(sigma_min, sigma_max, num_levels, schedule):
    """Return a monotonic array of sigmas based on the requested schedule."""
    sigma_min = jnp.asarray(sigma_min, dtype=jnp.float32)
    sigma_max = jnp.asarray(sigma_max, dtype=jnp.float32)
    num_levels = max(int(num_levels), 1)

    if num_levels == 1:
        return jnp.array([sigma_max], dtype=jnp.float32)

    if schedule == "linear":
        steps = jnp.linspace(0.0, 1.0, num_levels)
    elif schedule == "cosine":
        # Use the half-cosine that is common for beta schedules.
        theta = jnp.linspace(0.0, jnp.pi / 2.0, num_levels)
        steps = jnp.sin(theta)
        steps = steps / steps[-1]
    elif schedule == "geometric":
        sigma_min = jnp.maximum(sigma_min, 1e-5)
        ratio = (sigma_max / sigma_min) ** (1.0 / (num_levels - 1))
        steps = ratio ** jnp.arange(num_levels)
        return sigma_min * steps
    else:
        raise ValueError(f"Unknown schedule '{schedule}'")

    return sigma_min + steps * (sigma_max - sigma_min)


def sample_sigmas(key, sigmas, batch_size):
    """Sample noise levels (sigmas) for each element in the batch."""
    idx = jax.random.randint(key, (batch_size, 1), 0, sigmas.shape[0])
    return sigmas[idx], idx


def broadcast_to_match(values, ref):
    """Broadcast 1D per-example values to match `ref`'s rank."""
    if values.ndim >= ref.ndim:
        return values
    new_shape = values.shape + (1,) * (ref.ndim - values.ndim)
    return jnp.reshape(values, new_shape)


def sigma_to_alpha_bar(sigmas):
    """Map sigmas to the DDPM cumulative product alpha_bar."""
    return 1.0 / (1.0 + jnp.square(sigmas))


def get_cofficients(sigmas):

    alpha_bar = jnp.clip(
        sigma_to_alpha_bar(sigmas), 1e-5, 1.0
    )
    alpha_bar_prev = jnp.concatenate(
        [jnp.array([1.0]), alpha_bar[:-1]]
    )
    alphas = jnp.clip(alpha_bar / alpha_bar_prev, 1e-4, 0.9999)
    betas = 1.0 - alphas

    return alphas, alpha_bar, alpha_bar_prev, betas
