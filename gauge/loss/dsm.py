"""Denoising score matching (DSM) loss utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gauge.loss import noise


def make_dsm_loss(loss_cfg, sigmas, apply_fn):
    """Return a DSM loss aligned with the config schedule."""

    def loss_fn(params, clean_data, class_l, key):
        batch_size = clean_data.shape[0]
        key_sigma, key_noise = jax.random.split(key)

        sigma_vals, _ = noise.sample_sigmas(key_sigma, sigmas, batch_size)
        sigma_vals = sigma_vals.astype(clean_data.dtype)
        sigma_bc = noise.broadcast_to_match(sigma_vals, clean_data)

        eps = jax.random.normal(
            key_noise, clean_data.shape, dtype=clean_data.dtype)
        noisy = clean_data + sigma_bc * eps

        # Target score for a Gaussian-corrupted sample.
        target = -(noisy - clean_data) / jnp.square(sigma_bc)
        pred_score = apply_fn(params, noisy, sigma_vals, class_l)

        sq_error = jnp.square(pred_score - target)
        reduce_dims = tuple(range(1, sq_error.ndim))
        per_example = jnp.mean(sq_error, axis=reduce_dims)
        weights = jnp.square(sigma_vals)
        loss = jnp.mean(per_example * weights)
        return loss

    return loss_fn
