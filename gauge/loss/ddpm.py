"""Epsilon-prediction DDPM loss."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gauge.loss import noise


def make_ddpm_loss(loss_cfg, sigmas, apply_fn):
    """Return a jit-friendly DDPM loss function."""

    def loss_fn(params, clean_data, class_l, key):
        batch_size = clean_data.shape[0]
        key_sigma, key_noise = jax.random.split(key)

        sigma_vals, _ = noise.sample_sigmas(key_sigma, sigmas, batch_size)
        sigma_vals = sigma_vals.astype(clean_data.dtype)
        eps = jax.random.normal(
            key_noise, clean_data.shape, dtype=clean_data.dtype)

        alpha_bar = noise.sigma_to_alpha_bar(sigma_vals)
        clean_scale = noise.broadcast_to_match(jnp.sqrt(alpha_bar), clean_data)
        noise_scale = noise.broadcast_to_match(
            jnp.sqrt(1.0 - alpha_bar), clean_data)
        x_t = clean_scale * clean_data + noise_scale * eps

        pred_eps = apply_fn(params, x_t, sigma_vals, class_l)
        loss = jnp.mean(jnp.square(pred_eps - eps))
        return loss

    return loss_fn
