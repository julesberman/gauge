"""Denoising score matching (DSM) loss utilities."""

import jax
import jax.numpy as jnp

from gauge.loss import noise


def make_dsm_loss(loss_cfg, sigmas, apply_fn):
    """Return a DSM loss aligned with the config schedule."""

    def loss_fn(params, x0, class_l, key):
        batch_size = x0.shape[0]
        x_shape = x0.shape
        x0 = x0.reshape(batch_size, -1)

        key_sigma, key_noise = jax.random.split(key)
        sigmas, t_vals = noise.sample_vp_sde_sigmas(key_sigma, batch_size)
        alphas = jnp.sqrt(1.0 - jnp.square(sigmas))

        eps = jax.random.normal(key_noise, x0.shape)
        x_t = alphas * x0 + sigmas * eps

        pred_score = apply_fn(params, x_t.reshape(
            x_shape), t_vals, class_l).reshape(batch_size, -1)

        # Target score for N(alpha x0, sigma^2 I)
        target = -(x_t - alphas * x0) / jnp.square(sigmas)

        sq_error = jnp.mean(jnp.square(pred_score - target), axis=-1)
        weights = jnp.squeeze(jnp.square(sigmas))
        loss = jnp.mean(sq_error * weights)

        return loss

    return loss_fn
