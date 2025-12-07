"""Denoising score matching (DSM) loss utilities."""

import jax
import jax.numpy as jnp

from gauge.loss import noise


def make_dsm_loss(loss_cfg, sigmas, apply_fn):
    """Return a DSM loss aligned with the config schedule."""

    def loss_fn(params, x0, class_l, key):
        batch_size = x0.shape[0]
        key_sigma, key_noise = jax.random.split(key)

        sigma_vals, t_vals = noise.sample_vp_sde_sigmas(key_sigma, batch_size)
        sigma_bc = noise.broadcast_to_match(sigma_vals, x0)

        # Recover alpha from sigma: sigma^2 = 1 - alpha^2
        alpha_vals = jnp.sqrt(1.0 - jnp.square(sigma_vals))
        alpha_bc = noise.broadcast_to_match(alpha_vals, x0)

        eps = jax.random.normal(key_noise, x0.shape)
        x_t = alpha_bc * x0 + sigma_bc * eps

        # Target score for N(alpha x0, sigma^2 I)
        target = -(x_t - alpha_bc * x0) / jnp.square(sigma_bc)

        pred_score = apply_fn(params, x_t, t_vals, class_l)

        sq_error = jnp.square(pred_score - target)
        reduce_dims = tuple(range(1, sq_error.ndim))
        per_example = jnp.mean(sq_error, axis=reduce_dims)

        weights = jnp.squeeze(jnp.square(sigma_vals))
        loss = jnp.mean(per_example * weights)

        return loss

    return loss_fn
