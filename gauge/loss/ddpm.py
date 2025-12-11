# """Epsilon-prediction DDPM loss."""

# from __future__ import annotations

# import jax
# import jax.numpy as jnp

# from gauge.loss import noise


# def make_ddpm_loss(loss_cfg, sigmas, apply_fn):
#     """Return a jit-friendly DDPM loss function."""

#     def loss_fn(params, clean_data, class_l, key):
#         batch_size = clean_data.shape[0]
#         key_sigma, key_noise = jax.random.split(key)

#         sigma_vals, _, time = noise.sample_sigmas(
#             key_sigma, sigmas, batch_size)
#         eps = jax.random.normal(
#             key_noise, clean_data.shape, dtype=clean_data.dtype)

#         alpha_bar = noise.sigma_to_alpha_bar(sigma_vals)
#         clean_scale = noise.broadcast_to_match(jnp.sqrt(alpha_bar), clean_data)
#         noise_scale = noise.broadcast_to_match(
#             jnp.sqrt(1.0 - alpha_bar), clean_data)
#         x_t = clean_scale * clean_data + noise_scale * eps

#         pred_eps = apply_fn(params, x_t, time, class_l)
#         loss = jnp.mean(jnp.square(pred_eps - eps))
#         return loss

#     return loss_fn


# def make_sigma_schedule(sigma_min, sigma_max, num_levels, schedule):
#     """Return a monotonic array of sigmas based on the requested schedule."""
#     sigma_min = jnp.asarray(sigma_min, dtype=jnp.float32)
#     sigma_max = jnp.asarray(sigma_max, dtype=jnp.float32)
#     num_levels = max(int(num_levels), 1)

#     if num_levels == 1:
#         return jnp.array([sigma_max], dtype=jnp.float32)

#     if schedule == "linear":
#         steps = jnp.linspace(0.0, 1.0, num_levels)
#     elif schedule == "cosine":
#         # Use the half-cosine that is common for beta schedules.
#         theta = jnp.linspace(0.0, jnp.pi / 2.0, num_levels)
#         steps = jnp.sin(theta)
#         steps = steps / steps[-1]
#     elif schedule == "geometric":
#         sigma_min = jnp.maximum(sigma_min, 1e-5)
#         ratio = (sigma_max / sigma_min) ** (1.0 / (num_levels - 1))
#         steps = ratio ** jnp.arange(num_levels)
#         return sigma_min * steps
#     else:
#         raise ValueError(f"Unknown schedule '{schedule}'")

#     return sigma_min + steps * (sigma_max - sigma_min)


# def sample_sigmas(key, sigmas, batch_size):
#     """Sample noise levels (sigmas) for each element in the batch."""
#     idx = jax.random.randint(key, (batch_size, 1), 0, sigmas.shape[0])

#     time = idx / (sigmas.shape[0] - 1)
#     return sigmas[idx], idx, time


# def sigma_to_alpha_bar(sigmas):
#     """Map sigmas to the DDPM cumulative product alpha_bar."""
#     return 1.0 / (1.0 + jnp.square(sigmas))


# def get_cofficients(sigmas):

#     alpha_bar = jnp.clip(
#         sigma_to_alpha_bar(sigmas), 1e-5, 1.0
#     )
#     alpha_bar_prev = jnp.concatenate(
#         [jnp.array([1.0]), alpha_bar[:-1]]
#     )
#     alphas = jnp.clip(alpha_bar / alpha_bar_prev, 1e-4, 0.9999)
#     betas = 1.0 - alphas

#     return alphas, alpha_bar, alpha_bar_prev, betas
