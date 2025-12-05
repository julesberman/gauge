
"""Sampling utilities used during integration tests or quick eval runs."""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from gauge.config.config import Config
from gauge.loss import noise


def _prepare_labels(class_l, n_samples: int):
    if class_l is None:
        return None
    arr = jnp.asarray(class_l)
    arr = jnp.squeeze(arr)
    if arr.ndim == 0:
        arr = jnp.full((n_samples,), arr.astype(jnp.int32), dtype=jnp.int32)
    else:
        if arr.ndim > 1:
            raise ValueError("class_l must be scalar or 1D.")
        if arr.shape[0] == 1 and n_samples > 1:
            arr = jnp.full((n_samples,), arr[0], dtype=arr.dtype)
        elif arr.shape[0] != n_samples:
            raise ValueError(
                f"class_l has {arr.shape[0]} entries but {n_samples} samples requested.")
        arr = arr.astype(jnp.int32)

    arr = arr.reshape(-1, 1)
    return arr


def _clip_if_needed(x, clip_bounds):
    if clip_bounds is None:
        return x
    return jnp.clip(x, clip_bounds[0], clip_bounds[1])


def _sample_ddpm(apply_fn, x, labels, sigmas, alpha_bar, alpha_bar_prev,
                 alphas, betas, sample_shape, clip_bounds, extra_std, dtype, key,
                 return_traj=False, use_tqdm=True):
    """Run ancestral DDPM sampling."""
    n_samples = sample_shape[0]
    traj = [] if return_traj else None
    if traj is not None:
        traj.append(np.asarray(x))

    idx_range = range(sigmas.shape[0] - 1, -1, -1)
    if use_tqdm:
        idx_range = tqdm(idx_range, desc="sampling model", colour='magenta')
    for idx in idx_range:
        sigma_t = sigmas[idx]
        alpha_bar_t = alpha_bar[idx]
        alpha_bar_prev_t = alpha_bar_prev[idx]
        alpha_t = alphas[idx]
        beta_t = betas[idx]

        time_inp = jnp.full((n_samples, 1), sigma_t, dtype=dtype)
        eps_pred = apply_fn(x, time_inp, labels)

        pred_coef = (1.0 - alpha_t) / jnp.sqrt(1.0 - alpha_bar_t)

        mean = (x - pred_coef * eps_pred) / jnp.sqrt(alpha_t)
        if idx > 0:
            beta_tilde = (1.0 - alpha_bar_prev_t) / \
                (1.0 - alpha_bar_t) * beta_t
            beta_tilde = jnp.maximum(beta_tilde, 1e-8)
            sigma = jnp.sqrt(beta_tilde)
            if extra_std is not None:
                sigma = jnp.sqrt(jnp.square(sigma) + jnp.square(extra_std))
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, sample_shape, dtype=dtype)
            x = mean + sigma * noise
        else:
            x = mean

        x = _clip_if_needed(x, clip_bounds)

        if traj is not None:
            traj.append(np.asarray(x))

    return x, key, traj


def _sample_ddim(apply_fn, x, labels, sigmas, alpha_bar, alphas,
                 sample_shape, clip_bounds, extra_std, dtype, key, return_traj=False, use_tqdm=True):
    """Run deterministic (or eta-controlled) DDIM sampling."""
    n_samples = sample_shape[0]
    eta = extra_std if extra_std is not None else jnp.asarray(0.0, dtype=dtype)
    traj = [] if return_traj else None
    if traj is not None:
        traj.append(np.asarray(x))

    idx_range = range(sigmas.shape[0] - 1, -1, -1)
    if use_tqdm:
        idx_range = tqdm(idx_range, desc="sampling model", colour='magenta')
    for idx in idx_range:
        sigma_t = sigmas[idx]
        alpha_bar_t = alpha_bar[idx]
        alpha_bar_prev_t = alpha_bar[idx -
                                     1] if idx > 0 else jnp.array(1.0, dtype=dtype)
        alpha_t = alphas[idx]

        time_inp = jnp.full((n_samples, 1), sigma_t, dtype=dtype)
        eps_pred = apply_fn(x, time_inp, labels)

        sigma_eta = eta * \
            jnp.sqrt((1 - alpha_bar_prev_t) /
                     (1 - alpha_bar_t) * (1 - alpha_t))
        pred_x0 = (x - jnp.sqrt(1 - alpha_bar_t) *
                   eps_pred) / jnp.sqrt(alpha_bar_t)
        x = jnp.sqrt(alpha_bar_prev_t) * pred_x0 + \
            jnp.sqrt(1 - alpha_bar_prev_t - sigma_eta**2) * eps_pred

        if idx > 0:
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, sample_shape, dtype=dtype)
            x = x + sigma_eta * noise

        x = _clip_if_needed(x, clip_bounds)

        if traj is not None:
            traj.append(np.asarray(x))

    return x, key, traj


def _renormalize_to_uint8(arr):
    """Convert data in [-1, 1] to uint8 [0, 255]."""
    arr = np.asarray(arr)
    arr = np.clip(arr, -1.0, 1.0)
    arr = (arr + 1.0) / 2.0
    arr = np.rint(arr * 255.0)
    return arr.astype(np.uint8)


def sample_model(cfg: Config, apply_fn, n_samples, n_steps, data_shape, key, class_l=None,
                 return_trajectory=False, renormalize=True):
    """Sample batches from a trained DDPM/DDIM model.

    Args:
        renormalize: If True, map outputs from [-1, 1] to uint8 [0, 255].
    """
    method = cfg.integrate.method
    integr_cfg = cfg.integrate
    loss_cfg = cfg.loss
    dtype = jnp.float64 if cfg.x64 else jnp.float32
    n_samples = int(n_samples)

    data_shape = tuple(data_shape)
    sample_shape = (n_samples,) + data_shape
    labels = _prepare_labels(class_l, n_samples)

    sigmas = noise.make_sigma_schedule(
        loss_cfg.sigma_min,
        loss_cfg.sigma_max,
        n_steps,
        loss_cfg.schedule,
    ).astype(dtype)

    alphas, alpha_bar, alpha_bar_prev, betas = noise.get_cofficients(sigmas)

    extra_std = integr_cfg.var

    clip_range = getattr(integr_cfg, "clip", (-1.0, 1.0))

    key, noise_key = jax.random.split(key)
    x = jax.random.normal(noise_key, sample_shape, dtype=dtype)

    traj = None
    if method == "ddim":
        x, _, traj = _sample_ddim(
            apply_fn,
            x,
            labels,
            sigmas,
            alpha_bar,
            alphas,
            sample_shape,
            clip_range,
            extra_std,
            dtype,
            key,
            return_traj=return_trajectory,
        )
    else:
        x, _, traj = _sample_ddpm(
            apply_fn,
            x,
            labels,
            sigmas,
            alpha_bar,
            alpha_bar_prev,
            alphas,
            betas,
            sample_shape,
            clip_range,
            extra_std,
            dtype,
            key,
            return_traj=return_trajectory,
        )

    x = np.asarray(x)

    if renormalize:
        x = _renormalize_to_uint8(x)

    if return_trajectory:
        traj_stack = np.stack(traj, axis=0) if traj else None
        traj_stack = np.swapaxes(traj_stack, 0, 1)
        if renormalize and traj_stack is not None:
            traj_stack = _renormalize_to_uint8(traj_stack)
        return x, traj_stack

    return x
