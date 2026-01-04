
"""Sampling utilities used during integration tests or quick eval runs."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from gauge.config.config import Config
from gauge.loss.noise import NoiseSchedule


def _clip_if_needed(x, clip_bounds):
    if clip_bounds is None:
        return x
    return jnp.clip(x, clip_bounds[0], clip_bounds[1])


def _sample_ddim(apply_fn, x, labels, sigmas, alpha_bar, alphas,
                 sample_shape, clip_bounds, key, return_traj=False, use_tqdm=True):
    """Run deterministic (or eta-controlled) DDIM sampling."""
    n_samples = sample_shape[0]

    traj = [] if return_traj else None
    if traj is not None:
        traj.append(np.asarray(x))

    idx_range = range(sigmas.shape[0] - 1, -1, -1)
    if use_tqdm:
        idx_range = tqdm(idx_range, desc="sampling model", colour='magenta')
    for idx in idx_range:
        # sigma_t = sigmas[idx]
        alpha_bar_t = alpha_bar[idx]
        alpha_bar_prev_t = alpha_bar[idx -
                                     1] if idx > 0 else jnp.array(1.0)

        time = idx / (sigmas.shape[0] - 1)
        time_inp = jnp.full((n_samples, 1), time)
        eps_pred = apply_fn(x, time_inp, labels)

        pred_x0 = (x - jnp.sqrt(1 - alpha_bar_t) *
                   eps_pred) / jnp.sqrt(alpha_bar_t)
        x = jnp.sqrt(alpha_bar_prev_t) * pred_x0 + \
            jnp.sqrt(1 - alpha_bar_prev_t) * eps_pred

        x = _clip_if_needed(x, clip_bounds)

        if traj is not None:
            traj.append(np.asarray(x))

    np.stack(traj, axis=0)
    return x, traj


def _renormalize_to_uint8(arr):
    """Convert data in [-1, 1] to uint8 [0, 255]."""
    arr = np.asarray(arr)
    arr = np.clip(arr, -1.0, 1.0)
    arr = (arr + 1.0) / 2.0
    arr = np.rint(arr * 255.0)
    return arr.astype(np.uint8)


def _sample_dsm(
    apply_fn,
    x,
    labels,
    schedule: NoiseSchedule,
    n_steps,
    clip_range,
    key,
    return_traj=False,
    *,
    solver=None,
    rtol=1e-3,
    atol=1e-3,
):
    """Sample from the VP probability-flow ODE using a DSM-trained score model.

    Args:
        apply_fn: Callable (x_t, t, labels) -> score estimate. `t` has shape (B, 1).
                  Typically a params-partially-applied model.apply.
        x: Initial state at maximum time (t=1.0), e.g. standard Gaussian noise.
        labels: Conditioning labels, broadcastable to x's batch dimension.
        n_steps: If not None, use a fixed-step solver with this many steps.
                 If None, use an adaptive solver with (rtol, atol).
        clip_range: Optional (lo, hi) tuple to clip x when evaluating the model
                    and at the final sample.
        key: PRNGKey. Currently unused (kept purely for API compatibility).
        return_traj: If True, also return the full trajectory of states.
        solver: Optional diffrax solver instance. Defaults to Euler for fixed-step
                and Dopri5 for adaptive.
        rtol, atol: Tolerances for the adaptive solver.
        use_tqdm: If True AND n_steps is not None, use a simple Python-level
                  Euler integrator with a tqdm progress bar instead of diffrax.
                  This is mainly for debugging (slower, non-JIT).

    Returns:
        x_0 or (x_0, traj), where x_0 is approximately the sample at t ≈ 0.
    """

    # We treat x as x_{t=1} (noisiest) and integrate backwards to t ~ 0.
    t_start = 1.0
    t_end = 1e-3  # match training t_min; close to 0 but avoids totally unseen times

    if solver is None:
        if n_steps == -1 or n_steps is None:
            solver = dfx.Dopri5()  # adaptive default
        else:
            solver = dfx.Euler()   # fixed-step default

    batch_size = x.shape[0]

    # if labels is not None:
    #     labels = jnp.full((batch_size, 1), labels)

    def ode_fn(t, y, args):
        t_vec = jnp.full((batch_size, 1), t)
        t_snr = schedule.get_logsnr_time(t_vec)
        score_pred = apply_fn(y, t_snr, labels)
        drift = schedule.pf_ode_vt(y, t, score_pred)
        return drift
    # -------------------------------------------------------------------------
    # diffrax-based path (recommended for real sampling).
    # -------------------------------------------------------------------------
    if n_steps == -1 or n_steps is None:
        # Adaptive step size
        stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
        dt0 = (t_end - t_start) / 500.0  # just an initial guess (negative)
        max_steps = 1000
        saveat = dfx.SaveAt(steps=True)
    else:
        # Fixed number of (approximate) steps
        dt0 = (t_end - t_start) / float(n_steps)  # negative
        stepsize_controller = dfx.ConstantStepSize()
        max_steps = int(n_steps)*2  # a bit of slack, like your original
        ts = jnp.linspace(t_start, t_end, n_steps)
        saveat = dfx.SaveAt(ts=ts)

    term = dfx.ODETerm(ode_fn)

    sol = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=t_start,
        t1=t_end,
        dt0=dt0,
        y0=x,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        saveat=saveat
    )
    print(f'integrated w/ num_steps: {sol.stats["num_steps"]}')
    sol = sol.ys
    x_final = sol[-1]
    if clip_range is not None:
        lo, hi = clip_range
        x_final = jnp.clip(x_final, lo, hi)

    if not return_traj:
        return x_final

    traj = np.swapaxes(sol, 0, 1)

    return x_final, traj


def sample_model(cfg: Config, noise_schedule, apply_fn, n_samples, n_steps, data_shape, key, class_l=None, renormalize=True):
    """Sample batches from a trained DDPM/DDIM model.

    Args:
        renormalize: If True, map outputs from [-1, 1] to uint8 [0, 255].
    """
    method = cfg.score.method
    clip_range = cfg.integrate.clip
    n_samples = int(n_samples)
    data_shape = tuple(data_shape)
    sample_shape = (n_samples,) + data_shape

    key, noise_key = jax.random.split(key)
    x = jax.random.normal(noise_key, sample_shape)

    traj = None
    if method == "dsm":

        x, traj = _sample_dsm(
            apply_fn,
            x,
            class_l,
            noise_schedule,
            n_steps,
            clip_range,
            key,
            return_traj=True,
            solver=None,
            rtol=1e-3,
            atol=1e-3,
        )

    x = np.asarray(x)

    if renormalize:
        x = _renormalize_to_uint8(x)
        traj = _renormalize_to_uint8(traj)

    return x, traj
