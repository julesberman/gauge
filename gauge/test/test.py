

import jax
import matplotlib.pyplot as plt
import numpy as np

import gauge.io.result as R
from gauge.config.config import Config, get_outpath
from gauge.test.integrate import sample_model
from gauge.utils.plot import plot_grid, plot_grid_movie, scatter_movie


def run_test(cfg: Config, apply_fn, noise_schedule, dataset, data_shape, key, name=''):

    out_path = get_outpath()

    plot = cfg.test.plot
    n_steps = cfg.test.n_steps

    print("sampling score model...")
    for steps in n_steps:

        samples, trajectories = sample_model(cfg, noise_schedule, apply_fn,
                                             cfg.test.n_samples, steps, data_shape, key, class_l=None, renormalize=cfg.test.renormalize)

        trajectories = trajectories[:cfg.test.n_trajectories]

        samples = np.nan_to_num(samples, nan=0.0, posinf=1e9, neginf=-1e9)
        trajectories = np.nan_to_num(
            trajectories, nan=0.0, posinf=1e9, neginf=-1e9)

        if cfg.test.save_samples:
            R.RESULT[f'{name}_{steps}_samples'] = samples
        if cfg.test.save_trajectories:
            R.RESULT[f'{name}_{steps}_trajectories'] = trajectories

        if plot:
            if 'toy' in cfg.dataset:
                scatter_movie(np.swapaxes(trajectories, 0, 1), show=False, alpha=0.3,
                              save_to=out_path / f'{name}_{steps}_traj.gif')

                idx = jax.random.randint(key, shape=(
                    2048,), minval=0, maxval=cfg.test.n_samples)
                plt.scatter(x=samples[idx, 0], y=samples[idx, 1], alpha=0.3)
                plt.scatter(x=dataset[idx, 0], y=dataset[idx, 1], alpha=0.2)
                plt.savefig(out_path / f'{name}_{steps}_samples.png')
                plt.clf()

            else:
                plot_grid(samples[:36], show=False,
                          save_to=out_path / f'{name}_{steps}_samples.png')
                plot_grid_movie(trajectories, show=False,
                                save_to=out_path / f'{name}_{steps}_traj.gif')

    return samples
