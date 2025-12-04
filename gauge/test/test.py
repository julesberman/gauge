

import gauge.io.result as R
from gauge.config.config import Config, get_outpath
from gauge.test.integrate import sample_model
from gauge.utils.plot import plot_grid, plot_grid_movie


def run_test(cfg: Config, apply_fn, dataset, data_shape, key, name=''):

    out_path = get_outpath()

    plot = cfg.test.plot
    n_steps = cfg.test.n_steps

    print("sampling score model...")
    for steps in n_steps:

        samples, trajectories = sample_model(cfg, apply_fn,
                                             cfg.test.n_samples, steps, data_shape, key, class_l=None, return_trajectory=True)
        trajectories = trajectories[:cfg.test.n_trajectories]

        if cfg.test.save_samples:
            R.RESULT[f'{name}_{steps}_samples'] = samples
        if cfg.test.save_trajectories:
            R.RESULT[f'{name}_{steps}_trajectories'] = trajectories

        if plot:
            plot_grid(samples[:36], show=False,
                      save_to=out_path / f'{name}_{steps}_samples.png')
            plot_grid_movie(trajectories, show=False,
                            save_to=out_path / f'{name}_{steps}_traj.gif')

    return samples
