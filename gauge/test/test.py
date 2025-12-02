

import gauge.io.result as R
from gauge.config.config import Config, get_outpath
from gauge.test.eval import eval_samples
from gauge.test.integrate import sample_model
from gauge.utils.plot import plot_grid, plot_grid_movie


def run_test(cfg: Config, net, opt_params, dataset, data_shape, key):

    out_path = get_outpath()

    plot = cfg.test.plot
    samples = sample_model(cfg, net, opt_params,
                           cfg.test.n_samples, data_shape, class_l=None)

    if cfg.test.save_samples:
        R.RESULT['samples'] = samples

    eval_samples(cfg, samples)

    if plot:
        plot_grid(samples[:36], show=False,
                  save_to=out_path / 'samples.gif')

    if cfg.test.n_trajectories > 0:
        _, trajectories = sample_model(cfg, net, opt_params,
                                       cfg.test.n_trajectories, data_shape, class_l=None, return_trajectory=True)

        if cfg.test.save_trajectories:
            R.RESULT['trajectories'] = trajectories

        if plot:
            plot_grid_movie(trajectories, show=False,
                            save_to=out_path / 'traj.gif')

    return samples
