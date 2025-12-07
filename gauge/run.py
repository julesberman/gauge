import os

import hydra
import jax
from jax import jit

import gauge.io.result as R
from gauge.config.config import Config
from gauge.config.setup import setup
from gauge.data.dataloader import get_dataloader
from gauge.data.dataset import get_dataset
from gauge.io.load import load
from gauge.io.save import save_results
from gauge.loss.gauge import get_combine_V, get_gauge_loss
from gauge.loss.score import get_score_loss
from gauge.net.get import get_network
from gauge.test.test import run_test
from gauge.train.train import train_model


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key, dataset, data_shape, dataloader, score_loss, sigmas, Score_net, S_params_init, G_net, G_params_init = build(
        cfg)

    key, key_G, key_test, key_opt = jax.random.split(key, num=4)

    # score model training and test
    if cfg.load_score != '':
        print(f"loading score model from {cfg.load_score}...")
        load_cfg, df = load(cfg.load_score)
        score_params = df['score_opt_params']
        R.RESULT["score_opt_params"] = score_params
    else:
        score_params = train_model(cfg, Score_net, dataloader,
                                   score_loss, S_params_init, key_opt, name='score')

    @jit
    def apply_score(*args):
        return Score_net.apply(score_params, *args)
    run_test(cfg, apply_score, dataset, data_shape, key_test, name='score')

    if cfg.gauge.run:

        if cfg.gauge.compose:
            def apply_G(params, x_t, t_vals, class_l):
                s_t = Score_net.apply(score_params, x_t, t_vals, class_l)
                return G_net.apply(params, s_t, t_vals, class_l)
        else:
            apply_G = G_net.apply

        # do gauge model training and test
        gauge_loss = get_gauge_loss(cfg, apply_G, apply_score, sigmas)
        G_params = train_model(cfg, dataloader, gauge_loss,
                               G_params_init, key_G, name='gauge')

        apply_V = get_combine_V(cfg, Score_net, score_params, G_net, G_params)

        run_test(cfg, apply_V,
                 dataset, data_shape, key_test, name='gauge')

    save_results(R.RESULT, cfg)


def build(cfg: Config):

    key = setup(cfg)
    key, net_key, g_key = jax.random.split(key, num=3)

    dataset, data_shape = get_dataset(cfg)
    dataloader = get_dataloader(cfg, dataset)

    net, params_init = get_network(cfg.net, dataloader, net_key)
    G_net, G_params_init = get_network(cfg.gnet, dataloader, g_key)
    score_loss, sigmas = get_score_loss(cfg, net)

    return key, dataset, data_shape, dataloader, score_loss, sigmas,  net, params_init, G_net, G_params_init


if __name__ == "__main__":
    # dont let tensorflow grab GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['TFDS_DATA_DIR'] = '/scratch/jmb1174/tensorflow_datasets'

    run()
