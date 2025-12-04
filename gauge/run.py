import os

import hydra
import jax
import jax.numpy as jnp
from jax import jit

import gauge.io.result as R
from gauge.config.config import Config
from gauge.config.setup import setup
from gauge.data.dataloader import get_dataloader
from gauge.data.dataset import get_dataset
from gauge.io.load import load
from gauge.io.save import save_results
from gauge.loss.gauge import get_gauge_loss
from gauge.loss.noise import sigma_to_alpha_bar
from gauge.loss.score import get_score_loss
from gauge.net.get import get_network
from gauge.test.test import run_test
from gauge.train.train import train_model


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key, dataset, data_shape, dataloader, net, score_loss, sigmas, params_init = build(
        cfg)

    Score_net, G_net = net, net

    key, key_G, key_test, key_opt = jax.random.split(key, num=4)

    # score model training and test
    if cfg.load_score != '':
        print(f"loading score model from {cfg.load_score}...")
        load_cfg, df = load(cfg.load_score)
        score_params = df['score_opt_params']
    else:
        score_params = train_model(cfg, Score_net, dataloader,
                                   score_loss, params_init, key_opt, name='score')

    @jit
    def apply_score(*args):
        return Score_net.apply(score_params, *args)
    run_test(cfg, apply_score, dataset, data_shape, key_test, name='score')

    if cfg.gauge.run:
        # do gauge model training and test
        gauge_loss = get_gauge_loss(cfg, G_net, apply_score, sigmas)
        G_params = train_model(cfg, G_net, dataloader,
                               gauge_loss, params_init, key_G, name='gauge')

        @jit
        def apply_K(x, time_inp, labels):
            alpha_bar = sigma_to_alpha_bar(jnp.squeeze(time_inp))
            beta = jnp.sqrt(1-alpha_bar)
            beta = beta[:, None, None, None]
            return Score_net.apply(score_params, x, time_inp, labels) - beta * G_net.apply(G_params, x, time_inp, labels)

        run_test(cfg, apply_K,
                 dataset, data_shape, key_test, name='gauge')

    save_results(R.RESULT, cfg)


def build(cfg: Config):

    key = setup(cfg)
    key, net_key = jax.random.split(key, num=2)

    dataset, data_shape = get_dataset(cfg)
    dataloader = get_dataloader(cfg, dataset)

    net, params_init = get_network(cfg, dataloader, net_key)
    score_loss, sigmas = get_score_loss(cfg, net)

    return key, dataset, data_shape, dataloader, net, score_loss, sigmas, params_init


if __name__ == "__main__":
    # dont let tensorflow grab GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['TFDS_DATA_DIR'] = '/scratch/jmb1174/tensorflow_datasets'

    run()
