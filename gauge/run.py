import os

import hydra
import jax

import gauge.io.result as R
from gauge.config.config import Config
from gauge.config.setup import setup
from gauge.data.dataloader import get_dataloader
from gauge.data.dataset import get_dataset
from gauge.io.save import save_results
from gauge.loss.score import get_score_loss
from gauge.net.get import get_network
from gauge.test.test import run_test
from gauge.train.train import train_model


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key, dataset, dataloader, net, score_loss, sigmas, params_init = build(cfg)

    key, key_train, key_test, key_opt = jax.random.split(key, num=4)

    opt_params = train_model(cfg, net, dataloader,
                             score_loss, params_init, key_opt)

    samples = run_test(cfg, net, opt_params, dataset, key_test)

    save_results(R.RESULT, cfg)

    return samples


def build(cfg: Config):

    key = setup(cfg)
    key, net_key = jax.random.split(key, num=2)

    dataset, data_shape = get_dataset(cfg)
    dataloader = get_dataloader(cfg, dataset)

    net, params_init = get_network(cfg, dataloader, net_key)
    score_loss, sigmas = get_score_loss(cfg, net)

    return key, dataset, dataloader, net, score_loss, sigmas, params_init


if __name__ == "__main__":
    # dont let tensorflow grab GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['TFDS_DATA_DIR'] = '/scratch/jmb1174/tensorflow_datasets'

    run()
