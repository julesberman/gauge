import os

import hydra
import jax
from jax import jit

import gauge.io.result as R
from gauge.config.config import Config
from gauge.config.setup import setup
from gauge.data.dataloader import get_dataloader
from gauge.data.get import get_dataset
from gauge.io.load import load
from gauge.io.save import save_results
from gauge.loss.noise import get_noise_schedule
from gauge.loss.score import get_score_loss
from gauge.net.get import get_network
from gauge.test.test import run_test
from gauge.train.train import train_model


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key, dataset, data_shape, dataloader, noise_schedule, score_loss, Score_net, S_params_init = build(
        cfg)

    key, key_G, key_test, key_opt = jax.random.split(key, num=4)

    # score model training and test
    if cfg.load_score != '':
        print(f"loading score model from {cfg.load_score}...")
        load_cfg, df = load(cfg.load_score)
        score_params = df['score_opt_params']
        R.RESULT["score_opt_params"] = score_params

        @jit
        def apply_score(*args):
            return Score_net.apply(score_params, *args)
    else:
        score_params = train_model(cfg, dataloader,
                                   score_loss, S_params_init, key_opt, has_aux=True, name='score')

        for f_n in range(cfg.gauge.n_fields):
            @jit
            def apply_score(*args, f_n=f_n):
                return Score_net.apply(score_params, *args)[..., f_n:f_n + 1]

            run_test(cfg, apply_score, noise_schedule, dataset, data_shape,
                     key_test, name=f'score_{f_n}')

    if cfg.gauge.run:
        pass

    save_results(R.RESULT, cfg)


def build(cfg: Config):

    key = setup(cfg)
    key, net_key, g_key = jax.random.split(key, num=3)

    dataset, data_shape = get_dataset(cfg)
    dataloader = get_dataloader(cfg, dataset)

    net, params_init = get_network(cfg, dataloader, net_key)

    noise_schedule = get_noise_schedule(cfg)

    score_loss = get_score_loss(cfg, noise_schedule, net)

    return key, dataset, data_shape, dataloader, noise_schedule, score_loss,  net, params_init


if __name__ == "__main__":
    # dont let tensorflow grab GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['TFDS_DATA_DIR'] = '/scratch/jmb1174/tensorflow_datasets'

    run()
