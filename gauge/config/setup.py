import random
import secrets

import gauge.io.result as R
import jax
from gauge.config.config import Config
from gauge.io.result import init_result
from omegaconf import OmegaConf


def setup(config: Config):

    # init global results obj
    init_result()

    # print config
    print("\nCONFIG")
    print(OmegaConf.to_yaml(config))

    print(f"name: {config.name}")

    if config.x64:
        jax.config.update("jax_enable_x64", True)
        print("enabling x64")
    else:
        jax.config.update("jax_enable_x64", False)

    if config.platform is not None:
        jax.config.update("jax_platform_name", config.platform)

    # # if oop error see: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    devices = jax.devices()
    platform = devices[0].platform if devices else "unknown"
    device_count = len(devices) if devices else 0
    print(f"platform: {platform} — device_count: {device_count}")

    # list of available devices (CPUs, GPUs, TPUs)
    for gpu in devices:
        print(f"host_{gpu.id}: {gpu.device_kind}")
        R.RESULT[f"host_{gpu.id}"] = gpu.device_kind

    # set random seed, if none use random random seed
    if config.seed == -1:
        config.seed = secrets.randbelow(1e5)
        print(f"seed: {config.seed}")
    seed = config.seed
    key = jax.random.PRNGKey(seed)
    random.seed(seed)

    return key
