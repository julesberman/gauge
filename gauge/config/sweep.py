SWEEP = {
    "data.dataset": "mnist",
    "optimizer.iters": "100_000",
    "net.arch": "unet",
    "net.size": "l",
}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 8,
    "cpus_per_task": 16,
    "mem_gb": 100,
    "gres": "gpu:h100:1",
    "account": "extremedata",
}


SLURM_CONFIG_L = {
    "timeout_min": 60 * 8,
    "cpus_per_task": 16,
    "mem_gb": 500,
    "gres": "gpu:h100:4",
    "account": "extremedata",
}


def get_slurm_config():

    return SLURM_CONFIG_M
