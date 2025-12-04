SWEEP = {
    "dataset": "cifar10",
    "optimizer.iters": "350_000",
    "loss.method": 'ddpm',
    "integrate.method": 'ddim',
    "net.arch": "unet",
    "net.size": "m",
    "sample.batch_size": "256",
    "gauge.run": "true",
    "gauge.kinetic_a": "1.0,10.0"
}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 24,
    "cpus_per_task": 16,
    "mem_gb": 250,
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
