SWEEP = {
    "dataset": "mnist",
    "optimizer.iters": "250_000",
    "net.arch": "unet",
    "net.size": "m",
    "sample.batch_size": "256",
    "score.kind": "vp, ve",
    "gauge.n_fields": "3, 6, 12",
    "gauge.ortho_a": "1e-1, 1, 10",
    "net.heads": "multi",
    # "gauge.ortho_loss": "cos, gauss"
}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 10,
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
