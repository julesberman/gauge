SWEEP = {
    "dataset": "mnist",
    "optimizer.iters": "200_000",
    "loss.method": 'dsm',
    "net.arch": "unet",
    "net.size": "m",
    "sample.batch_size": "256",
    "score.kind": "ve, vp",
    "gauge.n_fields": "1, 4, 8, 16, 64",
    "gauge.ortho_a": "1e-3, 1e-2, 1e-1, 1"

}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 4,
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
