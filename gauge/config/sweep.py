SWEEP = {
    "dataset": "mnist",
    "optimizer.iters": "20_000, 100_000",
    "loss.method": 'dsm',
    "net.arch": "unet",
    "net.size": "m",
    "sample.batch_size": "256",
    "gauge.run": "True",
    "gauge.kinetic_a": "0",
    "gauge.gauge_a": "0",
    "gauge.compose": "True",
    "gnet.arch": "skew",
    "gnet.n_basis": "16,64",
    "gnet.kernel": "1,3,5",
    "load_score": "/scratch/jmb1174/sc/gauge/results/mnist/multi/run_g/0"
}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 5,
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
