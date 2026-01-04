SWEEP = {
    "dataset": "mnist",
    "net.size": "m",
    "optimizer.iters": "100_000",
    "score.kind": "vp",
    "score.target": "noise",
    "gauge.n_fields": "3",
    "gauge.var_a": "0, 1",
    "net.heads": "multi",
    "gauge.var_features": "v_err",
    "gauge.var_loss": "cos",
    "gauge.resample": "fixed, step",
    "gauge.n_functions": "0, 1, 2, 4, 8, 16, 32, 64, 5000",
    "gauge.omegas": "one",
    "gauge.weighted": "one",
}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 5,
    "cpus_per_task": 16,
    "mem_gb": 300,
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
