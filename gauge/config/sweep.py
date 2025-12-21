SWEEP = {
    "dataset": "mnist",
    "net.size": "m",
    "optimizer.iters": "250_000",
    "score.kind": "vp",
    "gauge.n_fields": "3",
    "gauge.var_a": "1",
    "net.heads": "multi",
    "gauge.var_features": "v_err",
    "gauge.var_loss": "cos",
    "score.fixed_t": "True",
    "gauge.u_stat": "True",
    "gauge.n_functions": "64, 512, 1024, 2048, 5000, 10_000",
}


# SWEEP = {
#     "dataset": "toy_swiss, toy_checker",
#     "net.size": "m",
#     "optimizer.iters": "250_000",
#     "score.kind": "vp",
#     "gauge.n_fields": "3, 8",
#     "gauge.var_a": "1",
#     "net.heads": "multi",
#     "gauge.var_features": "v_err, proj_err",
#     "gauge.var_loss": "cos",
#     "score.fixed_t": "True, False",
#     "gauge.u_stat": "True, False"

# }


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 16,
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
