from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from gauge.config.sweep import get_slurm_config, get_sweep
from gauge.utils.tools import epoch_time, unique_id


@dataclass
class Network:
    arch: str = "unet"
    size: str = "m"
    emb_features: list[int] = field(default_factory=lambda: [512, 512])


@dataclass
class Optimizer:
    lr: float = 2e-4
    iters: int = 20_000
    scheduler: str = 'cos'
    warm_up: bool = False
    optimizer: str = "adamw"


@dataclass
class Data:
    normalize: str = '-11'
    class_labels: bool = False


@dataclass
class Sample:
    batch_size: int = 1024
    shuffle: bool = True
    shuffle_buffer_size: Optional[int] = None
    channel_first: bool = False
    materialize: bool = True


@dataclass
class Integrate:
    # assume we have ddpm model, sample with ddpm or ddim
    method: str = "ddim"  # "ddpm", "ddim"
    clip: tuple = (-1, 1)  # clip data at each step
    var: float = 0.0  # varaince of noise in reverse process


@dataclass
class Loss:
    # ----------------------------------------------------------
    # Two supported loss types:
    #   - "ddpm" : epsilon-prediction MSE loss
    #   - "dsm"  : denoising score matching
    # ----------------------------------------------------------
    method: str = "ddpm"  # "ddpm", "dsm"

    # ----------------------------------------------------------
    # Shared noise parameters
    # ----------------------------------------------------------
    sigma_min: float = 0.01        # smallest noise level usable for both
    sigma_max: float = 1.0         # DSM uses full range; DDPM may ignore
    schedule: str = "linear"  # cosine, linear, geometric

    # Neutral name: used as timesteps (DDPM) or σ-count (DSM)
    num_levels: int = 1000


@dataclass
class Gauge:
    run: bool = True
    gauge_a: float = 1.0
    kinetic_a: float = 1.0
#    div_dist: str = 'rademacher'  # gaussian, sphere, unit, rademacher


@dataclass
class Test:
    n_steps: list[int] = field(default_factory=lambda: [
                               2, 5, 10, 25, 50, 100, 250])
    n_samples: int = 256
    save_samples: bool = False
    n_trajectories: int = 16
    save_trajectories: bool = False
    plot: bool = True


@dataclass
class Config:

    dataset: str = 'mnist'

    net: Network = field(default_factory=Network)

    optimizer: Optimizer = field(default_factory=Optimizer)
    data: Data = field(default_factory=Data)
    sample: Sample = field(default_factory=Sample)
    loss: Loss = field(default_factory=Loss)
    gauge: Gauge = field(default_factory=Gauge)

    integrate: Integrate = field(default_factory=Integrate)
    test: Test = field(default_factory=Test)

    # misc
    load_score: str = ''
    name: str = field(
        default_factory=lambda: f"{unique_id(4)}_{epoch_time(2)}")
    x64: bool = False  # whether to use 64 bit precision in jax

    platform: Union[str, None] = None  # gpu or cpu, None will let jax default

    seed: int = 1

    # hydra config configuration
    hydra: Any = field(default_factory=lambda: hydra_config)
    defaults: List[Any] = field(default_factory=lambda: defaults)


##########################
## hydra settings stuff ##
##########################
defaults = [
    # https://hydra.cc/docs/tutorials/structured_config/defaults/
    # "_self_",
    {"override hydra/launcher": "submitit_slurm"},
]


hydra_config = {
    # sets the out dir from config.problem and id
    "run": {"dir": "results/${dataset}/single/${name}"},
    "sweep": {"dir": "results/${dataset}/multi/${name}"},
    "sweeper": {"params": {**get_sweep()}},
    # https://hydra.cc/docs/1.2/plugins/submitit_launcher/
    "launcher": {**get_slurm_config()},
    # "job": {"env_set": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_FLAGS": "--xla_slow_operation_alarm=false", "TF_CPP_MIN_LOG_LEVEL": "3"}},
    "job_logging": {"root": {"level": "WARN"}},
}
cs = ConfigStore.instance()
cs.store(name="default", node=Config)


def get_outpath() -> Path:

    # save files to
    OUTDIR = HydraConfig.get().runtime.output_dir
    OUTDIR_PATH = Path(OUTDIR)

    return OUTDIR_PATH
