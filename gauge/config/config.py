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
    lr: float = 1e-4
    iters: int = 20_000
    scheduler: str = 'cos'
    warm_up: bool = False
    optimizer: str = "adamw"


@dataclass
class Data:
    normalize: bool = True
    class_labels: bool = False


@dataclass
class Sample:
    batch_size: int = 256
    shuffle: bool = True
    shuffle_buffer_size: Optional[int] = None
    materialize: bool = True


@dataclass
class Integrate:
    # assume we have ddpm model, sample with ddpm or ddim
    clip: tuple | None = (-1, 1)  # clip data at each step
    var: float = 0.0  # varaince of noise in reverse process


@dataclass
class Score:
    method: str = "dsm"  # "ddpm", "dsm"

    kind: str = "vp"  # vp or ve
    noise_min: float = -1  # -1 auto pick based on kind
    noise_max: float = -1  # -1 auto pick based on kind


@dataclass
class Gauge:
    run: bool = True
    n_fields: int = 1
    ortho_loss: str = 'cos'
    ortho_a: float = 1.0
    freeze_0: bool = False
    rewards: list[str] = field(default_factory=lambda: ['kinetic', 'prev'])


@dataclass
class Test:
    # n_steps: list[int] = field(default_factory=lambda: [
    #                            -1, 2, 5, 10, 25, 50, 100, 250, 500, 1000])
    n_steps: list[int] = field(default_factory=lambda: [
                               10, 50, 100, 500, 1000])
    n_samples: int = 256
    save_samples: bool = False
    n_trajectories: int = 16
    save_trajectories: bool = False
    plot: bool = True
    renormalize: bool = True


@dataclass
class Config:

    dataset: str = 'mnist'

    net: Network = field(default_factory=Network)
    gnet: Network = field(default_factory=lambda: Network(
        size="s", emb_features=[256, 256]))

    optimizer: Optimizer = field(default_factory=Optimizer)
    data: Data = field(default_factory=Data)
    sample: Sample = field(default_factory=Sample)
    score: Score = field(default_factory=Score)
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

toy_cfg = Config(
    dataset="toy_swiss",
    data=Data(normalize=False),
    sample=Sample(materialize=True, batch_size=2048),
    net=Network(arch='mlp'),
    gnet=Network(arch='mlp'),
    optimizer=Optimizer(lr=5e-4, iters=20_000),
    test=Test(n_trajectories=10_000, n_samples=10_000, renormalize=False),
    integrate=Integrate(clip=None)
)
cs.store(name="toy", node=toy_cfg)


def get_outpath() -> Path:

    # save files to
    OUTDIR = HydraConfig.get().runtime.output_dir
    OUTDIR_PATH = Path(OUTDIR)

    return OUTDIR_PATH
