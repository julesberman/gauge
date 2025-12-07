
"""Entry points for score-model losses."""

from gauge.config.config import Config
from gauge.loss import noise
from gauge.loss.ddpm import make_ddpm_loss
from gauge.loss.dsm import make_dsm_loss


def get_score_loss(cfg: Config, net):
    """Factory returning the configured score-model loss."""
    loss_cfg = cfg.loss
    apply_fn = net.apply
    method = loss_cfg.method.lower()

    sigmas = noise.make_sigma_schedule(
        loss_cfg.sigma_min,
        loss_cfg.sigma_max,
        loss_cfg.num_levels,
        loss_cfg.schedule,
    )

    if method == "ddim" or method == 'ddpm':
        return make_ddpm_loss(loss_cfg, sigmas, apply_fn), sigmas
    if method == "dsm":
        return make_dsm_loss(loss_cfg, sigmas, apply_fn), sigmas
    raise ValueError(f"Unsupported loss method '{loss_cfg.method}'.")
