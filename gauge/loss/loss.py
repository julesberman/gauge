
"""Entry points for score-model losses."""

from gauge.config.config import Config
from gauge.loss.dsm import make_dsm_loss


def get_score_loss(cfg: Config, noise_schedule, net):
    """Factory returning the configured score-model loss."""
    loss_cfg = cfg.score
    apply_fn = net.apply
    method = loss_cfg.method.lower()

    if method == "dsm":
        return make_dsm_loss(cfg, noise_schedule, apply_fn)
    raise ValueError(f"Unsupported loss method '{loss_cfg.method}'.")
