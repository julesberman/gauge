
import jax
import jax.numpy as jnp
from einops import rearrange

from gauge.config.config import Config
from gauge.loss.gauge import get_ortho_loss
from gauge.loss.noise import NoiseSchedule


def make_dsm_loss(cfg: Config, schedule: NoiseSchedule, apply_fn):

    n_fields = cfg.gauge.n_fields
    ortho_a = cfg.gauge.ortho_a
    ortho_fn = get_ortho_loss(cfg)

    def loss_fn(params, x0, class_l, key):
        batch_size = x0.shape[0]
        x_shape = x0.shape
        x0_flat = x0.reshape(batch_size, -1)
        final_loss, aux = 0.0, {}

        key_t, key_noise = jax.random.split(key)
        t = schedule.sample_time(key_t, batch_size)          # [B]

        # Forward perturbation x0 -> x_t using SDE marginal
        x_t_flat, _, _, _ = schedule.get_xt(key_noise, x0_flat, t)
        target = schedule.dsm_target(
            x_t_flat, x0_flat, t).reshape(batch_size, -1)

        # Network prediction: score(x_t, t, class_l)
        preds = apply_fn(
            params,
            x_t_flat.reshape(x_shape),
            t,
            class_l,
        )

        preds = rearrange(preds, 'B ... (C F) -> F B (... C)', F=n_fields)

        # DSM target and weighting
        score_loss = 0.0
        for p in preds:
            sq_error = jnp.mean((p - target) ** 2, axis=-1)
            weights = schedule.dsm_weight(t)
            score_loss += (jnp.mean(sq_error * weights) / n_fields)

        aux['scr'] = score_loss
        final_loss += score_loss

        if n_fields > 1:
            ortho_loss = ortho_fn(preds)
            aux['ort'] = ortho_loss
            final_loss += ortho_loss*ortho_a

        return score_loss, aux

    return loss_fn
