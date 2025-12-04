
import jax
import jax.numpy as jnp
from jax import vmap

from gauge.config.config import Config
from gauge.loss import noise
from gauge.utils.tools import hutch_div


def get_gauge_loss(cfg: Config, G_net, apply_score, sigmas):

    gauge_a = cfg.gauge.gauge_a
    kinetic_a = cfg.gauge.kinetic_a
    gauge_a = cfg.gauge.gauge_a

    def loss_fn(params, clean_data, class_l, key):

        # set_up
        batch_size = clean_data.shape[0]
        key_sigma, key_noise, div_key = jax.random.split(key, num=3)
        sigma_vals, _ = noise.sample_sigmas(key_sigma, sigmas, batch_size)
        eps = jax.random.normal(
            key_noise, clean_data.shape, dtype=clean_data.dtype)
        alpha_bar = noise.sigma_to_alpha_bar(sigma_vals)
        clean_scale = noise.broadcast_to_match(jnp.sqrt(alpha_bar), clean_data)
        noise_scale = noise.broadcast_to_match(
            jnp.sqrt(1.0 - alpha_bar), clean_data)
        x_t = clean_scale * clean_data + noise_scale * eps

        # eval fields
        score_field = apply_score(
            x_t, sigma_vals, class_l).reshape(batch_size, -1)
        G_field = G_net.apply(params, x_t, sigma_vals,
                              class_l).reshape(batch_size, -1)

        x_shape = x_t.shape

        def flat_G(x_flat, sig, cls):
            x = x_flat.reshape(x_shape)
            y = G_net.apply(params, x, sig, cls)
            return y.reshape(batch_size, -1)

        x_flat = x_t.reshape(batch_size, -1)
        div_G = hutch_div(flat_G)(x_flat, sigma_vals, class_l, key=div_key)

        # losses
        loss_gauge = div_G + vmap(jnp.dot)(G_field, score_field)
        loss_gauge = jnp.mean(loss_gauge**2)
        loss_kin = jnp.mean(jnp.sum(G_field**2, axis=-1))

        # combine
        final_loss = gauge_a*loss_gauge + kinetic_a*loss_kin

        return final_loss

    return loss_fn
