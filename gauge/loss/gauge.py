
import jax
import jax.numpy as jnp
from jax import vmap

from gauge.config.config import Config
from gauge.loss import noise
from gauge.utils.tools import hutch_div


def compute_A_B(alpha_bar, alpha_bar_prev):
    c = jnp.sqrt(alpha_bar)
    c_prev = jnp.sqrt(alpha_bar_prev)

    sigma = jnp.sqrt(1.0 - alpha_bar)
    sigma_prev = jnp.sqrt(1.0 - alpha_bar_prev)

    ratio = c_prev / c
    A = ratio - 1.0
    B = ratio * (1.0 - alpha_bar) - sigma_prev * sigma
    return A, B


def get_gauge_loss(cfg: Config, G_net, apply_score, sigmas):

    kinetic_a = cfg.gauge.kinetic_a
    gauge_a = cfg.gauge.gauge_a
    alphas_sched, alpha_bar_sched, alpha_bar_prev_sched, _ = noise.get_cofficients(
        sigmas)

    def loss_fn(params, clean_data, class_l, key):

        # set_up
        batch_size = clean_data.shape[0]
        data_shape = clean_data.shape
        key_sigma, key_noise, div_key = jax.random.split(key, num=3)
        sigma_vals, t_idx = noise.sample_sigmas(key_sigma, sigmas, batch_size)

        alpha_bar_t = alpha_bar_sched[t_idx]      # shape (batch,)
        alpha_bar_prev_t = alpha_bar_prev_sched[t_idx]  # shape (batch,)
        alpha_bar = alpha_bar_t[:, None]      # for broadcast
        alpha_bar_prev = alpha_bar_prev_t[:, None]

        eps = jax.random.normal(
            key_noise, clean_data.shape, dtype=clean_data.dtype)
        alpha_bar = noise.sigma_to_alpha_bar(sigma_vals)
        clean_scale = noise.broadcast_to_match(jnp.sqrt(alpha_bar), clean_data)
        noise_scale = noise.broadcast_to_match(
            jnp.sqrt(1.0 - alpha_bar), clean_data)
        x_t = clean_scale * clean_data + noise_scale * eps

        # get score field
        eps_field = apply_score(
            x_t, sigma_vals, class_l).reshape(batch_size, -1)
        score_field = - (1/noise_scale.reshape(batch_size, -1))*eps_field

        # get gauge field
        G_field = G_net.apply(params, x_t, sigma_vals,
                              class_l).reshape(batch_size, -1)

        x_flat = x_t.reshape(batch_size, -1)

        # assemeble reverse update velocity
        A, B = compute_A_B(alpha_bar, alpha_bar_prev)
        A, B = A[:, None], B[:, None]
        total_field = 0.5*(A*x_flat + B * (score_field+G_field))

        def flat_G(x_flat, sig, cls):
            x = x_flat.reshape(data_shape)
            y = G_net.apply(params, x, sig, cls)
            return y.reshape(batch_size, -1)

        div_G = hutch_div(flat_G)(x_flat, sigma_vals, class_l, key=div_key)
        G_score = vmap(jnp.dot)(G_field, score_field)

        # losses
        loss_gauge = div_G + G_score
        loss_gauge = jnp.mean(loss_gauge**2)

        loss_kin = jnp.mean(jnp.sum(total_field**2, axis=-1))

        # combine
        final_loss = gauge_a*loss_gauge + kinetic_a*loss_kin

        aux = {'gae': loss_gauge, 'kin': loss_kin}

        return final_loss, aux

    return loss_fn
