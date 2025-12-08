
import jax
import jax.numpy as jnp
from jax import jit, vmap

from gauge.config.config import Config
from gauge.loss import noise
from gauge.loss.noise import sigma_to_alpha_bar
from gauge.utils.tools import hutch_div


def get_combine_V(cfg: Config, Score_net, score_params, G_net, G_params):

    if cfg.loss.method == 'ddim':
        @jit
        def apply_V(x, time_inp, labels):
            alpha_bar = sigma_to_alpha_bar(jnp.squeeze(time_inp))
            beta = jnp.sqrt(1-alpha_bar)
            beta = beta[:, None, None, None]
            return Score_net.apply(score_params, x, time_inp, labels) - beta * G_net.apply(G_params, x, time_inp, labels)

    if cfg.loss.method == 'dsm':
        @jit
        def apply_V(x, time_inp, labels):
            return Score_net.apply(score_params, x, time_inp, labels) + G_net.apply(G_params, x, time_inp, labels)

    return apply_V


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

    if cfg.loss.method == 'ddim':
        loss_fn = get_gauge_ddim_loss(cfg, G_net, apply_score, sigmas)

    if cfg.loss.method == 'dsm':
        loss_fn = get_gauge_dsm_loss(cfg, G_net, apply_score)

    return loss_fn


def get_gauge_ddim_loss(cfg: Config, apply_G, apply_score, sigmas):

    kinetic_a = cfg.gauge.kinetic_a
    gauge_a = cfg.gauge.gauge_a
    alphas_sched, alpha_bar_sched, alpha_bar_prev_sched, _ = noise.get_cofficients(
        sigmas)

    def loss_fn(params, clean_data, class_l, key):

        # set_up
        batch_size = clean_data.shape[0]
        data_shape = clean_data.shape
        key_sigma, key_noise, div_key = jax.random.split(key, num=3)
        sigma_vals, t_idx, time = noise.sample_sigmas(
            key_sigma, sigmas, batch_size)

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
            x_t, time, class_l).reshape(batch_size, -1)
        score_field = - (1/noise_scale.reshape(batch_size, -1))*eps_field

        # get gauge field
        G_field = apply_G(params, x_t, time,
                          class_l).reshape(batch_size, -1)
        x_flat = x_t.reshape(batch_size, -1)

        # assemeble reverse update velocity
        A, B = compute_A_B(alpha_bar, alpha_bar_prev)
        A, B = A[:, None], B[:, None]
        total_field = 0.5*(A*x_flat + B * (score_field+G_field))

        def flat_G(x_flat, sig, cls):
            x = x_flat.reshape(data_shape)
            y = apply_G(params, x, sig, cls)
            return y.reshape(batch_size, -1)

        div_G = hutch_div(flat_G)(x_flat, time, class_l, key=div_key)
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


def align_vecs(v1, v2):
    v_norm = jnp.linalg.norm(v1, axis=-1)
    d_norm = jnp.linalg.norm(v2, axis=-1)
    cos = jnp.sum(v1 * v2, axis=-1) / (v_norm * d_norm + 1e-6)
    loss_cos = jnp.mean((1.0 - cos) ** 2)
    return loss_cos


def get_gauge_dsm_loss(cfg: Config, apply_G, apply_score):

    kinetic_a = cfg.gauge.kinetic_a
    gauge_a = cfg.gauge.gauge_a
    end_a = cfg.gauge.end_a

    def beta_t(t, beta_min=0.1, beta_max=20.0):
        return beta_min + t * (beta_max - beta_min)

    def loss_fn(params, x0, class_l, key):

        # set_up
        batch_size = x0.shape[0]
        data_shape = x0.shape

        key_sigma, key_noise, div_key, skey = jax.random.split(key, num=4)
        sigma_vals, t_vals = noise.sample_vp_sde_sigmas(key_sigma, batch_size)
        sigma_bc = noise.broadcast_to_match(sigma_vals, x0)

        eps = jax.random.normal(key_noise, x0.shape)
        alpha_vals = jnp.sqrt(1.0 - jnp.square(sigma_vals))
        alpha_bc = noise.broadcast_to_match(alpha_vals, x0)

        x_t = alpha_bc * x0 + sigma_bc * eps
        x_t_flat = x_t.reshape(batch_size, -1)

        # get score field
        score_field = apply_score(
            x_t, t_vals, class_l).reshape(batch_size, -1)

        # get gauge field
        G_field = apply_G(params, x_t, t_vals,
                          class_l).reshape(batch_size, -1)

        beta = beta_t(t_vals)
        total_field = -0.5 * beta * x_t_flat - \
            0.5 * beta * (score_field+G_field)

        def flat_G(x_flat, sig, cls):
            x = x_flat.reshape(data_shape)
            y = apply_G(params, x, sig, cls)
            return y.reshape(batch_size, -1)

        div_G = hutch_div(flat_G)(x_t_flat, t_vals, class_l, key=div_key)
        G_score = vmap(jnp.dot)(G_field, score_field)

        # losses
        loss_gauge = div_G + G_score
        loss_gauge = jnp.mean(loss_gauge**2)

        loss_kin = jnp.mean(jnp.sum(total_field**2, axis=-1))

        # align to end
        dt = jax.random.uniform(skey, minval=-0.48, maxval=0.48)
        t_vals = t_vals + dt
        t_vals = jnp.clip(t_vals, 1e-3, 1.0)

        sigma_vals = noise.vp_t_to_sigma(t_vals)
        sigma_bc = noise.broadcast_to_match(sigma_vals, x0)

        alpha_vals = jnp.sqrt(jnp.clip(1.0 - jnp.square(sigma_vals), 0.0, 1.0))
        alpha_bc = noise.broadcast_to_match(alpha_vals, x0)

        x_t = alpha_bc * x0 + sigma_bc * eps
        x_t_flat = x_t.reshape(batch_size, -1)
        score_field = apply_score(
            x_t, t_vals, class_l).reshape(batch_size, -1)
        G_field = apply_G(params, x_t, t_vals,
                          class_l).reshape(batch_size, -1)
        beta = beta_t(t_vals)
        total_field2 = -0.5 * beta * x_t_flat - \
            0.5 * beta * (score_field+G_field)
        loss_end = align_vecs(total_field, total_field2)

        # combine
        final_loss = gauge_a*loss_gauge + kinetic_a*loss_kin + end_a*loss_end

        # aux = {'gae': loss_gauge, 'kin': loss_kin}

        aux = {
            'gae': loss_gauge,
            'kin': loss_kin,
            'end': loss_end,
            'div': jnp.mean(div_G**2),
            'gscore': jnp.mean(G_score**2),
            # 'sigma': jnp.mean(sigma_vals),
            # 'w_gae':  jnp.mean(weights*loss_gauge**2),
            # 'w_kin':  jnp.mean(weights*jnp.sum(total_field**2, axis=-1))
        }

        return final_loss, aux

    return loss_fn
