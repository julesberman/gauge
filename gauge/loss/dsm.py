
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap

from gauge.config.config import Config
from gauge.loss.gauge import get_ortho_loss
from gauge.loss.noise import NoiseSchedule
from gauge.utils.tools import hutch_div


def test_in_rff(values, x_f, n_functions, key, sigma=1.0, weights=None):
    F, B, D = values.shape
    B, one = sigma.shape

    k_w, k_b = jax.random.split(key)

    # We generate a base omega: (M, D), expand it to (B, M, D) to scale by sigma per-batch-item
    base_omega = jax.random.normal(k_w, (n_functions, D))  # (M, D)
    omega = (1/sigma[:, None, :]) * base_omega[None, :, :]

    b = jax.random.uniform(k_b, (n_functions,),
                           minval=0.0, maxval=2.0 * jnp.pi)
    b = jnp.broadcast_to(b, (B, n_functions))
    dot_wx = jnp.einsum("bmd,bd->bm", omega, x_f)
    z = dot_wx + b  # (B, M)

    sin_z = jnp.sin(z)      # (B,M)
    u = jnp.einsum("bmd,fbd->fbm", omega, values)            # (F,B,M)
    proj = -sin_z[None, :, :] * u                           # (F,B,M)

    # omega_sqnorm = jnp.sum(omega**2, axis=-1)                 # (M,)
    # lap_phi = -omega_sqnorm[None, :] * jnp.cos(z)            # (B, M)
    moments = proj  # + lap_phi[None]                    # (F,B,M)

    if weights is not None:
        moments = moments * weights[None, :, None]

    moments = jnp.mean(moments, axis=1)         # (F,M)
    test_residual = jnp.mean(moments ** 2, axis=1)    # (F,)

    return test_residual, moments, sin_z, omega


# def rff_test_loss(err, x_f, key,
#                   num_feats=1024, sigma=1.0, weights=None):
#     """
#     Same as your original, but now uses the shared helper.
#     Returns: (F,)
#     """
#     omega, sinz, moments = rff_params(
#         err, x_f, key, num_feats=num_feats, sigma=sigma)
#     return jnp.mean(moments ** 2, axis=1)


def rff_diversity_loss(err, sin_z, moments, omega, ridge=1e-3, eps=1e-6):
    """
    Batch diversity across F fields:
      1) compute err and RFF moments
      2) project err onto span{ g_m(b) = -sin(z_bm) * omega_m } via least squares
      3) compute mean squared off-diagonal cosine similarity of residuals

    Returns: scalar
    """

    F, B, D = err.shape
    M = omega.shape[0]

    # Gram A_mn = E_b[ <g_m, g_n> ] = E_b[ sinz_bm sinz_bn ] * <omega_m, omega_n>
    S = (sin_z.T @ sin_z) / B                 # (M,M)
    G = omega @ omega.T                     # (M,M)
    A = S * G + ridge * jnp.eye(M, dtype=err.dtype)

    # b_km = E_b[ err · g_m ] is exactly "moments" (F,M)
    c = jnp.linalg.solve(A, moments.T).T    # (F,M)

    # proj_err_kbd = sum_m c_km g_bm = - sum_m c_km sinz_bm omega_m
    proj_err = -jnp.einsum("bm,fm,md->fbd", sin_z, c, omega)  # (F,B,D)
    resid = err - proj_err

    if F <= 1:
        return jnp.array(0.0, dtype=err.dtype)

    # Cosine-sim matrix under batch mean
    gram = jnp.einsum("fbd,gbd->fg", resid, resid) / B       # (F,F)
    nrm = jnp.sqrt(jnp.clip(jnp.diag(gram), a_min=eps))       # (F,)
    corr = gram / (nrm[:, None] * nrm[None, :] + eps)        # (F,F)

    off = corr - jnp.eye(F, dtype=corr.dtype)
    return jnp.mean(off ** 2)


def make_dsm_loss(cfg: Config, schedule: NoiseSchedule, apply_fn):

    ortho_a = cfg.gauge.ortho_a
    freeze_0 = cfg.gauge.freeze_0
    ortho_fn = get_ortho_loss(cfg)
    n_functions = cfg.gauge.n_functions
    strong = False

    def loss_fn(params, x0, class_l, key):
        batch_size = x0.shape[0]
        x_shape = x0.shape
        x0_flat = x0.reshape(batch_size, -1)
        final_loss, aux = 0.0, {}

        key_t, key_noise, key = jax.random.split(key, num=3)
        t = schedule.sample_time(key_t, batch_size)
        weights = schedule.dsm_weight(t)

        # Forward perturbation x0 -> x_t using SDE marginal
        x_t, _, _, _ = schedule.get_xt(key_noise, x0, t)
        x_t_flat = rearrange(x_t, 'B ... -> B (...)')
        target = schedule.dsm_target(
            x_t_flat, x0_flat, t)
        target = rearrange(target, 'B ... -> B (...)')
        _, eps_sigmas = schedule.marginal_alpha_sigma(t)

        # Network prediction: score(x_t, t, class_l)
        preds = apply_fn(
            params,
            x_t_flat.reshape(x_shape),
            t,
            class_l,
        )

        preds = rearrange(preds, 'F B ...-> F B (...)')

        if freeze_0:
            pred_0 = preds[0]
            preds = preds[1:]
            sq_error = jnp.mean((pred_0 - target) ** 2, axis=-1)
            score_loss = jnp.mean(sq_error * weights)
            aux["scr0"] = score_loss
            final_loss += score_loss

        key, test_key = jax.random.split(key)
        err = (preds - target[None])

        # normal score loss
        if strong:
            score_loss = jnp.mean(jnp.mean((err**2), axis=-1) * weights[None])
            aux["strong"] = score_loss
            final_loss += score_loss

        noise_scale = True
        if noise_scale:
            test_residual, moments, sin_z, omega = test_in_rff(
                err, x_t_flat, n_functions, test_key, sigma=eps_sigmas, weights=weights)

            rff_loss = jnp.mean(test_residual)
            aux["rff"] = rff_loss
            final_loss += rff_loss

        # ortho loss
        if ortho_a != 0:
            vt_preds = vmap(schedule.pf_ode_vt,
                            (None, None, 0))(x_t_flat, t, err)

            ortho_loss = ortho_fn(vt_preds)  # batch_size

            aux['ort'] = ortho_loss
            final_loss += ortho_loss*ortho_a

        ########################

        # scales = [0.5, 1, 10, 50]
        # for scale in scales:
        #     test_residual, moments, sin_z, omega = test_in_rff(
        #         err, x_t_flat, n_functions, test_key, sigma=scale, weights=weights)

        #     rff_loss = jnp.mean(test_residual)  # * (1 / scale)
        #     aux[f"rff_{scale}"] = rff_loss
        #     final_loss += rff_loss / len(scales)

        # rff_loss = rff_test_loss(
        #     preds, target, x_t_flat, test_key, weights=weights)
        # rff_loss = jnp.mean(rff_loss)
        # aux['rff'] = rff_loss
        # final_loss += rff_loss

        # # # # DSM target and weighting
        # # score_loss = 0.0
        # # for p in preds:
        # #     sq_error = jnp.mean((p - target) ** 2, axis=-1)
        # #     sq_error = sq_error * weights
        # #     score_loss += (jnp.mean(sq_error) / n_fields)

        # # aux['scr'] = score_loss
        # # final_loss += score_loss

        # # ortho loss
        # if n_fields > 1 and ortho_a != 0:
        #     if freeze_0:
        #         preds.at[0].set(jax.lax.stop_gradient(preds[0]))

        #     vt_preds = vmap(schedule.pf_ode_vt,
        #                     (None, None, 0))(x_t_flat, t, preds)

        #     ortho_loss = ortho_fn(vt_preds, target_corr=corr)  # batch_size
        #     # ortho_loss = ortho_loss * 1/(jnp.squeeze(t)+1e-3)
        #     ortho_loss = jnp.mean(ortho_loss)

        #     aux['ort'] = ortho_loss
        #     final_loss += ortho_loss*ortho_a

        return final_loss, aux

    return loss_fn


def make_dsm_loss_old(cfg: Config, schedule: NoiseSchedule, apply_fn):

    n_fields = cfg.gauge.n_fields
    div_a = cfg.gauge.div_a
    ortho_a = cfg.gauge.ortho_a
    ortho_fn = get_ortho_loss(cfg)

    def loss_fn(params, x0, class_l, key):
        batch_size = x0.shape[0]
        x_shape = x0.shape
        x0_flat = x0.reshape(batch_size, -1)
        final_loss, aux = 0.0, {}

        key_t, key_noise, key = jax.random.split(key, num=3)
        t = schedule.sample_time(key_t, batch_size)
        weights = schedule.dsm_weight(t)

        # Forward perturbation x0 -> x_t using SDE marginal
        x_t_flat, _, _, _ = schedule.get_xt(key_noise, x0_flat, t)
        true_score = schedule.dsm_target(
            x_t_flat, x0_flat, t).reshape(batch_size, -1)

        # Network prediction: score(x_t, t, class_l)
        pred_score = apply_fn(
            params,
            x_t_flat.reshape(x_shape),
            t,
            0,
        )
        pred_score = rearrange(pred_score, 'B ... C -> B (... C)')

        # first is score
        sq_error = jnp.mean((pred_score - true_score) ** 2, axis=-1)
        score_loss = (jnp.mean(sq_error * weights))
        aux['scr'] = score_loss
        final_loss += score_loss

        if div_a != 0:
            loss_gauge = 0.0
            f_xs = []
            for f_i in range(1, n_fields):
                div_key, key = jax.random.split(key)

                def flat_G(x_flat, t, cls, f_i=f_i):
                    x = x_flat.reshape(x_shape)
                    y = apply_fn(params, x, t, f_i)
                    return y.reshape(batch_size, -1)

                div_G, f_x = hutch_div(flat_G)(
                    x_t_flat, t, class_l, key=div_key, return_fwd=True)
                G_score = vmap(jnp.dot)(f_x, true_score)
                f_xs.append(f_x)
                loss_gauge += jnp.mean(weights*(div_G + G_score)**2) / n_fields

            f_xs = jnp.asarray(f_xs)
            aux['div'] = loss_gauge
            final_loss += loss_gauge*div_a

            # ortho loss
            if n_fields > 1 and ortho_a != 0:
                ortho_loss = ortho_fn(f_xs, target_corr=0)  # batch_size
                ortho_loss = jnp.mean(ortho_loss)
                aux['ort'] = ortho_loss
                final_loss += ortho_loss*ortho_a

        return final_loss, aux

    return loss_fn
