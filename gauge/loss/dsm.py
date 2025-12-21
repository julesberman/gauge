
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap

from gauge.config.config import Config
from gauge.loss.gauge import (
    get_var_loss,
    projected_rho_div,
    test_in_rff,
)
from gauge.loss.noise import NoiseSchedule


def make_dsm_loss(cfg: Config, schedule: NoiseSchedule, apply_fn):

    var_a = cfg.gauge.var_a
    reg_a = cfg.gauge.reg_a
    fixed_t = cfg.score.fixed_t

    var_features = cfg.gauge.var_features
    var_loss_fn = get_var_loss(cfg)

    n_functions = cfg.gauge.n_functions
    strong = False

    def loss_fn(params, x0, class_l, key):
        batch_size = x0.shape[0]
        x_shape = x0.shape
        x0_flat = x0.reshape(batch_size, -1)
        final_loss, aux = 0.0, {}

        key_t, key_noise, key = jax.random.split(key, num=3)
        t = schedule.sample_time(key_t, batch_size, single=fixed_t)
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

        key, test_key = jax.random.split(key)
        err = (preds - target[None])

        # normal score loss
        if strong:
            score_loss = jnp.mean(jnp.mean((err**2), axis=-1) * weights[None])
            aux["strong"] = score_loss
            final_loss += score_loss

        test_residual, moments, sin_z, cos_z, omega = test_in_rff(
            err, x_t_flat, n_functions, test_key, eps_sigmas, weights=weights)

        rff_loss = jnp.mean(test_residual)
        aux["rff"] = rff_loss
        final_loss += rff_loss

        # ortho loss
        if var_a != 0:
            key, test_key = jax.random.split(key)
            if var_features == 'weak':
                _, features, _ = test_in_rff(
                    err, x_t_flat, n_functions, test_key, sigma=eps_sigmas, weights=weights, divfree=True)
            elif var_features == 'w_err':
                weighted_err = err * weights[None, :, None]
                features = rearrange(weighted_err, 'F ... -> F (...)')
            elif var_features == 'v_err':
                vt_target = vmap(schedule.pf_ode_vt)(x_t_flat, t, target)
                vt_preds = vmap(schedule.pf_ode_vt,
                                (None, None, 0))(x_t_flat, t, preds)
                vt_err = (vt_preds - vt_target[None])
                features = rearrange(vt_err, 'F ... -> F (...)')

            elif var_features == 'proj_err':
                features = projected_rho_div(
                    err, n_functions, sin_z, cos_z, omega, test_key, weights=weights)

            loss_val = var_loss_fn(features)
            aux['ort'] = loss_val
            final_loss += loss_val*var_a

        if reg_a != 0:
            pass
            # key, reg_key = jax.random.split(key)
            # features = projected_rho_div(
            #     err, n_functions, sin_z, cos_z, omega, reg_key, weights=weights)
            # kin_loss = jnp.mean((features-0.1)**2)
            # aux['kin'] = kin_loss
            # final_loss += kin_loss*reg_a

            # def flat_fn(x_flat, t, target, class_l):
            #     x = x_flat.reshape(1, *x_shape[1:])
            #     y = apply_fn(params, x, t, class_l)
            #     err = (jnp.squeeze(y) - jnp.squeeze(target))
            #     return err.reshape(-1)

            # jr = vmap(jacrand(reg_key, flat_fn), (0, 0, 0, None))(
            #     x_t_flat, t, target, class_l)
            # print(jr.shape)
            # reg_loss = jnp.mean(jr ** 2)  # (F,)
            # # reg_loss = jnp.mean(jnp.linalg.norm(jr, axis=-1))
            # aux['reg'] = reg_loss
            # final_loss += reg_loss*reg_a

        return final_loss, aux

    return loss_fn


# def make_dsm_loss_old(cfg: Config, schedule: NoiseSchedule, apply_fn):

#     n_fields = cfg.gauge.n_fields
#     div_a = cfg.gauge.div_a
#     ortho_a = cfg.gauge.ortho_a
#     ortho_fn = get_ortho_loss(cfg)

#     def loss_fn(params, x0, class_l, key):
#         batch_size = x0.shape[0]
#         x_shape = x0.shape
#         x0_flat = x0.reshape(batch_size, -1)
#         final_loss, aux = 0.0, {}

#         key_t, key_noise, key = jax.random.split(key, num=3)
#         t = schedule.sample_time(key_t, batch_size)
#         weights = schedule.dsm_weight(t)

#         # Forward perturbation x0 -> x_t using SDE marginal
#         x_t_flat, _, _, _ = schedule.get_xt(key_noise, x0_flat, t)
#         true_score = schedule.dsm_target(
#             x_t_flat, x0_flat, t).reshape(batch_size, -1)

#         # Network prediction: score(x_t, t, class_l)
#         pred_score = apply_fn(
#             params,
#             x_t_flat.reshape(x_shape),
#             t,
#             0,
#         )
#         pred_score = rearrange(pred_score, 'B ... C -> B (... C)')

#         # first is score
#         sq_error = jnp.mean((pred_score - true_score) ** 2, axis=-1)
#         score_loss = (jnp.mean(sq_error * weights))
#         aux['scr'] = score_loss
#         final_loss += score_loss

#         if div_a != 0:
#             loss_gauge = 0.0
#             f_xs = []
#             for f_i in range(1, n_fields):
#                 div_key, key = jax.random.split(key)

#                 def flat_G(x_flat, t, cls, f_i=f_i):
#                     x = x_flat.reshape(x_shape)
#                     y = apply_fn(params, x, t, f_i)
#                     return y.reshape(batch_size, -1)

#                 div_G, f_x = hutch_div(flat_G)(
#                     x_t_flat, t, class_l, key=div_key, return_fwd=True)
#                 G_score = vmap(jnp.dot)(f_x, true_score)
#                 f_xs.append(f_x)
#                 loss_gauge += jnp.mean(weights*(div_G + G_score)**2) / n_fields

#             f_xs = jnp.asarray(f_xs)
#             aux['div'] = loss_gauge
#             final_loss += loss_gauge*div_a

#             # ortho loss
#             if n_fields > 1 and ortho_a != 0:
#                 ortho_loss = ortho_fn(f_xs, target_corr=0)  # batch_size
#                 ortho_loss = jnp.mean(ortho_loss)
#                 aux['ort'] = ortho_loss
#                 final_loss += ortho_loss*ortho_a

#         return final_loss, aux

#     return loss_fn
