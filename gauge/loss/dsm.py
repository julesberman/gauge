
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap

from gauge.config.config import Config
from gauge.loss.gauge import (
    get_var_loss,
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
    weighted = cfg.gauge.weighted
    omegas = cfg.gauge.omegas
    reg_fn = cfg.gauge.reg_fn
    resample = cfg.gauge.resample

    def loss_fn(params, x0, class_l, key):
        batch_size = x0.shape[0]
        x_shape = x0.shape
        final_loss, aux = 0.0, {}

        key_t, key_noise, key = jax.random.split(key, num=3)
        time = schedule.sample_time(key_t, batch_size, single=fixed_t)
        t_snr = schedule.get_logsnr_time(time)
        # Forward perturbation x0 -> x_t using SDE marginal
        x_t, target, alpha, sigmas, eps = schedule.get_parameters(
            key_noise, x0, time)
        x_t_flat = rearrange(x_t, 'B ... -> B (...)')
        target = rearrange(target, 'B ... -> B (...)')

        # Network prediction: score(x_t, t, class_l)
        preds = apply_fn(
            params,
            x_t_flat.reshape(x_shape),
            t_snr,
            class_l,
        )

        preds = rearrange(preds, 'F B ...-> F B (...)')

        key, test_key = jax.random.split(key)
        err = (preds - target[None])

        if weighted == 'weights':
            weights = schedule.dsm_weight(time[0, 0])
        if weighted == 'one':
            weights = jnp.array(1.0)
        if weighted == 'sig':
            weights = sigmas[0]
        if weighted == 'sig2':
            weights = sigmas[0]**2

        if omegas == 'one':
            omega = jnp.array(1.0)
        if omegas == 'sig':
            omega = sigmas[0]
        if omegas == 'adj':
            omega = jnp.sqrt(sigmas[0]**2 + alpha[0]**2)

        # normal score loss
        if n_functions == 0:
            score_loss = jnp.mean((err**2), axis=-1)
            if weights is not None:
                score_loss = jnp.mean(score_loss * weights)
            else:
                score_loss = jnp.mean(score_loss)
            aux["strong"] = score_loss
            final_loss += score_loss

        else:

            if resample == 'fixed':
                test_key = jax.random.PRNGKey(1)

            test_residual, moments, sin_z, cos_z, omega_sig = test_in_rff(
                err, x_t_flat, n_functions, test_key, omega)

            rff_loss = jnp.mean(test_residual * weights)
            aux["rff"] = rff_loss
            final_loss += rff_loss

        # aux['t'] = time[0][0]
        # aux['sig'] = sigmas[0]
        # aux['w'] = weights
        # aux['omega'] = omega
        # aux["x_t"] = jnp.mean(jnp.linalg.norm(x_t_flat, axis=-1))
        # aux["err"] = jnp.mean(jnp.linalg.norm(err, axis=-1))
        # aux["target"] = jnp.mean(jnp.linalg.norm(target, axis=-1))

        # ortho loss
        if var_a != 0:
            key, test_key = jax.random.split(key)
            if resample == 'fixed':
                test_key = jax.random.PRNGKey(1)
            if var_features == 'err':
                features = rearrange(err, 'F ... -> F (...)')
            elif var_features == 'v_err':
                vt_target = vmap(schedule.pf_ode_vt)(x_t_flat, time, target)
                vt_preds = vmap(schedule.pf_ode_vt,
                                (None, None, 0))(x_t_flat, time, preds)
                vt_err = (vt_preds - vt_target[None])
                features = rearrange(vt_err, 'F ... -> F (...)')
            elif var_features == 'vt':
                vt_preds = vmap(schedule.pf_ode_vt,
                                (None, None, 0))(x_t_flat, time, preds)
                features = rearrange(vt_preds, 'F ... -> F (...)')
            elif var_features == 'm_vt':
                vt_preds = vmap(schedule.pf_ode_vt,
                                (None, None, 0))(x_t_flat, time, preds)
                lm_omega = 10
                _, features, _, _, _ = test_in_rff(
                    vt_preds, x_t_flat, n_functions, test_key, lm_omega)

            loss_val = var_loss_fn(features)
            aux['ort'] = loss_val
            final_loss += loss_val*var_a

        if reg_a != 0:
            # -----------------------
            # helpers (Hutchinson JVP)
            # -----------------------
            def _as_batched(x):
                # ensure leading batch dim of size 1
                return x[None] if x.ndim == 0 else x[None, ...]

            def score_mean_single(x_flat, t_snr_i, class_i):
                # x_flat: (D,)
                x = x_flat.reshape(1, *x_shape[1:])           # (1, ...)
                t_in = _as_batched(t_snr_i)                   # (1, ...)
                # (1, ...)  (assumes class_l is per-example)
                c_in = _as_batched(class_i)
                y = apply_fn(params, x, t_in, c_in)           # (F, 1, ...)
                y = rearrange(y, 'F B ... -> F B (...)')      # (F, 1, D)
                return jnp.mean(y, axis=0)[0]                 # (D,)

            def vt_mean_single(x_flat, t_snr_i, time_i, class_i):
                # PF-ODE velocity from the *mean* predicted score (vt is affine in score)
                s_bar = score_mean_single(x_flat, t_snr_i, class_i)      # (D,)
                # (D,)
                return schedule.pf_ode_vt(x_flat, time_i, s_bar)

            def hutch_jac_frob(vec_fn, x_flat, key_i):
                # Estimates ||J||_F^2 using E_r ||J r||^2 with r Rademacher
                r = jax.random.rademacher(
                    key_i, x_flat.shape).astype(x_flat.dtype)
                _, jvp_out = jax.jvp(vec_fn, (x_flat,), (r,))
                return jnp.sum(jvp_out ** 2)

            # -----------------------------------------
            # existing: kin (minimum-norm in score space)
            # -----------------------------------------
            if reg_fn == 'kin':
                kin_loss = jnp.mean(err ** 2)
                aux['kin'] = kin_loss
                final_loss += reg_a * kin_loss

            # -----------------------------------------
            # existing (fixed): jac on mean score field
            # -----------------------------------------
            if reg_fn == 'jac':
                key, reg_key = jax.random.split(key)
                reg_keys = jax.random.split(reg_key, batch_size)

                def per_example(x_flat, t_snr_i, class_i, k_i):
                    return hutch_jac_frob(
                        lambda x_: score_mean_single(x_, t_snr_i, class_i),
                        x_flat,
                        k_i
                    )

                jac_loss = jnp.mean(vmap(per_example)(
                    x_t_flat, t_snr, class_l, reg_keys))
                aux['jac'] = jac_loss
                final_loss += reg_a * jac_loss

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
#         true_score = schedule.get_target(
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
