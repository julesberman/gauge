import jax
import jax.numpy as jnp
from jax import vmap

from gauge.config.config import Config


def get_var_loss(cfg: Config):

    if cfg.gauge.var_loss == 'cos':
        return loss_cos
    elif cfg.gauge.var_loss == 'gauss':
        return loss_gauss
    elif cfg.gauge.var_loss == 'dpp':
        return dpp_logdet_loss
    elif cfg.gauge.var_loss == 'ortho':
        return orthonormal_loss
    else:
        raise ValueError(
            f"Unsupported diversity loss '{cfg.gauge.ortho_loss}'.")


def loss_weak_cos(v_t, target, x_t_flat, n_functions, sigma, weights, key):
    err = v_t - target[None]
    test_residual, moments, omega = test_in_rff(
        err, x_t_flat, n_functions, key, sigma=sigma, weights=weights, divfree=True)

    loss = jnp.mean(pairwise_cosine_sim(moments)**2)
    # loss = dpp_logdet_loss(moments)
    # loss = orthonormal_loss(moments)

    return loss


def loss_cos(features):
    cos_fields = pairwise_cosine_sim(features)
    error = (cos_fields)**2
    return jnp.mean(error)


def orthonormal_loss(moments: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    moments: (F, d)
    Returns scalar loss encouraging Z Z^T ≈ I where Z is row-normalized moments.
    """
    z = moments / (jnp.linalg.norm(moments, axis=-
                   1, keepdims=True) + eps)  # (F,d)
    # (F,F)
    gram = z @ z.T
    F = moments.shape[0]
    return jnp.mean((gram - jnp.eye(F, dtype=moments.dtype)) ** 2)


def dpp_logdet_loss(moments: jnp.ndarray, eps: float = 1e-6, norm_eps: float = 1e-8) -> jnp.ndarray:
    """
    moments: (F, d)
    Returns scalar loss: -log det(Z Z^T + eps I), with Z row-normalized.
    """
    z = moments / (jnp.linalg.norm(moments, axis=-1,
                   keepdims=True) + norm_eps)  # (F,d)
    # (F,F)
    K = z @ z.T
    F = moments.shape[0]
    sign, logdet = jnp.linalg.slogdet(
        K + eps * jnp.eye(F, dtype=moments.dtype))
    # sign should be +1 (PSD + eps I), but guard anyway:
    return -logdet


def loss_gauss(features):
    """
    Gaussian Repulsion with Dynamic Median Heuristic.
    """

    # 1. Get all pairwise squared L2 distances
    dists_sq = pairwise_sq_distances(features)

    base_scale = jnp.median(dists_sq)
    base_scale = jnp.maximum(base_scale, 1e-6)  # prevent div by 0

    scales = jnp.asarray([10.0])

    def compute_kernel(bw):
        return jnp.exp(-dists_sq / bw)

    kernels = vmap(compute_kernel)(scales)

    return jnp.mean(kernels)


# def test_in_rff(values, x_f, n_functions, key, sigma=1.0, weights=None, divfree=False):
#     F, B, D = values.shape
#     B, one = sigma.shape

#     k_w, k_b, k_A = jax.random.split(key, num=3)

#     scale = 1  # jnp.sqrt(2.0 / n_functions)

#     # We generate a base omega: (M, D), expand it to (B, M, D) to scale by sigma per-batch-item
#     base_omega = jax.random.normal(k_w, (n_functions, D))  # (M, D)
#     omega = (1/sigma[:, None, :]) * base_omega[None, :, :]

#     b = jax.random.uniform(k_b, (n_functions,),
#                            minval=0.0, maxval=2.0 * jnp.pi)
#     b = jnp.broadcast_to(b, (B, n_functions))
#     dot_wx = jnp.einsum("bmd,bd->bm", omega, x_f)
#     z = dot_wx + b  # (B, M)
#     sin_z = jnp.sin(z)   # (B,M)
#     cos_z = jnp.cos(z)   # (B,M)

#     if divfree:
#         # === Divergence-free plane wave: g(x,t)=cos(z)*p with p ⟂ omega ===
#         # (M, D) global
#         a = jax.random.normal(k_A, (n_functions, D))
#         a = jnp.broadcast_to(a[None, :, :], omega.shape)  # (B,M,D)

#         omega_norm2 = jnp.sum(omega * omega, axis=-1, keepdims=True) + 1e-8
#         proj = jnp.sum(omega * a, axis=-1, keepdims=True) / omega_norm2
#         p = a - omega * proj
#         p = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)

#         g_cos = scale * cos_z[..., None] * p              # (B,M,D)
#         g_sin = scale * sin_z[..., None] * p              # (B,M,D)

#         omega_v_cos = jnp.einsum("fbd,bmd->fbm", values, g_cos)  # (F,B,M)
#         omega_v_sin = jnp.einsum("fbd,bmd->fbm", values, g_sin)  # (F,B,M)

#         omega_v = jnp.concatenate(
#             [omega_v_cos, omega_v_sin], axis=-1)  # (F,B,2M)

#     else:
#         # === NOT divergence-free (gradient test): g = -sin(z) * omega ===
#         dot = jnp.einsum("fbd,bmd->fbm", values, omega)  # (F,B,M)
#         omega_v_cos = -scale * sin_z[None, :, :] * dot   # moments for ∇cos
#         omega_v_sin = scale * cos_z[None, :, :] * dot   # moments for ∇sin
#         omega_v = jnp.concatenate(
#             [omega_v_cos, omega_v_sin], axis=-1)  # (F,B,2M)

#     if weights is not None:
#         omega_v = omega_v * weights[None, :, None]

#     # Feature per field per test: average over samples b
#     moments = jnp.mean(omega_v, axis=1)

#     test_residual = jnp.mean(moments ** 2, axis=1)  # (F,)

#     return test_residual, moments, omega


def pairwise_sq_distances(X: jnp.ndarray) -> jnp.ndarray:
    """
    X: (n, d)
    Returns squared euclidean distances for unique pairs (i<j).
    """
    n = X.shape[0]
    i, j = jnp.triu_indices(n, k=1)

    def dist_pair(ii, jj):
        diff = X[ii] - X[jj]
        return jnp.sum(diff**2)

    return vmap(dist_pair)(i, j)


# --- Existing Cosine Code ---


def pairwise_cosine_sim(X):
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    eps = 1e-8
    Xn = X / (norms + eps)
    G = Xn @ Xn.T              # (n,n)
    i, j = jnp.triu_indices(X.shape[0], k=1)
    return G[i, j]


def test_in_rff(values, x_t, n_functions, key, sigma=1.0, weights=None, u_stat=False):
    # values are the residuals (v_t(x) - score(x,t))
    # x_t is the noisey data time t
    # weights are B dim array of sigma(t)**2
    F, B, D = values.shape
    B, one = sigma.shape

    k_w, k_b, k_A = jax.random.split(key, num=3)

    # We generate a base omega: (M, D), expand it to (B, M, D) to scale by sigma per-batch-item
    # In general these sigmas should all be the same if we draw each batch at a fixed time
    base_omega = jax.random.normal(k_w, (n_functions, D))  # (M, D)
    omega = (1/sigma[:, None, :]) * base_omega[None, :, :]  # (B, M, D)

    b = jax.random.uniform(k_b, (n_functions,),
                           minval=0.0, maxval=2.0 * jnp.pi)
    b = jnp.broadcast_to(b, (B, n_functions))
    dot_wx = jnp.einsum("bmd,bd->bm", omega, x_t)
    z = dot_wx + b  # (B, M)
    sin_z = jnp.sin(z)   # (B,M)
    cos_z = jnp.cos(z)   # (B,M)

    omega_dot = jnp.einsum("fbd,bmd->fbm", values, omega)  # (F,B,M)
    omega_v_cos = -sin_z[None, :, :] * omega_dot   # moments for ∇cos
    omega_v_sin = cos_z[None, :, :] * omega_dot   # moments for ∇sin
    omega_v = jnp.concatenate(
        [omega_v_cos, omega_v_sin], axis=-1)  # (F,B,2M)

    if weights is not None:
        omega_v = omega_v * weights[None, :, None]

    # Feature per field per test: average over samples b
    if u_stat:
        sum_z = jnp.sum(omega_v, axis=1)           # (F, 2M)
        sum_z2 = jnp.sum(omega_v * omega_v, axis=1)  # (F, 2M)
        denom = B * (B - 1)
        mu2_unbiased = (sum_z * sum_z - sum_z2) / denom  # (F, 2M)
        test_residual = jnp.mean(mu2_unbiased, axis=1)  # (F,)
    else:
        moments = jnp.mean(omega_v, axis=1)

    test_residual = jnp.mean(moments ** 2, axis=1)  # (F,)

    return test_residual, moments, sin_z, cos_z, omega


def projected_rho_div(
    residuals,      # (F,B,D)  r^i(x,t) = v^i - v*
    n_functions,    # M
    sin_z,          # (B,M)
    cos_z,          # (B,M)
    omega,          # (B,M,D)
    key,
    weights=None,
    ridge=1e-4,
    eps=1e-8,
):
    """
    Returns:
      c_proj: (F, 2M)  = Monte-Carlo estimate of E_rho[ r_perp · D ],
              where r_perp is the L2(rho)-orthogonal complement of the gradient span
              spanned by [∇cos(z_m), ∇sin(z_m)] and D is a solenoidal RFF vector test family.
    """

    F, B, D = residuals.shape
    M = n_functions

    # ----- Gradient test vectors: g_cos = ∇cos(z) = -sin(z)*omega, g_sin = ∇sin(z) = cos(z)*omega
    g_cos = -sin_z[..., None] * omega     # (B,M,D)
    g_sin = cos_z[..., None] * omega     # (B,M,D)

    # ----- b_i = E[ G^T r_i ] using omega_dot = r·omega
    omega_dot = jnp.einsum("fbd,bmd->fbm", residuals, omega)         # (F,B,M)

    if weights is not None:
        omega_dot = omega_dot * weights[None, :, None]

    b_cos = jnp.mean((-sin_z[None, :, :] * omega_dot), axis=1)       # (F,M)
    b_sin = jnp.mean((cos_z[None, :, :] * omega_dot), axis=1)       # (F,M)
    b_vec = jnp.concatenate([b_cos, b_sin], axis=-1)                 # (F,2M)

    # ----- A = E[ G^T G ] (2M x 2M), computed blockwise for efficiency
    A_cc = jnp.einsum("bmd,bnd->mn", g_cos, g_cos) / B               # (M,M)
    A_cs = jnp.einsum("bmd,bnd->mn", g_cos, g_sin) / B               # (M,M)
    A_sc = jnp.einsum("bmd,bnd->mn", g_sin, g_cos) / B               # (M,M)
    A_ss = jnp.einsum("bmd,bnd->mn", g_sin, g_sin) / B               # (M,M)
    A = jnp.block([[A_cc, A_cs],
                   [A_sc, A_ss]])                                    # (2M,2M)

    # ----- alpha_i solves (A + ridge I) alpha_i = b_i  (L2(rho) projection onto gradient span)
    A_reg = A + ridge * jnp.eye(2*M, dtype=residuals.dtype)
    alpha = jnp.linalg.solve(A_reg, b_vec.T).T                       # (F,2M)

    # ===== Build solenoidal (unweighted divergence-free) RFF vector tests D using P(omega) =====
    # a(b,m) ⟂ omega(b,m)
    k_u, = jax.random.split(key, 1)
    u = jax.random.normal(k_u, (M, D), dtype=residuals.dtype)        # (M,D)

    omega_norm = jnp.linalg.norm(omega, axis=-1, keepdims=True) + eps
    omega_hat = omega / omega_norm                                   # (B,M,D)
    # (1,M,D) broadcast to B
    u_b = u[None, :, :]

    u_dot = jnp.einsum("bmd,bmd->bm", u_b, omega_hat)                 # (B,M)
    u_proj = u_b - u_dot[..., None] * omega_hat                       # (B,M,D)
    u_proj_norm = jnp.linalg.norm(u_proj, axis=-1, keepdims=True) + eps
    # (B,M,D), omega·a=0
    a = u_proj / u_proj_norm

    d_cos = cos_z[..., None] * a                                      # (B,M,D)
    d_sin = sin_z[..., None] * a                                      # (B,M,D)

    # ----- Raw div-free features: c_raw = E[ r · D ]
    a_dot = jnp.einsum("fbd,bmd->fbm", residuals, a)                  # (F,B,M)
    c_cos = jnp.mean((cos_z[None, :, :] * a_dot), axis=1)             # (F,M)
    c_sin = jnp.mean((sin_z[None, :, :] * a_dot), axis=1)             # (F,M)
    c_raw = jnp.concatenate([c_cos, c_sin], axis=-1)                  # (F,2M)

    # ----- Cross term C = E[ G^T D ] so we can do: E[(r-Gα)·D] = c_raw - α C
    C_cc = jnp.einsum("bmd,bnd->mn", g_cos, d_cos) / B                # (M,M)
    C_cs = jnp.einsum("bmd,bnd->mn", g_cos, d_sin) / B                # (M,M)
    C_sc = jnp.einsum("bmd,bnd->mn", g_sin, d_cos) / B                # (M,M)
    C_ss = jnp.einsum("bmd,bnd->mn", g_sin, d_sin) / B                # (M,M)
    C = jnp.block([[C_cc, C_cs],
                   [C_sc, C_ss]])                                     # (2M,2M)

    # ----- Projected features in the rho-divergence-free complement (within the gradient span)
    c_proj = c_raw - alpha @ C                                        # (F,2M)

    return c_proj
