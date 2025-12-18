import jax.numpy as jnp
from einops import rearrange
from jax import vmap

from gauge.config.config import Config


def get_ortho_loss(cfg: Config):

    if cfg.gauge.ortho_loss == 'cos':
        return lambda v: multi_cos_loss(v, target_corr=0.0)
    elif cfg.gauge.ortho_loss == 'gauss':
        return multi_gaussian_loss

    return multi_cos_loss


def multi_gaussian_loss(v_t_fields):
    """
    Gaussian Repulsion with Dynamic Median Heuristic.
    """
    v_t_fields = rearrange(v_t_fields, 'F B D -> F (B D)')

    # 1. Get all pairwise squared L2 distances
    dists_sq = pairwise_sq_distances(v_t_fields)

    # 2. Dynamic Heuristic: Set bandwidth (2*sigma^2) to the median squared distance.
    # This ensures roughly half of the pairs fall within the repulsion range.
    bandwidth = jnp.median(dists_sq)

    # Prevent division by zero if all fields have collapsed to identical values
    bandwidth = jnp.maximum(bandwidth, 1e-6)

    # 3. Compute Gaussian Kernel: k(x,y) = exp( -||x-y||^2 / bandwidth )
    # lax.stop_gradient(bandwidth) is optional but usually preferred
    kernels = jnp.exp(-dists_sq / bandwidth)

    return jnp.mean(kernels)


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

def multi_cos_loss(v_t_fields, target_corr=0.0):
    v_t_fields = rearrange(v_t_fields, 'F B D -> F (B D)')
    cos_fields = pairwise_correlations(v_t_fields)

    error = (cos_fields - target_corr)**2
    return jnp.mean(error)


def pairwise_correlations(X: jnp.ndarray) -> jnp.ndarray:
    X = jnp.asarray(X)
    Xc = X - jnp.mean(X, axis=1, keepdims=True)
    norms = jnp.linalg.norm(Xc, axis=1)

    n = X.shape[0]
    i, j = jnp.triu_indices(n, k=1)

    def corr_pair(ii, jj):
        denom = norms[ii] * norms[jj]
        dot = jnp.dot(Xc[ii], Xc[jj])
        return jnp.where(denom > 0, dot / denom, 0.0)

    return vmap(corr_pair)(i, j)
