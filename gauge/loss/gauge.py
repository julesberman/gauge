

import jax.numpy as jnp
from einops import rearrange
from jax import vmap

from gauge.config.config import Config


def get_ortho_loss(cfg: Config):

    if cfg.gauge.ortho_loss == 'cos':
        loss_fn = multi_cos_loss

    return loss_fn


def multi_cos_loss(v_t_fields, target_corr=0.0):
    v_t_fields = rearrange(v_t_fields, 'F B D -> B F D')
    cos_fields = vmap(cosine_spread_avg)(v_t_fields)
    error = (cos_fields - target_corr)**2
    return jnp.mean(error)


def cosine_spread_avg(x, eps=1e-6):
    """
    Average pairwise cosine distance (covers 1 & 3).

    x: array of shape (n, d) (vectors need not be normalized)

    Returns:
        1 - average cosine similarity over all i < j.
        Maximizing this spreads vectors out globally.
    """
    # Normalize to unit length
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)  # (n, d)

    # Pairwise cosine similarities
    sim = x @ x.T  # (n, n)

    n = x.shape[0]
    # Upper triangle without diagonal
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    sims = jnp.where(mask, sim, 0.0)

    mean_sim = (2.0 / (n * (n - 1))) * sims.sum()
    return mean_sim
