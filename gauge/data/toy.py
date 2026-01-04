import math

import jax.numpy as jnp
import numpy as np

from gauge.utils.tools import normalize


def _sample_swiss_roll(n, rng=None):
    rng = np.random.default_rng(rng)
    t = rng.uniform(1.5 * math.pi, 4.5 * math.pi, size=n)
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1)
    return data * 0.1


def _sample_checkerboard(n, rng=None):
    rng = np.random.default_rng(rng)
    i = rng.integers(0, 4, size=n)
    j = rng.integers(0, 4, size=n)
    i = np.where(((i + j) & 1) == 0, i, (i + 1) & 3)  # keep every other cell
    u = rng.random((n, 2))
    x = -1.0 + (i + u[:, 0]) * 0.5
    y = -1.0 + (j + u[:, 1]) * 0.5
    return np.stack([x, y], axis=1)


def _sample_multimodal_gaussian(n, rng, num_modes=6, radius=2.0, sigma=0.5):
    # 8 Gaussians on a circle
    angles = np.linspace(0, 2 * math.pi, num_modes, endpoint=False)
    means = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    comps = rng.integers(0, num_modes, size=n)
    mean = means[comps]
    noise = rng.normal(0, sigma, size=(n, 2))
    data = mean + noise
    return data


def get_2d_dataset(name: str, n_samples: int = 50_000, seed: int | None = None):
    rng = np.random.default_rng(seed)
    name = name.lower()

    if "swiss" in name:
        data = _sample_swiss_roll(n_samples, rng)
    elif "checker" in name:
        data = _sample_checkerboard(n_samples, rng)
    elif "gmm" in name:
        data = _sample_multimodal_gaussian(n_samples, rng)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # data = _normalize_to_unit_square(data)
    data, _ = normalize(data, method='-11')
    data = jnp.asarray(data)

    return data, (2,)
