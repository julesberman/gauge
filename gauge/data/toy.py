import math

import jax.numpy as jnp
import numpy as np

from gauge.utils.tools import normalize


def _normalize_to_unit_square(x: np.ndarray) -> np.ndarray:
    # Affine map each dim to [-1, 1]
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    x = 2.0 * (x - x_min) / (x_max - x_min + 1e-8) - 1.0
    return x.astype("float32")


def _sample_swiss_roll(n, rng):
    t = rng.uniform(1.5 * math.pi, 4.5 * math.pi, size=n)
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1)
    return data


def _sample_checkerboard(n, rng):
    x1 = rng.uniform(-2, 2, size=n)
    x2 = rng.uniform(-2, 2, size=n)
    x2 += (np.floor(x1) % 2)  # offset every other column
    data = np.stack([x1, x2], axis=1)
    return data


def _sample_multimodal_gaussian(n, rng, num_modes=8, radius=2.0, sigma=0.1):
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

# Example:
# ds = make_2d_dataset("swissroll", n_samples=5000, seed=0)
# dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
