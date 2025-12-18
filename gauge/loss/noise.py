"""Noise schedule utilities for score-model losses."""


import jax.numpy as jnp
from jax import random

from gauge.config.config import Config


def get_noise_schedule(cfg: Config):

    Noise = NoiseSchedule(
        kind=cfg.score.kind, noise_min=cfg.score.noise_min, noise_max=cfg.score.noise_max)

    return Noise


class NoiseSchedule:
    """
    Simple VP / VE noise schedule.

    kind: "vp" uses beta in [noise_min, noise_max]
    kind: "ve" uses sigma in [noise_min, noise_max]
    """

    def __init__(self, kind: str, noise_min: float, noise_max: float):
        assert kind in ("vp", "ve")
        self.kind = kind

        auto_noise = {"vp_min": 0.1, "vp_max": 20.0,
                      "ve_min": 0.01, "ve_max": 50.0}
        if noise_min == -1:
            noise_min = auto_noise[f'{kind}_min']
        if noise_max == -1:
            noise_max = auto_noise[f'{kind}_max']

        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)

    # ---------- basic schedules ----------

    def beta(self, t):
        """VP beta(t) (linear). Only valid for kind='vp'."""
        assert self.kind == "vp"
        b0, b1 = self.noise_min, self.noise_max
        return b0 + t * (b1 - b0)

    def marginal_alpha_sigma(self, t):
        """
        Returns (alpha(t), sigma(t)) for both VP and VE.
        t: scalar or [B]
        """
        t = jnp.asarray(t)

        if self.kind == "vp":
            b0, b1 = self.noise_min, self.noise_max
            int_beta = b0 * t + 0.5 * (b1 - b0) * t**2         # ∫_0^t β(s) ds
            alpha = jnp.exp(-0.5 * int_beta)
            sigma = jnp.sqrt(jnp.maximum(1.0 - alpha**2, 1e-12))
            return alpha, sigma

        # VE: geometric sigma(t) in [sigma_min, sigma_max]
        s0, s1 = self.noise_min, self.noise_max
        log_ratio = jnp.log(s1 / s0)
        sigma = s0 * jnp.exp(log_ratio * t)
        alpha = jnp.ones_like(sigma)
        return alpha, sigma

    def diffusion(self, t):
        """
        g(t) in d x_t = f(x,t) dt + g(t) dW_t.
        """
        t = jnp.asarray(t)

        if self.kind == "vp":
            return jnp.sqrt(self.beta(t))

        # VE: g^2(t) = d/dt sigma^2(t)
        s0, s1 = self.noise_min, self.noise_max
        log_ratio = jnp.log(s1 / s0)
        _, sigma = self.marginal_alpha_sigma(t)
        return sigma * jnp.sqrt(2.0 * log_ratio)

    # ---------- utilities for training ----------

    def sample_time(self, rng, batch_size: int):
        """t ~ Uniform[0,1]."""
        return random.uniform(rng, (batch_size, 1), minval=1e-3, maxval=1.0)

    def get_xt(self, rng, x0, t):
        """
        x_t = alpha(t) x0 + sigma(t) eps, eps ~ N(0,I).
        x0: [B, ...]; t: [B] or scalar.
        """
        alpha, sigma = self.marginal_alpha_sigma(t)
        while alpha.ndim < x0.ndim:
            alpha = alpha[..., None]
            sigma = sigma[..., None]
        eps = random.normal(rng, x0.shape)
        x_t = alpha * x0 + sigma * eps
        return x_t, eps, alpha, sigma

    def dsm_target(self, x_t, x0, t):
        """
        ∇_x log q(x_t | x0) for both VP and VE:
        = -(x_t - alpha(t) x0) / sigma(t)^2
        """
        alpha, sigma = self.marginal_alpha_sigma(t)
        while alpha.ndim < x_t.ndim:
            alpha = alpha[..., None]
            sigma = sigma[..., None]

        sig = jnp.maximum(sigma**2, 1e-12)

        return -(x_t - alpha * x0) / sig

    def dsm_weight(self, t):
        """Simple λ(t) = σ(t)^2 weighting."""
        _, sigma = self.marginal_alpha_sigma(t)
        return jnp.squeeze(sigma**2)

    # ---------- utilities for sampling ----------

    def prior_sample(self, rng, shape):
        """
        Sample from the terminal prior p_1(x).

        VP:   N(0, I)
        VE:   N(0, sigma_max^2 I)
        """
        z = random.normal(rng, shape)
        if self.kind == "vp":
            return z
        return self.noise_max * z

    def reverse_sde_step(self, rng, x, t, dt, score_fn):
        """
        One Euler–Maruyama step of the reverse-time SDE:

          dx = [f(x,t) - g(t)^2 s_theta(x,t)] dt + g(t) dW

        dt should be negative when integrating 1 -> 0.
        score_fn(x, t) -> score with same shape as x.
        """
        g = self.diffusion(t)

        if self.kind == "vp":
            b = self.beta(t)
            f = -0.5 * b * x
        else:  # VE
            f = 0.0

        s = score_fn(x, t)
        drift = f - (g**2) * s

        z = random.normal(rng, x.shape)
        return x + drift * dt + g * jnp.sqrt(jnp.abs(dt)) * z

    def pf_ode_vt(self, x, t, score_pred):
        """
        One Euler step of the reverse probability-flow ODE:

          dx/dt = f(x,t) - 0.5 g(t)^2 s_theta(x,t)

        dt should be negative when integrating 1 -> 0.
        """
        g = self.diffusion(t)

        if self.kind == "vp":
            b = self.beta(t)
            f = -0.5 * b * x
        else:  # VE
            f = 0.0

        drift = f - 0.5 * (g**2) * score_pred
        return drift
