"""Noise schedule utilities for score-model losses."""


import jax.numpy as jnp
from jax import random

from gauge.config.config import Config


def get_noise_schedule(cfg: Config):

    Noise = NoiseSchedule(
        kind=cfg.score.kind, target=cfg.score.target, noise_min=cfg.score.noise_min, noise_max=cfg.score.noise_max, log_t=cfg.score.log_t)

    t_min = Noise.t_min
    s0, s1 = Noise.get_sigma_min_max()

    print(f't_min: {t_min} | sigma range: {s0}, {s1}')

    return Noise


class NoiseSchedule:
    """
    Simple VP / VE noise schedule.

    kind:
      - "vp": beta(t) linear in [noise_min, noise_max]
      - "ve": sigma(t) geometric in [noise_min, noise_max]

    target (what the network predicts):
      - "noise": eps
      - "score": conditional score of N(alpha x0, sigma^2 I), i.e. -(x_t - alpha x0)/sigma^2
      - "v": v-parameterization (VP only): v = alpha * eps - sigma * x0
    """

    def __init__(self, kind: str, target: str, noise_min: float, noise_max: float, log_t: bool):
        assert kind in ("vp", "ve")
        assert target in ("noise", "score", "v")
        if target == "v":
            # v-parameterization is standard/orthonormal for VP schedules (alpha^2 + sigma^2 = 1).
            assert kind == "vp", "v-parameterization is supported only for kind='vp'."

        self.kind = kind
        self.target = target
        self.log_t = log_t

        auto_noise = {
            "vp_min": 0.1, "vp_max": 20.0,
            "ve_min": 0.01, "ve_max": 50.0,
        }
        if noise_min == -1:
            noise_min = auto_noise[f"{kind}_min"]
        if noise_max == -1:
            noise_max = auto_noise[f"{kind}_max"]

        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        self.t_min = 1e-3

    # ----------------- helpers -----------------

    @staticmethod
    def _broadcast_like(a, x):
        """Broadcast a (possibly [B,1]) array to match x.ndim by appending singleton dims."""
        while a.ndim < x.ndim:
            a = a[..., None]
        return a

    # ----------------- basic schedules -----------------

    def beta(self, t):
        """VP beta(t) (linear). Only valid for kind='vp'."""
        assert self.kind == "vp"
        t = jnp.asarray(t)
        b0, b1 = self.noise_min, self.noise_max
        return b0 + t * (b1 - b0)

    def marginal_alpha_sigma(self, t):
        """
        Returns (alpha(t), sigma(t)) for both VP and VE.
        t: scalar or array (e.g. [B] or [B,1])
        """
        t = jnp.asarray(t)

        if self.kind == "vp":
            b0, b1 = self.noise_min, self.noise_max
            int_beta = b0 * t + 0.5 * (b1 - b0) * t**2  # ∫_0^t β(s) ds
            alpha = jnp.exp(-0.5 * int_beta)
            sigma2 = jnp.maximum(1.0 - alpha**2, 1e-12)
            sigma = jnp.sqrt(sigma2)
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

        # VE: g^2(t) = d/dt sigma^2(t) = 2 * sigma^2 * d/dt log sigma
        s0, s1 = self.noise_min, self.noise_max
        log_ratio = jnp.log(s1 / s0)
        _, sigma = self.marginal_alpha_sigma(t)
        return sigma * jnp.sqrt(2.0 * log_ratio)

    # ----------------- utilities for training -----------------

    def sample_time(self, rng, batch_size: int, single: bool = False):
        """t ~ Uniform[t_min, 1]. Returns shape [B,1]."""
        t_min = self.t_min
        if single:
            t = random.uniform(rng, (1, 1), minval=t_min, maxval=1.0)
            t = jnp.repeat(t, batch_size, axis=0)
        else:
            t = random.uniform(rng, (batch_size, 1), minval=t_min, maxval=1.0)
        return t

    def get_logsnr_time(self, t):
        if self.log_t:
            alpha_t, sigma_t = self.marginal_alpha_sigma(t)
            return jnp.log(jnp.maximum(alpha_t**2, 1e-12) / jnp.maximum(sigma_t**2, 1e-12))
        else:
            return t

    def get_parameters(self, rng, x0, t):
        """
        Forward perturbation and training targets.

        Returns:
          x_t: perturbed sample
          target: according to self.target in {"noise","score","v"}
          alpha_t: alpha(t) squeezed
          sigma_t: sigma(t) squeezed
          eps: sampled noise
        """
        alpha_t, sigma_t = self.marginal_alpha_sigma(t)

        alpha = self._broadcast_like(alpha_t, x0)
        sigma = self._broadcast_like(sigma_t, x0)

        eps = random.normal(rng, x0.shape)
        x_t = alpha * x0 + sigma * eps

        if self.target == "noise":
            target = eps

        elif self.target == "score":
            # conditional score of N(alpha x0, sigma^2 I)
            target = -(x_t - alpha * x0) / jnp.maximum(sigma**2, 1e-12)

        elif self.target == "v":
            # VP only (enforced in __init__)
            target = alpha * eps - sigma * x0

        else:
            raise ValueError(f"Unknown target={self.target}")

        return x_t, target, jnp.squeeze(alpha_t), jnp.squeeze(sigma_t), eps

    def dsm_weight(self, t):
        """
        Optional loss reweighting (returns shape like t, squeezed).

        - score: sigma^2 (classic weighted DSM form)
        - noise: VP -> beta / sigma^2 ; VE -> 1
        - v: VP -> beta * alpha^2 / sigma^2
        """
        eps = 1e-8
        t = jnp.asarray(t)
        alpha_t, sigma_t = self.marginal_alpha_sigma(t)

        if self.target == "score":
            return jnp.squeeze(sigma_t**2)

        if self.target == "noise":
            if self.kind == "vp":
                beta_t = self.beta(t)
                return jnp.squeeze(beta_t / jnp.maximum(sigma_t**2, eps))
            # VE: simple eps MSE is the common choice
            return jnp.squeeze(jnp.ones_like(sigma_t))

        if self.target == "v":
            snr = (alpha_t**2) / jnp.maximum(sigma_t**2, eps)
            gamma = 5.0
            w = jnp.minimum(snr, gamma) / (snr + 1.0)
            return jnp.squeeze(w)

        raise ValueError(f"Unknown target={self.target}")

    def get_sigma_min_max(self):
        t_min = self.t_min
        s0 = self.marginal_alpha_sigma(t_min)[1]
        s1 = self.marginal_alpha_sigma(1.0)[1]
        return s0, s1

    # ----------------- utilities for sampling -----------------

    def pf_ode_vt(self, x, t, network_pred):
        """
        Probability flow ODE:
          dx/dt = f(x,t) - 0.5 g(t)^2 * score(x,t)

        network_pred is interpreted according to self.target:
          - "score": network_pred is score
          - "noise": network_pred is eps
          - "v":     network_pred is v (VP only)
        """
        t = jnp.asarray(t)

        # drift f and diffusion g
        g = self.diffusion(t)
        if self.kind == "vp":
            b = self.beta(t)
            f = -0.5 * b * x
        else:  # VE
            f = 0.0

        g = self._broadcast_like(g, x)

        if self.target == "score":
            score = network_pred

        else:
            alpha_t, sigma_t = self.marginal_alpha_sigma(t)
            sigma = self._broadcast_like(sigma_t, x)

            if self.target == "noise":
                eps = network_pred

            elif self.target == "v":
                # VP only (enforced in __init__)
                alpha = self._broadcast_like(alpha_t, x)
                # VP identity: eps = sigma * x_t + alpha * v
                eps = sigma * x + alpha * network_pred

            else:
                raise ValueError(f"Unknown target={self.target}")

            score = -eps / jnp.maximum(sigma, 1e-12)

        return f - 0.5 * (g**2) * score
