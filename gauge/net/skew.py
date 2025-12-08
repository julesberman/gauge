from dataclasses import field
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax

from gauge.net.unet import FeedFoward, PeriodicTimestep


class SkewNet(nn.Module):
    """Skew-symmetric convolution operator K(t) applied to a score field.

    Given x ≈ s(x,t) (shape (B, H, W, C)) and per-sample time (B, 1),
    this module returns a(x,t) = K(t) x, where:

        K(t) = sum_l w_l(t) K_l

    and each K_l is a skew-symmetric convolution operator. The result is
    more expressive than a single kernel, but remains linear in x and skew.

    Args:
      kernel_size: spatial kernel size (odd, e.g. (3,3)).
      n_basis: number of skew conv bases K_l.
      time_embed_dim: dimension of sinusoidal time embedding.
      mlp_dim: hidden dim of time-conditioning MLP.
      scale_init: initial magnitude for the overall operator (via log_scale).
    """
    kernel_size: Tuple[int, int] = (3, 3)
    n_basis: int = 16
    time_embed_dim: int = 128
    emb_features: list = field(default_factory=lambda: [256, 256])

    @nn.compact
    def __call__(self, x, temb, class_l) -> jnp.ndarray:
        """
        Args:
          x:    (B, H, W, C) score tensor s(x,t)
          time: (B, 1) time values

        Returns:
          a:    (B, H, W, C) = K(t) x
        """
        B, H, W, C = x.shape
        kh, kw = self.kernel_size
        assert kh % 2 == 1 and kw % 2 == 1, "Use odd kernel sizes so center is well-defined."

        dtype = x.dtype

        # ------------------------------------------------------------------
        # 1. Time embedding with sinusoidal + MLP ("best practice" in diffusion)
        # ------------------------------------------------------------------
        if temb is not None:
            time_proj = PeriodicTimestep(
                self.emb_features[0], flip_sin_to_cos=True, freq_shift=0)
            temb = time_proj(jnp.squeeze(temb))
            temb = FeedFoward(features=self.emb_features)(temb)

        # Map time embedding to basis weights w(t) ∈ ℝ^{n_basis}
        w = nn.Dense(self.n_basis, dtype=dtype)(
            temb)              # (B, n_basis)
        # Optional: normalize weights a bit for stability
        # (B, n_basis)
        w = w / jnp.sqrt(self.n_basis).astype(dtype)

        # ------------------------------------------------------------------
        # 2. Learn a set of base raw kernels and skew-symmetrize them
        # ------------------------------------------------------------------
        kernel_shape = (self.n_basis, kh, kw, C, C)
        # Standard variance-scaled initialization
        kernel_init = nn.initializers.variance_scaling(
            scale=1.0, mode="fan_avg", distribution="normal"
        )
        raw_kernels = self.param(
            "raw_kernels",
            kernel_init,
            kernel_shape
        ).astype(dtype)

        # Make each base kernel skew via spatial flip + channel transpose:
        #   Q_rev: reversed in (kh, kw)
        #   Q_rev_T: also transposed in channels -> Q[i',j']^T
        # (L, kh, kw, C, C)
        Q = raw_kernels
        # (L, kh, kw, C, C)
        Q_rev = jnp.flip(Q, axis=(1, 2))
        # (L, kh, kw, C, C)
        Q_rev_T = jnp.swapaxes(Q_rev, -1, -2)
        # (L, kh, kw, C, C)
        base_skew_kernels = 0.5 * (Q - Q_rev_T)

        # ------------------------------------------------------------------
        # 3. Mix base skew kernels with time-dependent weights
        # ------------------------------------------------------------------
        # skew_kernels[b] = sum_l w[b,l] * base_skew_kernels[l]
        # tensordot over l: (B, L) · (L, kh, kw, C, C) -> (B, kh, kw, C, C)
        skew_kernels = jnp.tensordot(w, base_skew_kernels, axes=((1,), (0,)))

        # ------------------------------------------------------------------
        # 5. Per-sample convolution: a_b = K(t_b) x_b
        # ------------------------------------------------------------------

        def single_sample_conv(x_i, k_i):
            # x_i: (H, W, C)
            # k_i: (kh, kw, C, C) -> HWIO format for conv_general_dilated
            x_i = x_i[None, ...]  # (1, H, W, C)
            y_i = lax.conv_general_dilated(
                lhs=x_i,
                rhs=k_i,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            return y_i[0]  # (H, W, C)

        a = jax.vmap(single_sample_conv, in_axes=(0, 0))(
            x, skew_kernels)  # (B, H, W, C)

        return a - x
