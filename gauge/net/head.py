"""
Lightweight, Flax-based U-Net implementation and supporting layers.

All modules follow Flax's Module API and can be composed or reused
independently of the full U-Net.
"""


from dataclasses import field
from typing import Callable, Optional

import jax
from flax import linen as nn

from gauge.net.unet import ConvLayer, ResidualBlock, UNet


class UnetHead(nn.Module):
    """
    Option A: shared UNet trunk + vmapped per-head mini-networks.

    - return_all=True  => [n_heads, B, H, W, out_channels]
    - return_all=False => [B, H, W, out_channels] (computed by selecting head_idx *after* all-head pass)

    For truly efficient "single head only" execution (and JVP per head),
    slice params externally and call VectorFieldHead directly (see note below).
    """
    n_heads: int
    out_channels: int

    # trunk config (matches your UNet)
    emb_features: list = field(default_factory=lambda: [512, 512])
    feature_depths: list = field(default_factory=lambda: [128, 256, 512])
    num_res_blocks: int = 2
    num_middle_res_blocks: int = 1
    activation: Callable = jax.nn.gelu
    norm_groups: int = 8
    n_classes: int = 3

    # head config
    head_width: Optional[int] = None
    head_depth: int = 2
    head_conv_type: str = "conv"
    head_out_kernel_size: tuple = (3, 3)

    def setup(self):
        self.trunk = UNet(
            out_channels=1,
            emb_features=self.emb_features,
            feature_depths=self.feature_depths,
            num_res_blocks=self.num_res_blocks,
            num_middle_res_blocks=self.num_middle_res_blocks,
            activation=self.activation,
            norm_groups=self.norm_groups,
            n_classes=self.n_classes,
            is_trunk=True,
        )

        # Create a vmapped head module with independent params per head (params axis = 0)
        HeadVmap = nn.vmap(
            VectorFieldHead,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(None, None),  # (h, temb) shared across heads
            out_axes=0,            # stack heads on axis 0
            axis_size=self.n_heads,
        )
        self.heads = HeadVmap(
            out_channels=self.out_channels,
            head_width=self.head_width,
            head_depth=self.head_depth,
            conv_type=self.head_conv_type,
            activation=self.activation,
            norm_groups=self.norm_groups,
            out_kernel_size=self.head_out_kernel_size,
        )

    @nn.compact
    def __call__(self, x, time, class_l):
        # shared compute once
        h, temb = self.trunk(x, time, class_l)
        # [n_heads, B, H, W, C_out]
        y_all = self.heads(h, temb)

        return y_all


class VectorFieldHead(nn.Module):
    """A small per-head network: (optional proj) + N residual blocks + output conv."""
    out_channels: int
    head_width: Optional[int] = None   # if None, keep trunk width
    head_depth: int = 3
    conv_type: str = "conv"
    activation: Callable = jax.nn.gelu
    norm_groups: int = 8
    out_kernel_size: tuple = (3, 3)
    proj_kernel_size: tuple = (1, 1)

    @nn.compact
    def __call__(self, h, temb):
        x = h
        in_ch = x.shape[-1]
        width = in_ch if self.head_width is None else self.head_width

        # Optional projection to head width
        if width != in_ch:
            x = ConvLayer(self.conv_type, features=width,
                          kernel_size=self.proj_kernel_size, name="head_proj")(x)

        # A few conditional residual blocks so heads can actually specialize
        for k in range(self.head_depth):
            x = ResidualBlock(
                self.conv_type,
                features=width,
                activation=self.activation,
                norm_groups=self.norm_groups,
                name=f"head_res_{k}",
            )(x, temb)

        # Output vector field
        y = ConvLayer(self.conv_type, features=self.out_channels,
                      kernel_size=self.out_kernel_size, name="head_out")(x)
        return y
