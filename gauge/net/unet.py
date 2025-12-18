"""
Lightweight, Flax-based U-Net implementation and supporting layers.

All modules follow Flax's Module API and can be composed or reused
independently of the full U-Net.
"""


import math
from dataclasses import field
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


class PeriodicTimestep(nn.Module):
    r"""
    Wrapper Module for sinusoidal Time step Embeddings as described in https://huggingface.co/papers/2006.11239

    Args:
        dim (`int`, *optional*, defaults to `32`):
            Time step embedding dimension.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sinusoidal function from sine to cosine.
        freq_shift (`float`, *optional*, defaults to `1`):
            Frequency shift applied to the sinusoidal embeddings.
    """

    dim: int = 32
    flip_sin_to_cos: bool = False
    freq_shift: float = 1

    @nn.compact
    def __call__(self, timesteps):
        return get_sinusoidal_embeddings(
            timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
        )


def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Returns the positional encoding (same as Tensor2Tensor).

    Args:
        timesteps (`jnp.ndarray` of shape `(N,)`):
            A 1-D array of N indices, one per batch element. These may be fractional.
        embedding_dim (`int`):
            The number of output channels.
        freq_shift (`float`, *optional*, defaults to `1`):
            Shift applied to the frequency scaling of the embeddings.
        min_timescale (`float`, *optional*, defaults to `1`):
            The smallest time unit used in the sinusoidal calculation (should probably be 0.0).
        max_timescale (`float`, *optional*, defaults to `1.0e4`):
            The largest time unit used in the sinusoidal calculation.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the order of sinusoidal components to cosine first.
        scale (`float`, *optional*, defaults to `1.0`):
            A scaling factor applied to the positional embeddings.

    Returns:
        a Tensor of timing signals [N, num_channels]
    """
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
    assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
    num_timescales = float(embedding_dim // 2)
    log_timescale_increment = math.log(
        max_timescale / min_timescale) / (num_timescales - freq_shift)
    inv_timescales = min_timescale * \
        jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32)
                * -log_timescale_increment)
    emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)

    # scale embeddings
    scaled_time = scale * emb

    if flip_sin_to_cos:
        signal = jnp.concatenate(
            [jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1)
    else:
        signal = jnp.concatenate(
            [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
    signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
    return signal


class FeedFoward(nn.Module):
    """Simple multilayer perceptron used for time / class embeddings."""
    features: list
    activation: Callable = jax.nn.gelu

    @nn.compact
    def __call__(self, x):
        for f in self.features:
            x = nn.DenseGeneral(f)(x)
            x = self.activation(x)
        return x


class SeparableConv(nn.Module):
    """Depth-wise separable convolution (depth-wise + point-wise)."""
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    use_bias: bool = False
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        # Depth-wise
        depthwise = nn.Conv(
            features=in_features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            feature_group_count=in_features,
            use_bias=self.use_bias,
            padding=self.padding,
        )(x)
        # Point-wise
        pointwise = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            use_bias=self.use_bias,
        )(depthwise)
        return pointwise


class ConvLayer(nn.Module):
    """Wrapper that dispatches to a regular, separable, or transpose conv."""
    conv_type: str
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)

    def setup(self):
        if self.conv_type == "conv":
            self.conv = nn.Conv(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )
        elif self.conv_type == "separable":
            self.conv = SeparableConv(
                features=self.features,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )

    def __call__(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsampling followed by a 3×3 conv."""
    features: int
    scale: int
    activation: Callable = jax.nn.swish

    @nn.compact
    def __call__(self, x, residual=None):
        B, H, W, C = x.shape
        out = jax.image.resize(
            x, (B, H * self.scale, W * self.scale, C), method="nearest"
        )
        out = ConvLayer("conv", features=self.features,
                        kernel_size=(3, 3))(out)
        if residual is not None:
            out = jnp.concatenate([out, residual], axis=-1)
        return out


class Downsample(nn.Module):
    """Strided 3×3 conv for spatial downsampling."""
    features: int
    scale: int
    activation: Callable = jax.nn.swish

    @nn.compact
    def __call__(self, x, residual=None):
        out = ConvLayer("conv", features=self.features,
                        kernel_size=(3, 3), strides=(2, 2))(x)
        if residual is not None:
            # Match spatial dims if residual is higher-resolution
            if residual.shape[1] > out.shape[1]:
                residual = nn.avg_pool(residual, window_shape=(
                    2, 2), strides=(2, 2), padding="SAME")
            out = jnp.concatenate([out, residual], axis=-1)
        return out


class ResidualBlock(nn.Module):
    """(Optionally conditional) residual block with two convolutions."""
    conv_type: str
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    padding: str = "SAME"
    activation: Callable = jax.nn.swish
    direction: str = None
    res: int = 2
    norm_groups: int = 8

    def setup(self):
        norm_cls = partial(
            nn.GroupNorm, self.norm_groups) if self.norm_groups > 0 else partial(nn.RMSNorm, 1e-5)
        self.norm1 = norm_cls()
        self.norm2 = norm_cls()

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        temb: jax.Array,
    ):
        residual = x
        out = x

        # First conditioning injection
        if temb is not None:
            cond = nn.DenseGeneral(
                features=out.shape[-1], name="temb_projection1")(temb)
            out += jnp.expand_dims(jnp.expand_dims(cond, 1), 1)

        out = self.activation(self.norm1(out))
        out = ConvLayer(self.conv_type, features=self.features,
                        kernel_size=self.kernel_size, strides=self.strides, name="conv1")(out)

        # Second conditioning injection
        if temb is not None:
            cond = nn.DenseGeneral(
                features=self.features, name="temb_projection2")(temb)
            out += jnp.expand_dims(jnp.expand_dims(cond, 1), 1)

        out = self.activation(self.norm2(out))
        out = ConvLayer(self.conv_type, features=self.features,
                        kernel_size=self.kernel_size, strides=self.strides, name="conv2")(out)

        # Adjust residual channels if needed
        if residual.shape != out.shape:
            residual = ConvLayer(self.conv_type, features=self.features, kernel_size=(
                1, 1), strides=1, name="residual_conv")(residual)

        out += residual

        return out


class UNet(nn.Module):
    """Conditional U-Net with optionally embedded labels & class information.

    The network follows the classic encoder–bottleneck–decoder topology
    with residual blocks and (optionally) per-sample conditioning.

    Parameters
    ----------
    out_channels : int, default 1
        Number of channels produced by the final convolution.
    emb_features : list[int]
        Hidden dimensions for the label / class embedding MLP.
    feature_depths : list[int]
    num_res_blocks : int, default 2
        Residual blocks per resolution level.
    num_middle_res_blocks : int, default 1
        Residual blocks at the bottleneck.
    activation : Callable, default jax.nn.gelu
        Activation function used throughout the model.
    norm_groups : int, default 8
        Group count for GroupNorm (0 switches to RMSNorm).
    label_in : {"channel", "conditional"}, default "channel"
        * "channel": concatenate `label` to `x` on the channel axis.
        * "conditional": embed `label` and add as a conditioning vector.
    class_cond : bool, default False
        If True, the model is additionally conditioned on integer class labels.
    n_classes : int, default 101
        Vocabulary size for class embedding when `class_cond=True`.

    Call Parameters
    ---------------
    x : jax.Array
        Input tensor of shape (B, H, W, C).
        Additiional time steps can be wrapped in the channel dimension.
    time : jax.Array
        Stochastic label. Either concatenated or embedded depending on
        `label_in`. Shape matches `x` for "channel" or (B, L) otherwise.
    class_l : jax.Array
        Integer class labels (B,) or (B, 1) when `class_cond=True`.

    Returns
    -------
    jax.Array
        Output tensor of shape (B, H, W, out_channels).
    """
    out_channels: int
    emb_features: list = field(default_factory=lambda: [512, 512])
    feature_depths: list = field(default_factory=lambda: [128, 256, 512])
    num_res_blocks: int = 2
    num_middle_res_blocks: int = 1
    activation: Callable = jax.nn.gelu
    norm_groups: int = 8
    n_classes: int = 3
    is_trunk: bool = False

    def setup(self):
        norm_cls = partial(
            nn.GroupNorm, self.norm_groups) if self.norm_groups > 0 else partial(nn.RMSNorm, 1e-5)
        self.conv_out_norm = norm_cls()

    @nn.compact
    def __call__(self, x, time, class_l):
        if time is not None:
            time_proj = PeriodicTimestep(
                self.emb_features[0], flip_sin_to_cos=True, freq_shift=0)
            time = time_proj(jnp.squeeze(time))
            time = FeedFoward(features=self.emb_features)(time)

        if class_l is not None:
            class_l = nn.Embed(self.n_classes, self.emb_features[0])(class_l)
            class_l = jnp.squeeze(class_l)
            class_l = FeedFoward(features=self.emb_features)(class_l)

        temb = None
        if time is not None and class_l is not None:
            temb = jnp.concatenate(
                [time, class_l], axis=-1)
        if time is None and class_l is not None:
            temb = class_l
        if time is not None and class_l is None:
            temb = time

        conv_type = up_conv_type = down_conv_type = middle_conv_type = "conv"

        # Stem
        x = ConvLayer(
            conv_type, features=self.feature_depths[0], kernel_size=(3, 3))(x)
        downs = [x]

        # Encoder
        for i, dim_out in enumerate(self.feature_depths):
            for j in range(self.num_res_blocks):
                x = ResidualBlock(
                    down_conv_type,
                    name=f"down_{i}_residual_{j}",
                    features=x.shape[-1],
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                )(x, temb)
                downs.append(x)
            if i != len(self.feature_depths) - 1:
                x = Downsample(features=dim_out, scale=2,
                               name=f"down_{i}_downsample")(x)

        # Bottleneck
        middle_dim_out = self.feature_depths[-1]
        for j in range(self.num_middle_res_blocks):
            x = ResidualBlock(middle_conv_type, name=f"middle_res1_{j}", features=middle_dim_out,
                              activation=self.activation, norm_groups=self.norm_groups)(x, temb)
            x = ResidualBlock(middle_conv_type, name=f"middle_res2_{j}", features=middle_dim_out,
                              activation=self.activation, norm_groups=self.norm_groups)(x, temb)

        # Decoder
        for i, dim_out in enumerate(reversed(self.feature_depths)):
            for j in range(self.num_res_blocks):
                x = jnp.concatenate([x, downs.pop()], axis=-1)
                x = ResidualBlock(
                    up_conv_type,
                    name=f"up_{i}_residual_{j}",
                    features=dim_out,
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                )(x, temb)
            if i != len(self.feature_depths) - 1:
                x = Upsample(
                    features=self.feature_depths[-i], scale=2, name=f"up_{i}_upsample")(x)

        # Head
        x = ConvLayer(
            conv_type, features=self.feature_depths[0], kernel_size=(3, 3))(x)
        x = jnp.concatenate([x, downs.pop()], axis=-1)
        x = ResidualBlock(conv_type, name="final_residual",
                          features=self.feature_depths[0], activation=self.activation, norm_groups=self.norm_groups)(x, temb)

        x = self.activation(self.conv_out_norm(x))
        if self.is_trunk:
            return x, temb

        out = ConvLayer(conv_type, features=self.out_channels,
                        kernel_size=(3, 3))(x)
        return out
