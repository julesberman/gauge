from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from gauge.config.config import Config
from gauge.net.head import UnetHead
from gauge.net.mlp import DNN
from gauge.net.skew import SkewNet
from gauge.net.unet import UNet
from gauge.utils.tools import pshape


def get_arch(net_cfg, out_channels, n_heads=1, n_classes=1):

    if net_cfg.arch == "unet":
        net = get_unet_size(net_cfg.size, out_channels,
                            net_cfg.emb_features, n_classes, n_heads)
    if net_cfg.arch == "mlp":
        net = DNN(width=128, depth=6, out_features=out_channels,
                  heads=n_heads, residual=False)
    if net_cfg.arch == "skew":
        k = net_cfg.kernel
        net = SkewNet(n_basis=net_cfg.n_basis, kernel_size=(k, k))
    return net


def get_unet_size(size, out_channels, emb_features, n_classes, n_heads):

    if n_heads == 1:
        UnetConstructor = partial(UNet, out_channels)
    else:
        UnetConstructor = partial(UnetHead, n_heads, out_channels)
    if size == "s":
        net = UnetConstructor(
            feature_depths=[96, 128],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=1,
            n_classes=n_classes
        )
    elif size == "m":
        net = UnetConstructor(
            feature_depths=[128, 128, 360],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=2,
            n_classes=n_classes
        )
    elif size == "l":
        net = UnetConstructor(
            feature_depths=[128, 256, 512],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=2,
            n_classes=n_classes
        )

    return net


def get_network(cfg: Config, dataloader, n_classes, key):

    n_fields = cfg.gauge.n_fields
    heads = cfg.net.heads
    x_data, class_l = next(iter(dataloader))
    out_channels = x_data.shape[-1]
    time = np.ones((x_data.shape[0], 1))

    pshape(x_data, class_l, title='dataloader sample')

    if heads == 'channel':
        out_channels = out_channels*n_fields
        net = get_arch(cfg.net, out_channels, n_classes=n_classes)
        params_init = net.init(key, x_data, time, class_l)

        def apply_fn(*args):
            y = net.apply(*args)
            return rearrange(y, 'B ... (C F) -> F B ... C', F=n_fields)

    if heads == 'class':
        class_l = jnp.ones_like(time, dtype=jnp.int32)
        net = get_arch(cfg.net, out_channels, n_classes=n_fields)
        params_init = net.init(key, x_data, time, class_l)

        def apply_fn(params, x, time, idx):
            class_l = jnp.ones_like(time) * idx
            class_l = jnp.asarray(class_l, dtype=jnp.int32)
            return net.apply(params, x, time, class_l)

    if heads == 'multi':
        n_heads = n_fields
        net = get_arch(cfg.net, out_channels,
                       n_heads=n_heads, n_classes=n_classes)
        params_init = net.init(key, x_data, time, class_l)

        def apply_fn(params, x, time, class_l):
            return net.apply(params, x, time, class_l)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_init))
    print(f"n_params {param_count:,}")

    return net, apply_fn, params_init
