import jax
import numpy as np

from gauge.net.mlp import DNN
from gauge.net.unet import UNet
from gauge.utils.tools import pshape


def get_arch(net_cfg, out_channels):

    if net_cfg.arch == "unet":
        net = get_unet_size(net_cfg.size, out_channels, net_cfg.emb_features)
    if net_cfg.arch == "mlp":
        net = DNN(width=128, depth=7, out_features=out_channels)

    return net


def get_unet_size(size, out_channels, emb_features):

    if size == "s":
        net = UNet(
            feature_depths=[96, 128],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=1,
            out_channels=out_channels,
        )
    elif size == "m":
        net = UNet(
            feature_depths=[128, 128, 360],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=2,
            out_channels=out_channels,
        )
    elif size == "l":
        net = UNet(
            feature_depths=[128, 256, 512],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=2,
            out_channels=out_channels,
        )

    return net


def get_network(net_cfg, dataloader, key):

    x_data, class_l = next(iter(dataloader))

    pshape(x_data, class_l, title='dataloader sample')

    out_channels = x_data.shape[-1]
    time = np.ones((x_data.shape[0], 1))

    net = get_arch(net_cfg, out_channels)

    params_init = net.init(key, x_data, time, class_l)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_init))
    print(f"n_params {param_count:,}")

    return net, params_init
