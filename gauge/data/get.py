"""Utilities for preparing TFDS image datasets used throughout Gauge."""


from gauge.config.config import Config
from gauge.data.image import get_image_dataset
from gauge.data.toy import get_2d_dataset


def get_dataset(config: Config):

    if 'toy' in config.dataset:
        ds, shape = get_2d_dataset(config.dataset)
    else:
        ds, shape = get_image_dataset(config)

    return ds, shape
