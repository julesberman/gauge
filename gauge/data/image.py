"""Utilities for preparing TFDS image datasets used throughout Gauge."""
from typing import Tuple

import tensorflow_datasets as tfds

from gauge.config.config import Config


def get_image_dataset(config: Config):
    """Download TFDS splits as numpy arrays normalized to [-1, 1]."""
    name = config.dataset.lower()
    tfds_name, image_shape, has_labels, n_classes = _resolve_dataset(name)
    data_dir = '/scratch/jmb1174/tensorflow_datasets/'
    builder = tfds.builder(tfds_name, data_dir=data_dir)
    builder.download_and_prepare(file_format="array_record")
    sources = tfds.data_source(tfds_name, data_dir=data_dir)
    train_ds = sources["train"]

    return train_ds, image_shape, n_classes


def _resolve_dataset(name: str) -> Tuple:
    """Map friendly dataset names to TFDS builders + target image shapes."""
    if name == "mnist":
        return "mnist", (28, 28, 1), True, 10
    if name in {"cifar10", "cfiar10"}:
        return "cifar10", (32, 32, 3), True, 10
    if name == "flowers":
        return "tf_flowers", (128, 128, 3), False, 1
    if name == "celeba":
        return "celeb_a", (64, 64, 3), False, 1

    raise ValueError(
        "Unsupported dataset {!r}. Use mnist, cifar10, flowers, celeba, or lsu_bedrooms.".format(
            name)
    )
