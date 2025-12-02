"""Utilities for preparing TFDS image datasets used throughout Gauge."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow_datasets as tfds

from gauge.config.config import Config


def get_dataset(config: Config):
    """Download TFDS splits as numpy arrays normalized to [-1, 1]."""
    name = config.dataset.lower()
    tfds_name, image_shape = _resolve_dataset(name)
    data_dir = os.environ.get("TFDS_DATA_DIR")
    builder = tfds.builder(tfds_name, data_dir=data_dir)
    builder.download_and_prepare()  # no-op if data already exists

    sources = tfds.data_source(tfds_name, data_dir=data_dir)
    train_ds = sources["train"]

    return train_ds, image_shape


def _resolve_dataset(name: str) -> Tuple:
    """Map friendly dataset names to TFDS builders + target image shapes."""
    if name == "mnist":
        return "mnist", (28, 28, 1)
    if name in {"cifar10", "cfiar10"}:
        return "cifar10", (32, 32, 3)
    if name == "flowers":
        return "tf_flowers", (128, 128, 3)
    if name == "celeba":
        return "celeb_a", (64, 64, 3)
    if name in {"bedrooms", "lsu_bedrooms", "lsun_bedrooms", "lsun_bedroom"}:
        return "lsun/bedroom", (64, 64, 3)
    raise ValueError(
        "Unsupported dataset {!r}. Use mnist, cifar10, flowers, celeba, or lsu_bedrooms.".format(
            name)
    )


def _infer_label_key(
    builder: tfds.core.dataset_builder.DatasetBuilder,
) -> Tuple[Optional[str], bool]:
    """Find the label key (if any) exposed by the TFDS builder."""
    supervised_keys = builder.info.supervised_keys
    if supervised_keys:
        return supervised_keys[1], True

    features = getattr(builder.info, "features", None)
    if features is not None:
        for candidate in ("label", "labels", "class"):
            if candidate in features:
                return candidate, False
    return None, False


def _materialize_split(
    builder: tfds.core.dataset_builder.DatasetBuilder,
    split: str,
    target_shape: Tuple[int, int, int],
    label_key: Optional[str],
    use_supervised: bool,
) -> Dict[str, Optional[np.ndarray]]:
    """Load a TFDS split into contiguous numpy arrays."""
    n = builder.info.splits[split].num_examples
    images = np.empty((n,) + target_shape, dtype=np.float32)
    labels: Optional[np.ndarray] = None

    as_supervised = bool(label_key and use_supervised)
    ds = builder.as_dataset(split=split, as_supervised=as_supervised)
    iterator: Iterable = tfds.as_numpy(ds)

    for idx, example in enumerate(iterator):
        label = None
        if as_supervised:
            image, label = example
        elif isinstance(example, dict):
            image = example.get("image", next(iter(example.values())))
            if label_key is not None:
                label = example.get(label_key)
        elif isinstance(example, tuple):
            image = example[0]
            label = example[1] if len(example) > 1 else None
        else:
            image = example

        np_img = np.asarray(image)
        if np_img.ndim == 2:
            np_img = np_img[..., None]

        if label is not None and label_key is not None:
            if labels is None:
                labels = np.empty((n,), dtype=np.int32)
            labels[idx] = np.int32(label)

    return {"images": _normalize_to_minus_one_one(images), "labels": labels}


def _normalize_to_minus_one_one(images: np.ndarray) -> np.ndarray:
    """Map 0–255 intensities to [-1, 1]."""
    return images.astype(np.float32, copy=False) / 127.5 - 1.0
