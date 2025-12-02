import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from gauge.config.config import Config
from gauge.utils.tools import get_cpu_count


def get_dataloader(
    cfg: Config,
    dataset,
    batch_size: int = 128,
    shuffle: bool = True,

):
    """
    Create an *infinite* PyTorch-style batch iterator from a TFDS dataset,
    without loading the whole dataset into RAM.

    Yields:
        (images, labels_or_none)

        images:  torch.float32, shape [B, C, H, W], in [0, 1]
        labels:  torch.long, shape [B], or None if use_labels=False
    """

    use_labels = cfg.data.class_labels
    batch_size = cfg.sample.batch_size

    n_cpus = get_cpu_count()
    num_workers = max(1, n_cpus - 4)

    # Map-style dataset already implemented by TFDS, so we just need a sampler.
    sampler = (
        RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    )

    def collate_with_labels(batch):
        # batch: list of dicts {"image": np.ndarray, "label": int}
        images = [b["image"] for b in batch]
        labels = [b["label"] for b in batch]

        x = np.stack(images, axis=0)  # [B, H, W, C]
        x = torch.from_numpy(x).permute(
            0, 3, 1, 2).float() / 255.0  # [B, C, H, W]
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    def collate_without_labels(batch):
        images = [b["image"] for b in batch]
        x = np.stack(images, axis=0)  # [B, H, W, C]
        x = torch.from_numpy(x).permute(
            0, 3, 1, 2).float() / 255.0  # [B, C, H, W]
        return x

    collate_fn = collate_with_labels if use_labels else collate_without_labels

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Wrap the (finite) DataLoader into an infinite generator
    def infinite():
        while True:
            for batch in loader:
                if use_labels:
                    images, labels = batch
                    yield images, labels
                else:
                    images = batch
                    yield images, None

    return infinite()
