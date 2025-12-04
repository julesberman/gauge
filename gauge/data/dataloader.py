import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm

from gauge.config.config import Config
from gauge.utils.tools import get_cpu_count


def _normalize_images(x: np.ndarray, channel_first: bool) -> np.ndarray:
    if channel_first:
        x = np.transpose(x, (0, 3, 1, 2))  # [B, C, H, W]
    return (x - 127.5) / 127.5


def materialize_data_source(ds, use_tqdm=False):
    """
    Materialize a tfds-like data_source into RAM as numpy arrays.
    """
    # --- Check available RAM ---
    per_sample_bytes = np.asarray(ds[0]["image"]).nbytes

    est_total_bytes = per_sample_bytes * len(ds)
    est_total_gb = est_total_bytes / (1024 ** 3)
    print(f"Estimated materialized size: ~{est_total_gb:.2f} GB "
          f"({len(ds)} samples × {per_sample_bytes/1024**2:.2f} MB/sample)")

    idx_iter = range(len(ds))
    if use_tqdm:
        idx_iter = tqdm(idx_iter, desc="materializing dataset", colour='green')

    images = []
    labels = []
    has_label = False

    for i in idx_iter:
        ex = ds[i]
        img = np.asarray(ex["image"])
        images.append(img)

        if "label" in ex:
            has_label = True
            labels.append(np.asarray(ex["label"]))

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0) if has_label else None

    class InMemoryDataSource:
        def __init__(self, images, labels):
            self._images = images
            self._labels = labels

        def __len__(self):
            return self._images.shape[0]

        def __getitem__(self, i):
            if self._labels is None:
                return {"image": self._images[i]}
            else:
                return {"image": self._images[i], "label": self._labels[i]}

    return InMemoryDataSource(images, labels)


def get_dataloader(
    cfg: Config,
    dataset,
    batch_size: int = 128,
    shuffle: bool = True,

):
    """
    Create an *infinite* PyTorch-style batch iterator from a TFDS dataset.

    When cfg.sample.materialize is True, the entire dataset is staged in host
    memory once and batches are served from RAM for maximum throughput.
    Otherwise we stream from disk via a standard DataLoader.

    Yields:
        (images, labels_or_none)

        images:  np.float32, shape [B, C, H, W], in [-1, 1]
        labels:  np.int32, shape [B], or None if use_labels=False
    """

    use_labels = cfg.data.class_labels
    batch_size = cfg.sample.batch_size
    channel_first = cfg.sample.channel_first
    shuffle = cfg.sample.shuffle if cfg.sample.shuffle is not None else shuffle

    if cfg.sample.materialize:
        dataset = materialize_data_source(dataset, use_tqdm=True)

    n_cpus = get_cpu_count()
    num_workers = max(1, n_cpus - 4)
    print(f"found {n_cpus} cpu cores, using {num_workers} workers")

    # Map-style dataset already implemented by TFDS, so we just need a sampler.
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    def collate_with_labels(batch):
        # batch: list of dicts {"image": np.ndarray, "label": int}
        images = [np.asarray(b["image"], dtype=np.float32) for b in batch]
        labels = [np.asarray(b["label"], dtype=np.int32) for b in batch]

        x = np.stack(images, axis=0)
        x = _normalize_images(x, channel_first)
        y = np.asarray(labels)
        return x, y

    def collate_without_labels(batch):
        images = [np.asarray(b["image"], dtype=np.float32) for b in batch]
        x = np.stack(images, axis=0)
        x = _normalize_images(x, channel_first)
        return x

    collate_fn = collate_with_labels if use_labels else collate_without_labels

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=4

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
