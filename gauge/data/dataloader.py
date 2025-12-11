import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gauge.config.config import Config
from gauge.utils.tools import get_cpu_count


class InMemoryDataSource:
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

    def __len__(self):
        return self._images.shape[0]

    def __getitem__(self, i):
        if self._labels is None:
            return self._images[i]
        else:
            return self._images[i],  self._labels[i]


def SimpleBatcher(X, y=None, batch_size=32, seed=None):
    rng = np.random.default_rng(seed)
    n = len(X)

    while True:
        idx = rng.integers(0, n, size=batch_size)  # with replacement
        if y is None:
            yield X[idx], None
        else:
            yield X[idx], y[idx]


def materialize_data_source(ds, use_tqdm=False, normalize_img=True):
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

    if normalize_img:
        images = (images - 127.5) / 127.5

    return images, labels


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
    normalize = cfg.data.normalize
    shuffle = cfg.sample.shuffle if cfg.sample.shuffle is not None else shuffle

    if cfg.sample.materialize:
        images, labels = materialize_data_source(
            dataset, use_tqdm=True, normalize_img=normalize)
        if not use_labels:
            labels = None
        return SimpleBatcher(images, y=labels, batch_size=batch_size)

    n_cpus = get_cpu_count()
    num_workers = max(1, n_cpus - 4)
    print(f"found {n_cpus} cpu cores, using {num_workers} workers")

    def numpy_collate(batch):
        # batch: list of items, each item is either
        #   (image,) OR (image, label)
        if isinstance(batch[0], tuple):
            # ZIP across batch → stack each field
            batch = list(zip(*batch))
            return tuple(np.stack(field) for field in batch)
        else:
            return np.stack(batch)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4,
        drop_last=True,
        shuffle=True,
        collate_fn=numpy_collate,
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
