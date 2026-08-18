"""Loads the Fashion-MNIST idx-ubyte files downloaded into this directory
(train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-
ubyte.gz, t10k-labels-idx1-ubyte.gz - gitignored, not committed).

No torchvision in this environment, so this parses the canonical IDX format
directly instead of going through torchvision.datasets.FashionMNIST.
"""
import gzip
import os

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 0x00000803, f"{path}: bad magic {magic:#x}, expected idx3-ubyte"
        n, h, w = (int.from_bytes(f.read(4), "big") for _ in range(3))
        buf = f.read(n * h * w)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, h, w)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 0x00000801, f"{path}: bad magic {magic:#x}, expected idx1-ubyte"
        n = int.from_bytes(f.read(4), "big")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def load(data_dir=DATA_DIR):
    """Returns (X, Y): all 70000 Fashion-MNIST images/labels, uint8, combining
    the dataset's own train (60000) and test (10000) splits into one pool -
    callers that want a specific train/val/test ratio (not the dataset's
    original 60k/10k one) should re-split this themselves.

    X: (70000, 28, 28) uint8, pixel values in [0, 255].
    Y: (70000,) uint8, class index in [0, 9] - see CLASS_NAMES.
    """
    x_train = _read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = _read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    x_test = _read_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = _read_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    X = np.concatenate([x_train, x_test], axis=0)
    Y = np.concatenate([y_train, y_test], axis=0)
    return X, Y


if __name__ == "__main__":
    X, Y = load()
    print(f"X: {X.shape} {X.dtype}, Y: {Y.shape} {Y.dtype}")
    counts = np.bincount(Y, minlength=10)
    for i, (name, c) in enumerate(zip(CLASS_NAMES, counts)):
        print(f"  [{i}] {name:12s} {c}")
