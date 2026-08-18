"""Trains FashionMNISTModel (fashion_model.py) on Fashion-MNIST with a
65:25:10 train:val:test split, class-balanced by construction (Fashion-MNIST
is already exactly balanced - 7000 images/class over 70000 total - so a
per-class stratified split keeps every split balanced too, rather than
leaving it to chance the way a plain random split would).

Plain torch only - no brevitas/quantization here, per the request. Runs on
GPU automatically if torch.cuda.is_available(); this environment's GPU is
currently blocked by a driver/library version mismatch (kernel module vs
NVML - needs a reboot or driver reload with root, which this session doesn't
have), so this ran on CPU here.

    python -m deepsocflow.py.brevitas.fashion_train
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from deepsocflow.py.brevitas.fashion_mnist import load, CLASS_NAMES
from deepsocflow.py.brevitas.fashion_model import FashionMNISTModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "fashion_cnn.pt")

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.65, 0.25, 0.10
BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 6           # early stop if val loss doesn't improve for this many epochs
LR = 1e-3
WEIGHT_DECAY = 1e-2     # AdamW's own weight decay
SEED = 0


def stratified_split(X, Y, seed=SEED):
    """Per-class split into (train, val, test) at TRAIN_FRAC:VAL_FRAC:TEST_FRAC.
    Splitting each class's own images independently - rather than shuffling
    the whole 70000 and cutting once - is what keeps every split's class
    distribution balanced, not just the split sizes."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(Y):
        idx = np.where(Y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)  # class-grouped order would bias early minibatches otherwise
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def to_dataset(X, Y, idx, mean, std):
    x = X[idx].astype(np.float32) / 255.0
    x = (x - mean) / std
    x = torch.from_numpy(x).unsqueeze(1)          # (N,28,28) -> (N,1,28,28)
    y = torch.from_numpy(Y[idx].astype(np.int64))
    return TensorDataset(x, y)


def run_epoch(model, loader, device, criterion, optimizer=None):
    """optimizer=None runs in eval mode (no grad, no update); otherwise trains."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += x.size(0)

    return total_loss / total_n, total_correct / total_n


def per_class_accuracy(model, loader, device, n_classes=10):
    model.eval()
    correct = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            for c in range(n_classes):
                mask = y == c
                total[c] += mask.sum().item()
                correct[c] += (preds[mask] == c).sum().item()
    return correct / np.maximum(total, 1)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    X, Y = load()
    train_idx, val_idx, test_idx = stratified_split(X, Y)
    print(f"split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        counts = np.bincount(Y[idx], minlength=10)
        print(f"  {name} per-class counts: {counts.tolist()} "
              f"(min={counts.min()}, max={counts.max()})")

    # Normalization stats from the TRAIN split only - using val/test here would
    # leak information about them into training.
    train_pixels = X[train_idx].astype(np.float32) / 255.0
    mean, std = float(train_pixels.mean()), float(train_pixels.std())
    print(f"train-set normalization: mean={mean:.4f} std={std:.4f}")

    train_ds = to_dataset(X, Y, train_idx, mean, std)
    val_ds = to_dataset(X, Y, val_idx, mean, std)
    test_ds = to_dataset(X, Y, test_idx, mean, std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FashionMNISTModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, optimizer=None)
        dt = time.time() - t0

        print(f"epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({dt:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "mean": mean, "std": std,
                "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
            }, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"early stop: no val_loss improvement for {PATIENCE} epochs")
                break

    # Final test evaluation with the BEST checkpoint (not necessarily the last
    # epoch's weights), matching what "test" is supposed to measure.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, device, criterion, optimizer=None)
    print(f"\nbest checkpoint: epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    class_acc = per_class_accuracy(model, test_loader, device)
    print("\nper-class test accuracy:")
    for name, acc in zip(CLASS_NAMES, class_acc):
        print(f"  {name:12s} {acc:.4f}")

    print(f"\nsaved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
