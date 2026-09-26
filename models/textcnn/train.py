"""
Training code for the TextCNN model.

Uses only the train and validation sets. The test set is NOT touched here.
Early stopping: training stops when validation loss stops improving,
and the best weights are restored.

Run the full training:
    uv run python models/textcnn/train.py
"""
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
RESULTS_DIR = ROOT / "results" / "textcnn"
sys.path.append(str(ROOT))
sys.path.append(str(MODEL_DIR))

from model import TextCNN, count_parameters  # noqa: E402
from preprocessing import build_embedding_matrix, build_vocab, encode_batch  # noqa: E402

CONFIG = {
    "seed": 42,
    "max_len": 50,
    "embed_dim": 100,
    "kernel_sizes": [2, 3, 4],
    "num_filters": 100,
    "dropout": 0.5,
    "freeze_embeddings": False,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "max_epochs": 30,
    "patience": 5,   # stop after this many epochs without validation improvement
}


def set_seed(seed: int):
    """Make runs repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(X, torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only measures."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            scores = model(xb)
            loss = loss_fn(scores, yb)
            if training:
                optimizer.zero_grad()   # forget the previous batch's gradients
                loss.backward()         # backpropagation: find which dials caused the error
                optimizer.step()        # turn the dials a little
            total_loss += loss.item() * len(yb)
            correct += (scores.argmax(dim=1) == yb).sum().item()
            count += len(yb)
    return total_loss / count, correct / count


def train_model(model, train_loader, val_loader, config, device):
    """Train with early stopping. Returns the model with best weights, history, and training time."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0

    start = time.time()
    for epoch in range(1, config["max_epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, device)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)
        print(f"Epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_val_loss:
            best_val_loss, best_epoch, bad_epochs = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                print(f"Early stopping at epoch {epoch} (best epoch was {best_epoch})")
                break

    train_time = time.time() - start
    model.load_state_dict(best_state)   # go back to the best version
    history["best_epoch"] = best_epoch
    history["train_time_sec"] = train_time
    return model, history


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="validation")
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="validation")
    ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for a in ax:
        a.axvline(history["best_epoch"], color="grey", linestyle="--", label="best epoch")
        a.legend()
    fig.suptitle("TextCNN learning curves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(config=CONFIG):
    from src.common.data import load_splits

    set_seed(config["seed"])
    device = get_device()
    print(f"Device: {device}")

    train, val, test, labels = load_splits()
    vocab = build_vocab(train["text"].tolist())
    X_train = encode_batch(train["text"], vocab, config["max_len"])
    X_val = encode_batch(val["text"], vocab, config["max_len"])
    matrix, _ = build_embedding_matrix(vocab)

    train_loader = make_loader(X_train, train["label"].values, config["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, val["label"].values, config["batch_size"], shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        num_classes=len(labels),
        kernel_sizes=tuple(config["kernel_sizes"]),
        num_filters=config["num_filters"],
        dropout=config["dropout"],
        pretrained=matrix,
        freeze_embeddings=config["freeze_embeddings"],
    ).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}\n")

    model, history = train_model(model, train_loader, val_loader, config, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")   # *.pt is ignored by git
    with open(RESULTS_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    plot_history(history, RESULTS_DIR / "learning_curves.png")

    best = history["best_epoch"] - 1
    print(f"\nBest epoch: {history['best_epoch']} | val loss {history['val_loss'][best]:.4f} "
          f"| val acc {history['val_acc'][best]:.4f}")
    print(f"Training time: {history['train_time_sec']:.1f} sec")
    return model, history


if __name__ == "__main__":
    main()
