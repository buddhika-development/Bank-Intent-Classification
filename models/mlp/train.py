"""
Train the MLP model on BANKING77.

Run:
    uv run python -m models.mlp.train
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
sys.path.append(str(ROOT))

from src.common.data import load_splits
from models.mlp.preprocessing import build_vocab, save_vocab, load_vocab, encode, MAX_LEN
from models.mlp.model import MLPClassifier

# ---- Config: everything that defines this run, so it's reproducible ----
SEED = 42
HIDDEN_DIM = 128
EMBED_DIM = 100
DROPOUT = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 4
FREEZE_EMBEDDINGS = False

RESULTS_DIR = ROOT / "results" / "mlp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class IntentDataset(Dataset):
    """Wraps encoded sentences + labels so DataLoader can batch them.
    No 'lengths' needed here -- the MLP masks padding internally."""

    def __init__(self, texts, labels, vocab):
        self.vocab = vocab
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(ids)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * ids.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            n += ids.size(0)

    return total_loss / n, correct / n


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df, label_names = load_splits()

    vocab_path = MODEL_DIR / "vocab.json"
    if vocab_path.exists():
        vocab = load_vocab()
    else:
        vocab = build_vocab(train_df["text"].tolist())
        save_vocab(vocab)
    print(f"Vocabulary size: {len(vocab)}")

    embedding_matrix = np.load(MODEL_DIR / "glove_embeddings.npy")

    train_ds = IntentDataset(train_df["text"], train_df["label"], vocab)
    val_ds = IntentDataset(val_df["text"], val_df["label"], vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(label_names),
        embedding_matrix=embedding_matrix,
        freeze_embeddings=FREEZE_EMBEDDINGS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_path = MODEL_DIR / "best_model.pt"

    start_time = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> new best model saved (val loss {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping: no improvement for {PATIENCE} epochs.")
                break

    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time:.1f} seconds ({training_time/60:.1f} min)")

    config = {
        "seed": SEED, "hidden_dim": HIDDEN_DIM, "embed_dim": EMBED_DIM, "dropout": DROPOUT,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "freeze_embeddings": FREEZE_EMBEDDINGS,
        "epochs_run": len(history["train_loss"]), "training_time_seconds": round(training_time, 1),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_loss": best_val_loss,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(RESULTS_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved: {best_model_path}")
    print(f"Saved: {RESULTS_DIR / 'config.json'}")
    print(f"Saved: {RESULTS_DIR / 'history.json'}")


if __name__ == "__main__":
    main()