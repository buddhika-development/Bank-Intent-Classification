"""
Evaluate the trained BiLSTM: learning curves + final test set metrics.

Run:
    uv run python -m models.bilstm.evaluate
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
sys.path.append(str(ROOT))

from src.common.data import load_splits
from src.common.metrics import (
    compute_metrics, save_metrics, save_classification_report,
    plot_confusion_matrix, print_and_save_top_confusions
)
from models.bilstm.preprocessing import load_vocab
from models.bilstm.model import BiLSTMClassifier
from models.bilstm.train import IntentDataset

RESULTS_DIR = ROOT / "results" / "bilstm"


def plot_learning_curves(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], label="train loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="val loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over training"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train accuracy", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="val accuracy", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over training"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curves.png", dpi=150)
    plt.show()
    print(f"Saved: {RESULTS_DIR / 'learning_curves.png'}")


def evaluate_on_test(model, test_loader, device, label_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    start = time.time()
    with torch.no_grad():
        for ids, lengths, labels in test_loader:
            ids, lengths = ids.to(device), lengths.to(device)
            logits = model(ids, lengths)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    inference_time = time.time() - start

    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["inference_time_seconds"] = inference_time
    metrics["n_test_samples"] = len(all_labels)

    print(f"\n{'='*50}")
    print("FINAL TEST SET RESULTS (touched once)")
    print(f"{'='*50}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"Macro ROC-AUC:      {metrics['macro_roc_auc']:.4f}")
    print(f"Inference time:     {inference_time:.2f}s for {len(all_labels)} sentences "
          f"({inference_time/len(all_labels)*1000:.2f}ms/sentence)")

    save_metrics(metrics, RESULTS_DIR / "test_metrics.json")
    save_classification_report(all_labels, all_preds, label_names, RESULTS_DIR / "classification_report.txt")

    cm = plot_confusion_matrix(
        all_labels, all_preds, label_names,
        title="Confusion Matrix (Test Set) - BiLSTM",
        save_path=RESULTS_DIR / "confusion_matrix.png",
    )
    print_and_save_top_confusions(cm, label_names, RESULTS_DIR / "top_confusions.txt")

    return all_preds, all_labels, cm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(RESULTS_DIR / "history.json") as f:
        history = json.load(f)
    plot_learning_curves(history)

    with open(RESULTS_DIR / "config.json") as f:
        config = json.load(f)

    train_df, val_df, test_df, label_names = load_splits()
    vocab = load_vocab()
    embedding_matrix = np.load(MODEL_DIR / "glove_embeddings.npy")

    model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=config["embed_dim"], hidden_dim=config["hidden_dim"],
        num_classes=len(label_names), embedding_matrix=embedding_matrix,
        freeze_embeddings=config["freeze_embeddings"], dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=device))
    print(f"Loaded best model (val loss {config['best_val_loss']:.4f}, "
          f"{config['epochs_run']} epochs, {config['n_parameters']:,} parameters)")

    test_ds = IntentDataset(test_df["text"], test_df["label"], vocab)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    evaluate_on_test(model, test_loader, device, label_names)


if __name__ == "__main__":
    main()