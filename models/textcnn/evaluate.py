"""
Final evaluation of the trained TextCNN on the TEST set.

Uses the shared metric functions in src/common/metrics.py, so the score is
calculated exactly like for the other three models.
Run this only after training and tuning are finished: the test set is used
for the final score and nothing else.

    uv run python models/textcnn/evaluate.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files; do not open a window

import pandas as pd
import torch

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
RESULTS_DIR = ROOT / "results" / "textcnn"
sys.path.append(str(ROOT))
sys.path.append(str(MODEL_DIR))

from model import TextCNN, count_parameters  # noqa: E402
from preprocessing import build_vocab, encode_batch  # noqa: E402
from src.common.data import load_splits  # noqa: E402
from src.common.metrics import (  # noqa: E402
    compute_metrics,
    plot_confusion_matrix,
    print_and_save_top_confusions,
    save_classification_report,
    save_metrics,
)


def predict(model, X, device, batch_size=64):
    """Return class probabilities (N, num_classes) for all sentences in X."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            scores = model(X[i:i + batch_size].to(device))
            outputs.append(torch.softmax(scores, dim=1).cpu())
    return torch.cat(outputs).numpy()


def main():
    with open(RESULTS_DIR / "config.json") as f:
        config = json.load(f)
    with open(RESULTS_DIR / "history.json") as f:
        history = json.load(f)

    device = torch.device("cpu")  # the official run was trained on CPU; also fair timing
    train, val, test, labels = load_splits()

    vocab = build_vocab(train["text"].tolist())        # from TRAIN only
    X_test = encode_batch(test["text"], vocab, config["max_len"])
    y_test = test["label"].values

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        num_classes=len(labels),
        kernel_sizes=tuple(config["kernel_sizes"]),
        num_filters=config["num_filters"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=device))
    model.to(device)

    predict(model, X_test[:64], device)                # warm-up, not timed
    start = time.time()
    probs = predict(model, X_test, device)
    inference_sec = time.time() - start
    y_pred = probs.argmax(axis=1)

    # Shared metrics: accuracy, macro precision/recall/F1, macro ROC-AUC
    metrics = compute_metrics(y_test, y_pred, probs)
    metrics.update({
        "parameters": count_parameters(model),
        "train_time_sec": history["train_time_sec"],
        "best_epoch": history["best_epoch"],
        "inference_time_total_sec": inference_sec,
        "inference_ms_per_sentence": 1000 * inference_sec / len(X_test),
        "device": str(device),
        "num_test_sentences": int(len(X_test)),
    })
    for k, v in metrics.items():
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, RESULTS_DIR / "test_metrics.json")
    save_classification_report(y_test, y_pred, labels, RESULTS_DIR / "per_class_report.txt")
    cm = plot_confusion_matrix(y_test, y_pred, labels, "TextCNN",
                               RESULTS_DIR / "confusion_matrix.png")
    print_and_save_top_confusions(cm, labels, RESULTS_DIR / "top_confusions.txt")

    # All wrong sentences, for the error analysis in the report
    wrong = test.assign(predicted=[labels[i] for i in y_pred], confidence=probs.max(axis=1))
    wrong = wrong[y_pred != y_test][["text", "category", "predicted", "confidence"]]
    wrong.to_csv(RESULTS_DIR / "misclassified_examples.csv", index=False)
    print(f"\nMisclassified: {len(wrong)} of {len(test)} sentences")


if __name__ == "__main__":
    main()
