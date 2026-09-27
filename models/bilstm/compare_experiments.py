"""
Compare the frozen vs fine-tuned embedding experiments side by side.

Run after both experiments have been trained and evaluated:
    uv run python -m models.bilstm.compare_experiments
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
sys.path.append(str(ROOT))

EXPERIMENTS = ["fine_tuned", "frozen"]
EXPERIMENTS_DIR = ROOT / "results" / "bilstm" / "experiments"


def load_experiment(name):
    with open(EXPERIMENTS_DIR / name / "test_metrics.json") as f:
        metrics = json.load(f)
    with open(EXPERIMENTS_DIR / name / "config.json") as f:
        config = json.load(f)
    return metrics, config


def main():
    results = {name: load_experiment(name) for name in EXPERIMENTS}

    # ---- Print a plain-text table ----
    print(f"\n{'Metric':<20} " + "".join(f"{name:>15}" for name in EXPERIMENTS))
    print("-" * (20 + 15 * len(EXPERIMENTS)))
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc"]
    for key in metric_keys:
        row = f"{key:<20} " + "".join(f"{results[name][0][key]:>15.4f}" for name in EXPERIMENTS)
        print(row)
    print()
    for key, label in [("epochs_run", "Epochs to converge"), ("training_time_seconds", "Training time (s)")]:
        row = f"{label:<20} " + "".join(f"{results[name][1][key]:>15}" for name in EXPERIMENTS)
        print(row)

    # ---- Bar chart: the 5 headline metrics, side by side ----
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(EXPERIMENTS):
        m = results[name][0]
        values = [m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"], m["macro_roc_auc"]]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=name.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0.75, 1.02)  # zoom in, since all values sit in this range
    ax.set_title("BiLSTM: Fine-tuned vs Frozen GloVe Embeddings (Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = EXPERIMENTS_DIR / "comparison_fine_tuned_vs_frozen.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()