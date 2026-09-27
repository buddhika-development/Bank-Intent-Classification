"""
Shared evaluation metrics for all four models.

Every model's evaluation script should import compute_metrics() and
plot_confusion_matrix() from here, so all four models' results are
computed identically and are honestly comparable.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, y_probs=None) -> dict:
    """
    Compute the standard metric set for a multi-class classifier.

    y_true, y_pred: 1D arrays of integer label ids
    y_probs: optional 2D array (n_samples, n_classes) of predicted
             probabilities, needed for ROC-AUC. If not provided,
             roc_auc is omitted.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }

    if y_probs is not None:
        y_probs = np.asarray(y_probs)
        metrics["macro_roc_auc"] = float(
            roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        )

    return metrics


def save_metrics(metrics: dict, path: Path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_classification_report(y_true, y_pred, label_names, path: Path):
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    with open(path, "w") as f:
        f.write(report)
    return report


def plot_confusion_matrix(y_true, y_pred, label_names, title: str, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True, square=True,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} \u2014 Accuracy {accuracy:.2%}")
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    return cm


def print_and_save_top_confusions(cm, label_names, save_path: Path, top_n: int = 15):
    """
    Find the most common mistakes: (true intent, predicted intent, count) triples,
    ignoring the diagonal (correct predictions).
    """
    confusions = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], label_names[i], label_names[j]))

    confusions.sort(reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP {top_n} MOST CONFUSED PAIRS (true \u2192 predicted)")
    print(f"{'='*70}")
    lines = [f"{'count':>5}  {'true intent':<45} -> predicted intent"]
    for count, true_label, pred_label in confusions[:top_n]:
        line = f"{count:>5}  {true_label:<45} -> {pred_label}"
        print(line)
        lines.append(line)

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {save_path}")

    return confusions