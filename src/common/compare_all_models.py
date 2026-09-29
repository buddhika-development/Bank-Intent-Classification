"""
Compare all four models (MLP, BiLSTM, TextCNN, DistilBERT) on the shared
test set, using each model's own result files.

Handles the fact that each model's code writes slightly different file
layouts and field names, since each was implemented independently.

Run once all four models have final results:
    uv run python -m src.common.compare_all_models
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Each model's code was written independently, so paths and field names
# differ. This table is the single place that maps each model's own
# layout to a common set of fields.
MODEL_CONFIGS = {
    "mlp": {
        "test_metrics": "mlp/experiments/fine_tuned/test_metrics.json",
        "config": "mlp/experiments/fine_tuned/config.json",
        "params_key": ("config", "n_parameters"),
        "train_time_key": ("config", "training_time_seconds"),
        "infer_time_key": ("metrics", "inference_time_seconds"),
        "n_test_key": ("metrics", "n_test_samples"),
        "model_file_size_mb": 0.64,  # verified: models/mlp/experiments/fine_tuned/best_model.pt, 676,154 bytes
    },
    "bilstm": {
        "test_metrics": "bilstm/experiments/fine_tuned/test_metrics.json",
        "config": "bilstm/experiments/fine_tuned/config.json",
        "params_key": ("config", "n_parameters"),
        "train_time_key": ("config", "training_time_seconds"),
        "infer_time_key": ("metrics", "inference_time_seconds"),
        "n_test_key": ("metrics", "n_test_samples"),
        "model_file_size_mb": 1.53,  # verified: models/bilstm/experiments/fine_tuned/best_model.pt, 1,607,605 bytes
    },
    "textcnn": {
        "test_metrics": "textcnn/test_metrics.json",
        "config": "textcnn/config.json",
        "params_key": ("metrics", "parameters"),
        "train_time_key": ("metrics", "train_time_sec"),
        "infer_time_key": ("metrics", "inference_time_total_sec"),
        "n_test_key": ("metrics", "num_test_sentences"),
        "model_file_size_mb": 1.02,  # from TextCNN's own report: best_model.pt, 1,039,670 bytes
    },
    "distilbert": {
        "test_metrics": "distilbert/experiments/fine_tuned/test_metrics.json",
        "config": "distilbert/experiments/fine_tuned/config.json",
        "params_key": ("config", "n_parameters"),
        "train_time_key": ("config", "training_time_seconds"),
        "infer_time_key": ("metrics", "inference_time_seconds"),
        "n_test_key": ("metrics", "n_test_samples"),
        "model_file_size_mb": 268.1,  # from DistilBERT's own report: model.safetensors, 268,063,276 bytes
    },
}

DISPLAY_NAMES = {
    "mlp": "MLP", "bilstm": "BiLSTM", "textcnn": "TextCNN", "distilbert": "DistilBERT"
}


def _get(d, path):
    """path is (which_file, key): which_file is 'metrics' or 'config'."""
    which, key = path
    return d[which][key]


def load_model(name, cfg):
    with open(RESULTS_ROOT / cfg["test_metrics"]) as f:
        metrics = json.load(f)
    with open(RESULTS_ROOT / cfg["config"]) as f:
        config = json.load(f)
    d = {"metrics": metrics, "config": config}

    return {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "macro_roc_auc": metrics["macro_roc_auc"],
        "n_parameters": _get(d, cfg["params_key"]),
        "training_time_seconds": _get(d, cfg["train_time_key"]),
        "inference_time_seconds": _get(d, cfg["infer_time_key"]),
        "n_test_samples": _get(d, cfg["n_test_key"]),
        "model_file_size_mb": cfg["model_file_size_mb"],
    }


def main():
    results = {}
    for name, cfg in MODEL_CONFIGS.items():
        try:
            results[name] = load_model(name, cfg)
        except FileNotFoundError as e:
            print(f"Skipping {name}: {e}")

    order = [m for m in ["mlp", "bilstm", "textcnn", "distilbert"] if m in results]

    # ---- Plain-text summary table ----
    print(f"\n{'Model':<12}{'Accuracy':>10}{'Macro F1':>10}{'ROC-AUC':>10}"
          f"{'Params':>14}{'Size(MB)':>10}{'Train(s)':>12}{'Infer(ms/sent)':>16}")
    print("-" * 94)
    for name in order:
        r = results[name]
        infer_ms = r["inference_time_seconds"] / r["n_test_samples"] * 1000
        print(f"{DISPLAY_NAMES[name]:<12}{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}"
              f"{r['macro_roc_auc']:>10.4f}{r['n_parameters']:>14,}{r['model_file_size_mb']:>10.2f}"
              f"{r['training_time_seconds']:>12.1f}{infer_ms:>16.3f}")

    print("\nNote: all models were trained on separate members' machines (different CPUs; "
          "see each model's own report for exact hardware). Training/inference times are "
          "indicative, not strictly comparable across models.")

    # ---- Chart 1: accuracy, macro F1, ROC-AUC grouped bars ----
    metric_labels = ["Accuracy", "Macro F1", "ROC-AUC"]
    x = np.arange(len(metric_labels))
    width = 0.8 / len(order)
    colors = {"mlp": "#ff7f0e", "bilstm": "#1f77b4", "textcnn": "#2ca02c", "distilbert": "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(order):
        r = results[name]
        values = [r["accuracy"], r["macro_f1"], r["macro_roc_auc"]]
        offset = (i - (len(order) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=DISPLAY_NAMES[name], color=colors[name])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0.75, 1.02)
    ax.set_title("BANKING77: All Four Models, Test Set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path1 = OUTPUT_DIR / "all_models_metrics.png"
    plt.savefig(path1, dpi=150)
    plt.show()
    print(f"\nSaved: {path1}")

    # ---- Chart 2: accuracy vs parameter count (the efficiency trade-off) ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for name in order:
        r = results[name]
        ax.scatter(r["n_parameters"], r["accuracy"] * 100, s=180, color=colors[name],
                   label=DISPLAY_NAMES[name], zorder=3, edgecolors="black", linewidths=0.8)
        ax.annotate(f"{DISPLAY_NAMES[name]}\n{r['accuracy']*100:.1f}%",
                    (r["n_parameters"], r["accuracy"] * 100),
                    textcoords="offset points", xytext=(10, 6), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Model Size")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    path2 = OUTPUT_DIR / "all_models_accuracy_vs_params.png"
    plt.savefig(path2, dpi=150)
    plt.show()
    print(f"Saved: {path2}")

    with open(OUTPUT_DIR / "all_models_summary.json", "w") as f:
        json.dump({DISPLAY_NAMES[n]: results[n] for n in order}, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'all_models_summary.json'}")


if __name__ == "__main__":
    main()