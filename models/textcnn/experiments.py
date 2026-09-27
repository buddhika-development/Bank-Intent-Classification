"""
Small set of TextCNN experiments (ablation study).

Every run trains on the TRAIN set and is scored on the VALIDATION set only.
The test set is not used here, so it stays unseen for the final evaluation.

    uv run python models/textcnn/experiments.py
"""
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
RESULTS_DIR = ROOT / "results" / "textcnn"
sys.path.append(str(ROOT))
sys.path.append(str(MODEL_DIR))

from model import TextCNN, count_parameters  # noqa: E402
from preprocessing import build_embedding_matrix, build_vocab, encode_batch  # noqa: E402
from src.common.data import load_splits  # noqa: E402
from train import CONFIG, get_device, make_loader, set_seed, train_model  # noqa: E402

# (name, changes to the baseline settings, seed, use GloVe?)
EXPERIMENTS = [
    ("baseline (seed 42)", {}, 42, True),
    ("baseline (seed 43)", {}, 43, True),
    ("baseline (seed 44)", {}, 44, True),
    ("embeddings frozen", {"freeze_embeddings": True}, 42, True),
    ("embeddings random (no GloVe)", {}, 42, False),
    ("dropout 0.3", {"dropout": 0.3}, 42, True),
    ("dropout 0.7", {"dropout": 0.7}, 42, True),
    ("kernels 3,4,5", {"kernel_sizes": [3, 4, 5]}, 42, True),
    ("kernels 1,2,3", {"kernel_sizes": [1, 2, 3]}, 42, True),
    ("kernel 3 only", {"kernel_sizes": [3]}, 42, True),
    ("filters 50", {"num_filters": 50}, 42, True),
    ("filters 200", {"num_filters": 200}, 42, True),
]


def run_one(name, changes, seed, use_glove, data, device):
    config = {**CONFIG, **changes, "seed": seed}
    set_seed(seed)
    train_loader = make_loader(data["X_train"], data["y_train"], config["batch_size"], True)
    val_loader = make_loader(data["X_val"], data["y_val"], config["batch_size"], False)

    model = TextCNN(
        vocab_size=data["vocab_size"],
        embed_dim=config["embed_dim"],
        num_classes=77,
        kernel_sizes=tuple(config["kernel_sizes"]),
        num_filters=config["num_filters"],
        dropout=config["dropout"],
        pretrained=data["matrix"] if use_glove else None,
        freeze_embeddings=config["freeze_embeddings"],
    ).to(device)

    with contextlib.redirect_stdout(io.StringIO()):   # hide the per-epoch lines
        model, history = train_model(model, train_loader, val_loader, config, device)

    b = history["best_epoch"] - 1
    return {
        "experiment": name,
        "val_acc": history["val_acc"][b],
        "val_loss": history["val_loss"][b],
        "train_acc": history["train_acc"][b],
        "overfit_gap": history["train_acc"][b] - history["val_acc"][b],
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["val_acc"]),
        "parameters": count_parameters(model),
        "train_time_sec": history["train_time_sec"],
    }


def main(experiments=EXPERIMENTS):
    device = get_device()
    print(f"Device: {device}")
    train, val, test, labels = load_splits()
    vocab = build_vocab(train["text"].tolist())
    matrix, _ = build_embedding_matrix(vocab)
    data = {
        "X_train": encode_batch(train["text"], vocab, CONFIG["max_len"]),
        "X_val": encode_batch(val["text"], vocab, CONFIG["max_len"]),
        "y_train": train["label"].values,
        "y_val": val["label"].values,
        "matrix": matrix,
        "vocab_size": len(vocab),
    }

    rows = []
    for name, changes, seed, use_glove in experiments:
        row = run_one(name, changes, seed, use_glove, data, device)
        rows.append(row)
        print(f"{name:32s} val acc {row['val_acc']:.4f} | val loss {row['val_loss']:.4f} "
              f"| gap {row['overfit_gap']:.3f} | params {row['parameters']:,} "
              f"| {row['train_time_sec']:.0f}s")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "experiments.csv", index=False)

    base = df[df["experiment"].str.startswith("baseline")]["val_acc"]
    print(f"\nBaseline over 3 seeds: mean {base.mean():.4f}, "
          f"min {base.min():.4f}, max {base.max():.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["experiment"], df["val_acc"], color="#4c72b0")
    ax.axvspan(base.min(), base.max(), color="grey", alpha=0.25, label="baseline range (3 seeds)")
    ax.set(xlim=(df["val_acc"].min() - 0.02, df["val_acc"].max() + 0.01),
           xlabel="Validation accuracy", title="TextCNN experiments (validation set)")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiments.png", dpi=150)
    plt.close(fig)
    return df


if __name__ == "__main__":
    main()
