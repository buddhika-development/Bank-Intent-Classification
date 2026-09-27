import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "../../results/distilbert"
DATA_DIR = "../../data/splits"


def load_test_set():
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    # Same JSON array format as train.py — list position is the id,
    # no ids stored explicitly in the file itself.
    with open(f"{DATA_DIR}/labels.json") as f:
        label_names = json.load(f)
    return test_df, label_names


@torch.no_grad()
def predict_batch(texts, tokenizer, model, device, batch_size=64, max_length=64):
    model.eval()
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
    return all_preds


def evaluator(run_id: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_df, label_names = load_test_set()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    preds = predict_batch(test_df["text"].tolist(), tokenizer, model, device)
    labels = test_df["label"].tolist()

    print("\n=== Overall ===")
    report = classification_report(
        labels, preds, target_names=label_names, digits=3, zero_division=0,
        output_dict=True,
    )
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.3f}")

    # Per-class breakdown, sorted worst-first — this is the part people skip
    # and then wonder why "93% accuracy" doesn't feel right in production.
    per_class = pd.DataFrame(report).T.iloc[:-3]  # drop accuracy/macro/weighted rows
    per_class = per_class.sort_values("f1-score")
    print("\n=== Worst 10 intents by F1 (inspect these first) ===")
    print(per_class.head(10)[["precision", "recall", "f1-score", "support"]])

    per_class.to_csv("data/per_class_metrics.csv")
    print("\nFull per-class metrics saved to data/per_class_metrics.csv")

    # Confusion matrix, restricted to the worst 15 classes so it's actually
    # readable (a full 77x77 matrix is unreadable as an image).
    worst_labels = per_class.head(15).index.tolist()
    worst_ids = [label_names.index(name) for name in worst_labels]

    cm = confusion_matrix(labels, preds, labels=worst_ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[label_names[i] for i in worst_ids],
        yticklabels=[label_names[i] for i in worst_ids],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix — 15 worst-performing intents")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("data/confusion_matrix_worst15.png", dpi=150)
    print("Confusion matrix saved to data/confusion_matrix_worst15.png")