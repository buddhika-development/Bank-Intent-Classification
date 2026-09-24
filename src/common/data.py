"""
Shared data loading for all four models.

Run ONCE to create the splits:
    uv run python -m src.common.data

Every model then loads the same files with load_splits().
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SIZE = 0.10  # 10% of the original train set becomes validation

# Official source: PolyAI (Casanueva et al., 2020), CC-BY-4.0
BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

ROOT = Path(__file__).resolve().parents[2]  # the repo root folder
SPLIT_DIR = ROOT / "data" / "splits"


def make_splits():
    train_full = pd.read_csv(f"{BASE_URL}/train.csv")
    test = pd.read_csv(f"{BASE_URL}/test.csv")

    # Map intent names -> numbers 0..76, sorted so it's the same on every machine
    labels = sorted(train_full["category"].unique())
    label2id = {name: i for i, name in enumerate(labels)}
    train_full["label"] = train_full["category"].map(label2id)
    test["label"] = test["category"].map(label2id)

    # Safety check: every test intent must also exist in train
    assert test["label"].notna().all(), "Test set has an intent not seen in train!"

    # Stratified split: each intent keeps the same proportion in train and val
    train, val = train_test_split(
        train_full,
        test_size=VAL_SIZE,
        stratify=train_full["label"],
        random_state=SEED,
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLIT_DIR / "train.csv", index=False)
    val.to_csv(SPLIT_DIR / "val.csv", index=False)
    test.to_csv(SPLIT_DIR / "test.csv", index=False)
    with open(SPLIT_DIR / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")
    print(f"intents: {len(labels)}")


def load_splits():
    """Used by every model: returns train, val, test DataFrames and the label names."""
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    with open(SPLIT_DIR / "labels.json") as f:
        labels = json.load(f)
    return train, val, test, labels


if __name__ == "__main__":
    make_splits()