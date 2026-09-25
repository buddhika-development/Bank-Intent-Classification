"""
Preprocessing for the BiLSTM model: build vocabulary, convert text to
padded integer sequences.
"""
import re
import json
from collections import Counter
from pathlib import Path

import torch

MAX_LEN = 50          # from EDA: covers >99% of sentences
MIN_FREQ = 2           # a word must appear at least twice to get its own ID
PAD_TOKEN = "<pad>"    # fills empty space after short sentences
UNK_TOKEN = "<unk>"    # stands in for words never seen in training

MODEL_DIR = Path(__file__).resolve().parent


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens. Same rule used in EDA."""
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(train_texts: list[str]) -> dict[str, int]:
    """
    Build a word -> integer id mapping from the TRAINING texts only.
    Special tokens <pad>=0 and <unk>=1 are added first, then real words
    sorted by frequency (most common first), so ids are reproducible.
    """
    counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common():
        if freq >= MIN_FREQ:
            vocab[word] = len(vocab)

    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    """
    Turn one sentence into a fixed-length list of integer ids.
    Unknown words -> <unk>. Short sentences padded with <pad> at the end.
    Long sentences truncated to max_len.
    """
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)]
    ids = ids[:max_len]
    ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


def encode_batch(texts, vocab: dict[str, int], max_len: int = MAX_LEN) -> torch.Tensor:
    """Encode a list/Series of sentences into a single tensor of shape (N, max_len)."""
    return torch.tensor([encode(t, vocab, max_len) for t in texts], dtype=torch.long)


def save_vocab(vocab: dict[str, int], path: Path = None):
    path = path or MODEL_DIR / "vocab.json"
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_vocab(path: Path = None) -> dict[str, int]:
    path = path or MODEL_DIR / "vocab.json"
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    sys.path.append(str(MODEL_DIR.parents[1]))  # repo root, for src.common
    from src.common.data import load_splits

    train, val, test, labels = load_splits()

    vocab = build_vocab(train["text"].tolist())
    save_vocab(vocab)
    print(f"Vocabulary size: {len(vocab)}")

    example = train["text"].iloc[0]
    print(f"\nExample: {example}")
    print(f"Tokens:  {tokenize(example)}")
    print(f"Encoded: {encode(example, vocab)[:15]} ... (truncated to first 15 of {MAX_LEN})")

    X_train = encode_batch(train["text"], vocab)
    X_val = encode_batch(val["text"], vocab)
    X_test = encode_batch(test["text"], vocab)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")