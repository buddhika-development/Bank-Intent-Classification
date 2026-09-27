"""
Preprocessing for the TextCNN model.

Steps: tokenize -> build vocabulary (train only) -> pad to fixed length
-> build a GloVe embedding matrix for the vocabulary.

Rules match the shared decisions (max length 50, keep stopwords, vocabulary
from training data only) so all four models are compared fairly.

Run this file to test it:
    uv run python models/textcnn/preprocessing.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent
sys.path.append(str(MODEL_DIR.parents[1]))  # repo root, so "src.common" can be imported

MAX_LEN = 50            # from EDA: covers more than 99% of sentences
MIN_FREQ = 2            # a word must appear at least twice to get its own id
PAD_TOKEN = "<pad>"     # fills empty space after short sentences
UNK_TOKEN = "<unk>"     # stands in for words not in the vocabulary
GLOVE_NAME = "glove-wiki-gigaword-100"  # 100-dimensional GloVe vectors
EMBED_DIM = 100
SEED = 42


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into word tokens (stopwords are kept)."""
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(train_texts: list[str]) -> dict[str, int]:
    """
    Map word -> integer id using TRAINING sentences only (no data leakage).
    <pad> is 0 and <unk> is 1. Real words follow, most frequent first.
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
    """One sentence -> list of exactly max_len integer ids (cut or padded)."""
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)]
    ids = ids[:max_len]
    return ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))


def encode_batch(texts, vocab: dict[str, int], max_len: int = MAX_LEN) -> torch.Tensor:
    """Many sentences -> one tensor of shape (number_of_sentences, max_len)."""
    return torch.tensor([encode(t, vocab, max_len) for t in texts], dtype=torch.long)


def build_embedding_matrix(vocab: dict[str, int]) -> tuple[torch.Tensor, int]:
    """
    Build a (vocab_size, EMBED_DIM) table. Row i holds the GloVe vector of word i.
    Words missing from GloVe get small random numbers. <pad> is all zeros.
    Returns the matrix and how many vocabulary words were found in GloVe.
    """
    import gensim.downloader as api  # imported here so other functions work without it

    glove = api.load(GLOVE_NAME)  # downloads once, then reused from cache
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.1, size=(len(vocab), EMBED_DIM)).astype("float32")
    matrix[vocab[PAD_TOKEN]] = 0.0

    found = 0
    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]
            found += 1
    return torch.from_numpy(matrix), found


def save_vocab(vocab: dict[str, int], path: Path = None):
    with open(path or MODEL_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)


def load_vocab(path: Path = None) -> dict[str, int]:
    with open(path or MODEL_DIR / "vocab.json") as f:
        return json.load(f)


if __name__ == "__main__":
    from src.common.data import load_splits

    train, val, test, labels = load_splits()

    vocab = build_vocab(train["text"].tolist())
    save_vocab(vocab)
    print(f"Vocabulary size: {len(vocab)}")

    example = train["text"].iloc[0]
    print(f"\nExample: {example}")
    print(f"Tokens:  {tokenize(example)}")
    print(f"Encoded: {encode(example, vocab)[:15]} ... (first 15 of {MAX_LEN})")

    X_train = encode_batch(train["text"], vocab)
    X_val = encode_batch(val["text"], vocab)
    X_test = encode_batch(test["text"], vocab)
    print(f"\nX_train shape: {tuple(X_train.shape)}")
    print(f"X_val shape:   {tuple(X_val.shape)}")
    print(f"X_test shape:  {tuple(X_test.shape)}")

    matrix, found = build_embedding_matrix(vocab)
    print(f"\nEmbedding matrix shape: {tuple(matrix.shape)}")
    print(f"Words found in GloVe: {found} of {len(vocab)} ({found / len(vocab):.1%})")
