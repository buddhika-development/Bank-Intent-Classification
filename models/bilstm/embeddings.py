"""
Build a GloVe embedding matrix aligned to our vocabulary.

Run once:
    uv run python -m models.bilstm.embeddings
"""
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
sys.path.append(str(MODEL_DIR.parents[1]))  # repo root, for src.common

from models.bilstm.preprocessing import load_vocab, PAD_TOKEN, UNK_TOKEN

EMBED_DIM = 100
GLOVE_PATH = MODEL_DIR.parents[1] / "data" / "glove" / "glove.6B.100d.txt"


def load_glove_vectors(path: Path) -> dict[str, np.ndarray]:
    """Load GloVe vectors from a local .txt file into a word -> vector dict."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            vectors[word] = vec
    return vectors


def build_embedding_matrix(vocab: dict[str, int]) -> np.ndarray:
    """
    For each word in our vocabulary, look up its GloVe vector.
    Words GloVe doesn't know get a small random vector instead.
    Row 0 (<pad>) is kept as all zeros.
    """
    print(f"Loading GloVe from {GLOVE_PATH}...")
    glove = load_glove_vectors(GLOVE_PATH)
    print(f"Loaded {len(glove)} GloVe words")

    vocab_size = len(vocab)
    matrix = np.random.normal(scale=0.1, size=(vocab_size, EMBED_DIM)).astype(np.float32)
    matrix[vocab[PAD_TOKEN]] = np.zeros(EMBED_DIM, dtype=np.float32)

    found = 0
    for word, idx in vocab.items():
        if word in (PAD_TOKEN, UNK_TOKEN):
            continue
        if word in glove:
            matrix[idx] = glove[word]
            found += 1

    coverage = found / (vocab_size - 2)
    print(f"Found {found}/{vocab_size - 2} words in GloVe ({coverage:.1%} coverage)")
    return matrix


if __name__ == "__main__":
    vocab = load_vocab()
    matrix = build_embedding_matrix(vocab)

    save_path = MODEL_DIR / "glove_embeddings.npy"
    np.save(save_path, matrix)
    print(f"Saved embedding matrix: shape {matrix.shape} -> {save_path}")