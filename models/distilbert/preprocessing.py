"""
Preprocessing for the DistilBERT model: WordPiece tokenization with the
pretrained tokenizer (no custom vocabulary, no GloVe).
"""
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64  # counted in WordPiece subwords, not words (BiLSTM/MLP use 50 words)


def load_tokenizer(path=MODEL_NAME):
    return AutoTokenizer.from_pretrained(path)


def tokenize_dataset(df, tokenizer, max_len: int = MAX_LEN) -> Dataset:
    """Turn a (text, label) DataFrame into a tokenized HF Dataset.
    Padding is left to DataCollatorWithPadding so each batch is padded to its own longest sentence."""
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_len),
        batched=True,
    )
    return ds.remove_columns(["text"])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.common.data import load_splits

    train, val, test, labels = load_splits()
    tokenizer = load_tokenizer()

    example = train["text"].iloc[0]
    print(f"Example: {example}")
    print(f"Tokens:  {tokenizer.tokenize(example)}")
    print(f"Encoded: {tokenizer(example)['input_ids']}")

    lengths = [len(ids) for ids in tokenizer(train["text"].tolist())["input_ids"]]
    covered = sum(n <= MAX_LEN for n in lengths) / len(lengths)
    print(f"\nMax subword length: {max(lengths)} | {covered:.2%} of train sentences fit in MAX_LEN={MAX_LEN}")
