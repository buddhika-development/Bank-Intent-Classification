"""
DistilBERT model for BANKING77 intent classification.
Pretrained 6-layer transformer encoder + a classification head on the [CLS] token.
"""
from transformers import AutoModelForSequenceClassification

from models.distilbert.preprocessing import MODEL_NAME


def build_model(label_names: list[str], freeze_encoder: bool = False, path: str = MODEL_NAME):
    """
    freeze_encoder=True  -> only the classification head is trained (DistilBERT as a fixed
                            feature extractor; the equivalent of frozen GloVe for MLP/BiLSTM)
    freeze_encoder=False -> the whole network is fine-tuned
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        num_labels=len(label_names),
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )
    for param in model.distilbert.parameters():
        param.requires_grad = not freeze_encoder
    return model


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.common.data import load_splits

    _, _, _, label_names = load_splits()

    for freeze in (False, True):
        model = build_model(label_names, freeze_encoder=freeze)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze_encoder={freeze}: total {n_params:,} | trainable {n_trainable:,}")
