"""
MLP model for BANKING77 intent classification.
Averages word embeddings into one sentence vector (no sequence/order information).
"""
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 77,
        pad_idx: int = 0,
        embedding_matrix=None,
        freeze_embeddings: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.pad_idx = pad_idx

        # 1. Embedding layer: same as BiLSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings

        # 2. A simple feedforward network on top of the averaged vector
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len) word IDs
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Masked average: only average over real words, not padding
        mask = (x != self.pad_idx).unsqueeze(-1).float()  # (batch, seq_len, 1): 1 for real words, 0 for padding
        summed = (embedded * mask).sum(dim=1)              # (batch, embed_dim)
        counts = mask.sum(dim=1).clamp(min=1)               # (batch, 1): avoid divide-by-zero
        averaged = summed / counts                           # (batch, embed_dim): the sentence vector

        hidden = self.relu(self.fc1(averaged))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)  # (batch, num_classes)
        return logits


if __name__ == "__main__":
    import numpy as np
    from models.mlp.preprocessing import load_vocab

    vocab = load_vocab()
    embedding_matrix = np.load("models/mlp/glove_embeddings.npy")

    model = MLPClassifier(
        vocab_size=len(vocab),
        embedding_matrix=embedding_matrix,
        freeze_embeddings=False,
    )
    print(model)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    dummy_input = torch.randint(0, len(vocab), (4, 50))
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}  (expected: [4, 77])")