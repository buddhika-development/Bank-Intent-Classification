"""
BiLSTM model for BANKING77 intent classification.
"""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 77,
        pad_idx: int = 0,
        embedding_matrix=None,   # numpy array (vocab_size, embed_dim) or None
        freeze_embeddings: bool = False,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 1. Embedding layer: word ID -> 100-number vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings

        # 2. Bidirectional LSTM: reads the sequence both directions
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. Dropout before the final layer, to reduce overfitting
        self.dropout = nn.Dropout(dropout)

        # 4. Final linear layer: sentence summary -> 77 intent scores
        #    hidden_dim * 2 because bidirectional concatenates both directions
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len) word IDs
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        if lengths is not None:
            # Pack the sequence so the LSTM skips padding positions entirely
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)

        # hidden shape: (num_layers * 2, batch, hidden_dim)
        # last layer's forward direction is hidden[-2], backward is hidden[-1]
        forward_final = hidden[-2]
        backward_final = hidden[-1]
        sentence_repr = torch.cat([forward_final, backward_final], dim=1)  # (batch, hidden_dim*2)

        sentence_repr = self.dropout(sentence_repr)
        logits = self.fc(sentence_repr)  # (batch, num_classes)
        return logits


if __name__ == "__main__":
    import numpy as np
    from models.bilstm.preprocessing import load_vocab

    vocab = load_vocab()
    embedding_matrix = np.load("models/bilstm/glove_embeddings.npy")

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_matrix=embedding_matrix,
        freeze_embeddings=False,  # fine-tuned, as decided
    )
    print(model)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # Fake batch: 4 sentences, seq_len 50
    dummy_input = torch.randint(0, len(vocab), (4, 50))
    dummy_lengths = torch.tensor([8, 12, 50, 3])
    output = model(dummy_input, dummy_lengths)
    print(f"\nOutput shape: {output.shape}  (expected: [4, 77])")