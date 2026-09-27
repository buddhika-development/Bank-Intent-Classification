"""
TextCNN model (Kim, 2014 style) for BANKING77 intent classification.

Flow:  word ids -> embeddings -> parallel Conv1d filters (2, 3, 4 words wide)
       -> ReLU -> max-pool over time -> join -> dropout -> linear -> 77 scores

Run this file to test the shapes:
    uv run python models/textcnn/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        num_classes: int = 77,
        kernel_sizes: tuple = (2, 3, 4),
        num_filters: int = 100,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)  # start from GloVe
        self.embedding.weight.requires_grad = not freeze_embeddings

        # One Conv1d per window size. Each looks at k neighbouring words.
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len) word ids
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)        # Conv1d wants (batch, embed_dim, seq_len)

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))                 # (batch, num_filters, seq_len - k + 1)
            p = F.max_pool1d(h, h.size(2))        # strongest match per filter
            pooled.append(p.squeeze(2))           # (batch, num_filters)

        features = torch.cat(pooled, dim=1)       # (batch, num_filters * n_kernels)
        return self.fc(self.dropout(features))    # (batch, num_classes) raw scores


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (used in the model comparison)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = TextCNN(vocab_size=1455, pretrained=torch.randn(1455, 100))
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    fake_batch = torch.randint(0, 1455, (8, 50))  # 8 fake sentences of 50 ids
    out = model(fake_batch)
    print(f"Input shape:  {tuple(fake_batch.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (expected (8, 77))")
