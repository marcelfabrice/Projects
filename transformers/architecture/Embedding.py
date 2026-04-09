import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, x : torch.Tensor):
        T = x.size(dim=-1)

        tok = self.token_embedding(x)  # (seq_len, emb_dim)

        positions = torch.arange(T)
        pos = self.position_embedding(positions)  # (T, E)

        return tok + pos  # (T, E)

