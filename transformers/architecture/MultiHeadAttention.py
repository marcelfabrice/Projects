from architecture.Attention import Attention
import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, masked=None):
        super().__init__()

        self.heads = nn.ModuleList([Attention(emb_dim, masked=masked) for _ in range(n_heads)])
        self.W = nn.Linear(n_heads * emb_dim, emb_dim)

    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
            v = q

        Z = torch.cat([h(q, k, v) for h in self.heads], dim=-1)
        return self.W(Z)

