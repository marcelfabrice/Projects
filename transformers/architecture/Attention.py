import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, emb_dim, masked=None):
        super().__init__()
        self.d_k = emb_dim
        self.masked = masked
        self.val = nn.Linear(emb_dim, emb_dim) # (T, E)
        self.key = nn.Linear(emb_dim, emb_dim) 
        self.query = nn.Linear(emb_dim, emb_dim) 
        self.attention_scores = None

    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
            v = q

        Q = self.query(q)
        K = self.key(k)
        V = self.val(v)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k**0.5)

        if self.masked == True:
            self.attention_scores = scores.masked_fill(
                torch.triu(torch.ones_like(scores), diagonal=1).bool(),
                float('-inf'))
        else:
            self.attention_scores = scores

        weights = nn.functional.softmax(self.attention_scores, dim=-1)
        self.last_attention = weights  # store for plotting

        attention = weights @ V
        return attention
    

