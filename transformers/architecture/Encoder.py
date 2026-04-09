from architecture.MultiHeadAttention import MultiHeadAttention
from architecture.FeedForward import FeedForward
from architecture.Embedding import Embedding
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_seq_len, n_attentionheads):
        super().__init__()

        self.embedding = Embedding(vocab_size, emb_dim, max_seq_len)
        self.attention = MultiHeadAttention(emb_dim, n_attentionheads)
        self.ff = FeedForward(emb_dim)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.embedding(x)
        subx = self.attention(x)
        normed = self.ln1( subx + x )
    
        forwarded = self.ff(normed)
        output = self.ln2(normed + forwarded)

        return output


