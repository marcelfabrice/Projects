from architecture.MultiHeadAttention import MultiHeadAttention
from architecture.FeedForward import FeedForward
from architecture.Embedding import Embedding
from architecture.Encoder import Encoder
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_seq_len, n_attentionheads):
        super().__init__()

        self.embedding = Embedding(vocab_size, emb_dim, max_seq_len)
        self.attention1 = MultiHeadAttention(emb_dim, n_attentionheads, masked=True)
        self.attention2 = MultiHeadAttention(emb_dim, n_attentionheads)
        self.ff = FeedForward(emb_dim)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ln3 = nn.LayerNorm(emb_dim)

        self.encoder = Encoder(vocab_size, emb_dim, max_seq_len, n_attentionheads)
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        x = self.embedding(tgt)
        x = self.ln1(x + self.attention1(x))
        x = self.ln2(x + self.attention2(x, encoder_output, encoder_output))
        x = self.ln3(x + self.ff(x))
        return self.output(x)


 
