import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(0.3)            
        )

    def forward(self, x):
        return self.net(x)

