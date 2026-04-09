from processing.Dataloader import TranslationDataset
from processing.Vocabulary import Vocabulary
from architecture.Decoder import Decoder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import torch
import time

class Transformer:
    def __init__(self, data, batch_size=1, max_seq_len=10,
                  embedding_dim=32, n_attentionheads=4, lr=0.01, max_epoch=50, device="mps"):
        with open(data, "rb") as f: en_de_pairs = pickle.load(f)[20000:60000]
        self.max_epoch = max_epoch

        self.vocab = Vocabulary(en_de_pairs)
        self.dataset = TranslationDataset(self.vocab, max_seq_len=10)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.device = torch.device(device) # MPS auf mac ist irgendwie langsamer

        self.decoder = Decoder(vocab_size=self.vocab._vocab_size, 
                               emb_dim=embedding_dim, 
                               max_seq_len=max_seq_len, 
                               n_attentionheads=n_attentionheads).to(self.device)
        
        self.lossfn = nn.CrossEntropyLoss(ignore_index=self.vocab.WordToIdx("<pad>"))
        self.optim = torch.optim.AdamW(self.decoder.parameters(), lr=lr)

        print("device: ", self.device)

    def fit(self):
        for epoch in range(self.max_epoch):
            start = time.time()
            total_loss, ind = 0, 0
            for src, tgt_in, tgt_out in self.loader:

                src = src.to(self.device)
                tgt_in = tgt_in.to(self.device)
                tgt_out = tgt_out.to(self.device)

                logits = self.decoder(src, tgt_in)
                B, T, C = logits.shape

                loss = self.lossfn(
                    logits.view(B*T, C),
                    tgt_out.view(B*T)
                )

                total_loss += loss.item()
                ind += 1

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            end = time.time()
            print(f"loss {(total_loss/ind)*1000} noch {((50-epoch-1)*(end-start))/60:.2f}min")

        with open("transformer.model", "wb") as f: pickle.dump(self, f)

    def generate(self, prompt):
        self.decoder.eval()

        tokens = prompt.lower().split()

        src_ids = [self.vocab.WordToIdx(w) for w in tokens]
        src = torch.tensor(src_ids, dtype=torch.long).to(self.device)

        bos = self.vocab.WordToIdx("<bos>")
        generated = [bos]

        for _ in range(10):
            tgt = torch.tensor(generated, dtype=torch.long).to(self.device)

            with torch.no_grad():
                logits = self.decoder(src, tgt)

            next_token_logits = logits[-1]
            next_token = torch.argmax(next_token_logits).item()

            generated.append(next_token)

            if self.vocab.IdxToWord(next_token) == "<eos>":
                break

        translated = [self.vocab.IdxToWord(i) for i in generated[1:]][:-1]

        print("Output:", " ".join(translated))

