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
    def __init__(self, data, batch_size=1, max_seq_len=10, embedding_dim=32, n_attentionheads=4):
        with open(data, "rb") as f: en_de_pairs = pickle.load(f)[:4000]

        self.vocab = Vocabulary(en_de_pairs)
        self.dataset = TranslationDataset(self.vocab, max_seq_len=10)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.decoder = Decoder(self.vocab._vocab_size, emb_dim=embedding_dim, max_seq_len=max_seq_len, n_attentionheads=n_attentionheads)
        self.lossfn = nn.CrossEntropyLoss(ignore_index=self.vocab.WordToIdx("<pad>"))
        self.optim = torch.optim.AdamW(self.decoder.parameters(), lr=0.001)

    def fit(self):
        for epoch in range(50):
            start = time.time()
            loss, ind = 0,0
            for src, tgt_in, tgt_out in self.loader:
                src, tgt_in, tgt_out = src[0], tgt_in[0], tgt_out[0] # Noch kein batch implementiert
                logits = self.decoder(src, tgt_in)
                T, C = logits.shape

                loss = self.lossfn(
                    logits.view(T, C),
                    tgt_out.view(T)
                )

                loss+=loss.item()
                ind+=1

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            end = time.time()
            print(f"loss {(loss/ind)} noch {((50-epoch-1)*(end-start))/60:.2f}min")

        with open("transformer.model", "wb") as f: pickle.dump(self, f)

    def generate(self, prompt):
        self.decoder.eval()

        tokens = prompt.lower().split()

        src_ids = [self.vocab.WordToIdx(w) for w in tokens]
        src = torch.tensor(src_ids, dtype=torch.long)

        bos = self.vocab.WordToIdx("<bos>")
        generated = [bos]

        for _ in range(10):
            tgt = torch.tensor(generated, dtype=torch.long)

            with torch.no_grad():
                logits = self.decoder(src, tgt)

            next_token_logits = logits[-1]
            next_token = torch.argmax(next_token_logits).item()

            generated.append(next_token)

            if self.vocab.IdxToWord(next_token) == "<eos>":
                break

        translated = [self.vocab.IdxToWord(i) for i in generated[1:]][:-1]

        print("Input:", prompt)
        print("Output:", " ".join(translated))


    def plot_attention(self, prompt):
        self.decoder.eval()

        tokens = prompt.lower().split()
        src_ids = [self.vocab.WordToIdx(w) for w in tokens]
        src = torch.tensor(src_ids, dtype=torch.long)

        bos = self.vocab.WordToIdx("<bos>")
        generated = [bos]

        for _ in range(3):   
            tgt = torch.tensor(generated)

            logits = self.decoder(src, tgt)
            next_token = torch.argmax(logits[-1]).item()

            generated.append(next_token)

        with torch.no_grad():
            encoder_output = self.decoder.encoder(src)
            x = self.decoder.embedding(tgt)

            _ = self.decoder.attention1(x)
            attn1 = self.decoder.attention1.heads[0].last_attention

            _ = self.decoder.attention2(x, encoder_output, encoder_output)
            attn2 = self.decoder.attention2.heads[0].last_attention

        src_words = tokens
        tgt_words = [self.vocab.IdxToWord(i) for i in generated]

        plt.figure(figsize=(6, 5))
        plt.title("Masked Self-Attention")
        plt.imshow(attn1.cpu(), aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(tgt_words)), tgt_words, rotation=45)
        plt.yticks(range(len(tgt_words)), tgt_words)
        plt.xlabel("Key tokens")
        plt.ylabel("Query tokens")

        plt.figure(figsize=(6, 5))
        plt.title("Cross-Attention")
        plt.imshow(attn2.cpu(), aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(src_words)), src_words, rotation=45)
        plt.yticks(range(len(tgt_words)), tgt_words)
        plt.xlabel("Encoder (input words)")
        plt.ylabel("Decoder (generated words)")

        plt.tight_layout()
        plt.show()



transformer = Transformer(data="data/preprocessed/translation.data")
transformer.fit()