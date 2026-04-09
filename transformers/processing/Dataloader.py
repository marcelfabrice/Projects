from processing.Vocabulary import Vocabulary
from torch.utils.data import Dataset
import torch.nn as nn
import tiktoken
import torch

class TranslationDataset(Dataset):
    def __init__(self, vocabulary: Vocabulary, max_seq_len=10):
        self.vocab = vocabulary
        self.pairs = vocabulary.src_target
        self.max_seq_len = max_seq_len

        self.enc = tiktoken.get_encoding("cl100k_base")

        self.vocab.add_words(["<bos>", "<eos>", "<pad>"])

        self.bos = self.vocab.WordToIdx("<bos>")
        self.eos = self.vocab.WordToIdx("<eos>")
        self.pad = self.vocab.WordToIdx("<pad>")

        self.encoded_data = []

        for de, en in self.pairs:
            de_tokens = de.lower().split()
            en_tokens = en.lower().split()

            de_ids = [self.vocab.WordToIdx(t) for t in de_tokens]
            en_ids = [self.vocab.WordToIdx(t) for t in en_tokens]

            de_ids = de_ids[:self.max_seq_len]
            en_ids = en_ids[:self.max_seq_len - 1]

            src = de_ids + [self.pad] * (self.max_seq_len - len(de_ids))

            tgt_in = [self.bos] + en_ids
            tgt_in = tgt_in + [self.pad] * (self.max_seq_len - len(tgt_in))

            tgt_out = en_ids + [self.eos]
            tgt_out = tgt_out + [self.pad] * (self.max_seq_len - len(tgt_out))

            self.encoded_data.append((
                torch.tensor(src, dtype=torch.long),
                torch.tensor(tgt_in, dtype=torch.long),
                torch.tensor(tgt_out, dtype=torch.long)
            ))

    def pad_sequence(self, seq):
        if len(seq) < self.max_seq_len:
            seq = seq + [self.pad] * (self.max_seq_len - len(seq))
        else:
            seq = seq[:self.max_seq_len]
        return seq

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

