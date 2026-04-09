from processing.Vocabulary import Vocabulary
from torch.utils.data import Dataset
import torch.nn as nn
import torch

class TranslationDataset(Dataset):
    def __init__(self, vocabulary : Vocabulary, max_seq_len=10):
        self.vocab = vocabulary
        self.pairs = vocabulary.src_target
        self.max_seq_len = max_seq_len

        # Special Tokens hinzufügen
        self.vocab.add_words(["<bos>", "<eos>", "<pad>"])

        self.bos = self.vocab.WordToIdx("<bos>")
        self.eos = self.vocab.WordToIdx("<eos>")
        self.pad = self.vocab.WordToIdx("<pad>")

    def tokenize(self, sentence):
        return sentence.lower().split()

    def encode(self, tokens):
        return [self.vocab.WordToIdx(t) for t in tokens]

    def pad_sequence(self, seq):
        if len(seq) < self.max_seq_len:
            seq = seq + [self.pad] * (self.max_seq_len - len(seq))
        else:
            seq = seq[:self.max_seq_len]
        return seq

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        de, en = self.pairs[idx]

        de_tokens = self.tokenize(de)
        en_tokens = self.tokenize(en)

        de_ids = self.encode(de_tokens)
        en_ids = self.encode(en_tokens)

        de_ids = de_ids[:self.max_seq_len]
        en_ids = en_ids[:self.max_seq_len - 1]

        src = de_ids + [self.pad] * (self.max_seq_len - len(de_ids))

        tgt_in = [self.bos] + en_ids
        tgt_in = tgt_in + [self.pad] * (self.max_seq_len - len(tgt_in))

        tgt_out = en_ids + [self.eos]
        tgt_out = tgt_out + [self.pad] * (self.max_seq_len - len(tgt_out))

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long)
        )

