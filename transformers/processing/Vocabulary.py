import torch

class Vocabulary:
    def __init__(self, src_target):
        self.src_target = src_target
        self.words = []
        
        for src, tgt in src_target:
            self.words += src.lower().split()
            self.words += tgt.lower().split()

        self.special_tokens = ["<unk>"]
        self.words += self.special_tokens

        self._vocab = sorted(list(set(self.words)))
        self._stoi = {w: i for i, w in enumerate(self._vocab)}
        self._itos = {i: w for w, i in self._stoi.items()}
        self._vocab_size = len(self._vocab)

        self.unk = self._stoi["<unk>"]

        self.data = torch.tensor([self._stoi[w] for w in self.words], dtype=torch.long)

    def WordToIdx(self, word):
        return self._stoi.get(word, self.unk)

    def IdxToWord(self, idx):
        return self._itos.get(idx, "<unk>")
    
    def get_data(self):
        return self.data

    def add_word(self, word):
        if word not in self._stoi:
            idx = len(self._vocab)
            self._vocab.append(word)
            self._stoi[word] = idx
            self._itos[idx] = word
            self._vocab_size += 1

    def add_words(self, words):
        for word in words:
            self.add_word(word)
