class Tokenizer:
    def __init__(self, pairs):
        self.pairs_to_translate = pairs

    def tokenize(self):
        all_words = []
        for de, en in self.pairs_to_translate:
            all_words += de.lower().split()
            all_words += en.lower().split()
        return all_words
