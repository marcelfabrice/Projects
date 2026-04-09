
from Transformer import Transformer
from datasets import load_dataset
import pickle

dataset = load_dataset("wmt14", "de-en")["train"]

src_tgt = []
for idx in range(100000, 200000):
    sample = dataset[idx]["translation"]
    src_tgt.append((sample["de"], sample["en"]))

with open("data/preprocessed/wmt14.data", "wb") as f: pickle.dump(src_tgt, f)

transformer = Transformer( data="data/preprocessed/wmt14.data", 
                          batch_size=64, 
                          lr=0.003,
                          embedding_dim=128,
                          n_attentionheads=4,
                          max_epoch=15 )
transformer.fit()