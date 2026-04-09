import pickle
from Transformer import Transformer

with open("transformer.model", "rb") as f: transformer = pickle.load(f)

transformer.generate("hi there i like you")

