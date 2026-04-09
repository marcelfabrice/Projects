import pickle
from Transformer import Transformer
import os

with open("transformer.model", "rb") as f: transformer = pickle.load(f)

while True:
    prompt = input("Übersetze: ")
    transformer.generate(prompt)

