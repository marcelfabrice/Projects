import pickle
import os
from Transformer import Transformer

with open("transformer.model", "rb") as f: transformer = pickle.load(f)

while True:
    prompt = input("Übersetze: ")
    os.system("clear")
    transformer.generate(prompt)

