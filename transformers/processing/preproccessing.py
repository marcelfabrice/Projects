import pickle

#from datasets import load_dataset
#dataset = load_dataset("wmt14", "de-en")

new_lines = []

with open("deu.txt", "r", encoding="utf-8") as f:
    for line in f:
        cleaned = line.split("CC-BY 2.0")[0]
        new_lines.append(cleaned.strip())

with open("cleaned.txt", "w", encoding="utf-8") as f:
    for line in new_lines:
        f.write(line + "\n")

import re

pairs = []

with open("cleaned.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Split bei Satzende (. ! ?)
        sentences = re.split(r"[.!?]", line)

        # leere Strings entfernen
        sentences = [s.strip() for s in sentences if s.strip()]

        # nur wenn genau 2 Sätze vorhanden
        if len(sentences) >= 2:
            pairs.append((sentences[0], sentences[1]))


with open("translation.data", "wb") as f: pickle.dump(pairs, f)


