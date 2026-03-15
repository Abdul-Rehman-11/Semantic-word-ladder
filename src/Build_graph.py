import numpy as np
import pickle

GLOVE_FILE = "data/glove.100d.20000.txt"
K = 100

print("Loading...")

words = []
vectors = []

with open(GLOVE_FILE, "r", encoding="utf8") as f:
    for line in f:
        parts = line.split()
        words.append(parts[0])
        vectors.append(np.array(parts[1:], dtype=float))

vectors = np.array(vectors)

#this makes the norm A and B of unit length which makes the cosinesimilarity faster and saves us time as our program just need to calculate the dot product now
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

print("Building neighbor graph...")

neighbors = {}

for i, word in enumerate(words):
    sims = vectors @ vectors[i]
    top_indices = np.argsort(-sims)[1:K+1]
    neighbors[word] = [words[j] for j in top_indices]

print("saving the graph")

with open("neighbors.pkl", "wb") as f:
    pickle.dump(neighbors, f)

print("Done! enjoy faster searching :)")