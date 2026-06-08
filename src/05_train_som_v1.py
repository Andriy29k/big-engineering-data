import numpy as np

from pathlib import Path
from minisom import MiniSom

Path("../models").mkdir(
    parents=True,
    exist_ok=True
)

X = np.load(
    "../data/normalized_fragments_v1.npy"
)

print(
    f"Dataset shape: {X.shape}"
)

som_rows = 10
som_cols = 7

som = MiniSom(
    x=som_rows,
    y=som_cols,
    input_len=20,
    sigma=1.5,
    learning_rate=0.5,
    random_seed=42
)

som.random_weights_init(X)

print("Training SOM V1...")

som.train_random(
    X,
    10000
)

weights = som.get_weights()

np.save(
    "../models/som_weights_v1.npy",
    weights
)

print()
print("Training completed")
print(
    f"Weights shape: {weights.shape}"
)