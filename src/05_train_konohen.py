import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from minisom import MiniSom

# Create output directories
Path("../plots/clusters").mkdir(
    parents=True,
    exist_ok=True
)

Path("../models").mkdir(
    parents=True,
    exist_ok=True
)

BACKGROUND = "#111111"
GRID = "#444444"
TEXT = "#ffffff"

# Load normalized fragments
X = np.load(
    "../data/normalized_fragments.npy"
)

print(
    f"Dataset shape: {X.shape}"
)

# Create SOM
som_rows = 6
som_cols = 5

som = MiniSom(
    x=som_rows,
    y=som_cols,
    input_len=21,
    sigma=1.5,
    learning_rate=0.5,
    random_seed=42
)

# Initialize weights
som.random_weights_init(X)

print("Training SOM...")

som.train_random(
    X,
    5000
)

print("Training completed")

# Save weights
weights = som.get_weights()

np.save(
    "../models/som_weights.npy",
    weights
)

print(
    f"Weights shape: {weights.shape}"
)