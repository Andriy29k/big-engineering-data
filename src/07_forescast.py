import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from minisom import MiniSom

# Create output directory
Path("../plots/forecast").mkdir(
    parents=True,
    exist_ok=True
)

BACKGROUND = "#111111"
GRID = "#444444"
TEXT = "#ffffff"

PREFIX_COLOR = "#ff9933"
PATTERN_COLOR = "#3399ff"
FORECAST_COLOR = "#33cc66"
REAL_COLOR = "#ff4444"

WINDOW_SIZE = 21
PREFIX_SIZE = 14
POSTFIX_SIZE = 7

# Load normalized fragments
X = np.load(
    "../data/normalized_fragments.npy"
)

metadata = np.load(
    "../data/fragment_metadata.npy",
    allow_pickle=True
)

print(f"Dataset shape: {X.shape}")

# Use all fragments except last for training
X_train = X[:-1]

# Last fragment for testing
test_fragment = X[-1]

# Metadata of test fragment
test_meta = metadata[-1]

xmin = test_meta["xmin"]
xmax = test_meta["xmax"]

# Split fragment
test_prefix = test_fragment[:PREFIX_SIZE]
real_postfix = test_fragment[PREFIX_SIZE:]

print(f"Train fragments: {len(X_train)}")
print("Training SOM...")

# Create SOM
som = MiniSom(
    x=6,
    y=5,
    input_len=21,
    sigma=1.5,
    learning_rate=0.5,
    random_seed=42
)

som.random_weights_init(X_train)

som.train_random(
    X_train,
    5000
)

print("Training completed")

# Search best matching cluster
best_distance = np.inf
best_pattern = None
best_cluster = None

weights = som.get_weights()

for row in range(6):

    for col in range(5):

        pattern = weights[row, col]

        pattern_prefix = pattern[:PREFIX_SIZE]

        distance = np.linalg.norm(
            test_prefix - pattern_prefix
        )

        if distance < best_distance:

            best_distance = distance
            best_pattern = pattern
            best_cluster = (row, col)

print()
print(f"Target cluster: {best_cluster}")
print(f"Distance: {best_distance:.4f}")

# Forecast in normalized space
forecast_postfix = best_pattern[PREFIX_SIZE:]

# Denormalization
real_forecast = (
    forecast_postfix * (xmax - xmin)
    + xmin
)

real_values = (
    real_postfix * (xmax - xmin)
    + xmin
)

# Metrics
mae = np.mean(
    np.abs(
        real_values - real_forecast
    )
)

rmse = np.sqrt(
    np.mean(
        (real_values - real_forecast) ** 2
    )
)

print()
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")

# ==========================
# Plot 1
# Prefix matching
# ==========================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

fig.patch.set_facecolor(
    BACKGROUND
)

ax.set_facecolor(
    BACKGROUND
)

ax.plot(
    test_prefix,
    color=PREFIX_COLOR,
    linewidth=3,
    label="Experimental Prefix"
)

ax.plot(
    best_pattern[:PREFIX_SIZE],
    color=PATTERN_COLOR,
    linewidth=3,
    label="Cluster Prefix"
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.4,
    color=GRID
)

ax.legend()

ax.tick_params(
    colors=TEXT
)

for spine in ax.spines.values():
    spine.set_color("#666666")

plt.tight_layout()

plt.savefig(
    "../plots/forecast/prefix_matching.png",
    dpi=300,
    facecolor=fig.get_facecolor()
)

plt.close()

# ==========================
# Plot 2
# Forecast vs Real
# ==========================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

fig.patch.set_facecolor(
    BACKGROUND
)

ax.set_facecolor(
    BACKGROUND
)

days = np.arange(POSTFIX_SIZE)

ax.plot(
    days,
    real_forecast,
    color=FORECAST_COLOR,
    linewidth=3,
    marker="o",
    label="Forecast"
)

ax.plot(
    days,
    real_values,
    color=REAL_COLOR,
    linewidth=3,
    marker="o",
    label="Real Values"
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.4,
    color=GRID
)

ax.legend()

ax.tick_params(
    colors=TEXT
)

for spine in ax.spines.values():
    spine.set_color("#666666")

plt.tight_layout()

plt.savefig(
    "../plots/forecast/forecast_vs_real.png",
    dpi=300,
    facecolor=fig.get_facecolor()
)

plt.close()

print()
print("Forecast plots saved.")