import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from minisom import MiniSom

# Create output directories
Path("../plots/clusters").mkdir(
    parents=True,
    exist_ok=True
)

BACKGROUND = "#111111"
GRID = "#444444"
TEXT = "#ffffff"

PATTERN_COLOR = "#33ccff"

# Load normalized fragments
X = np.load(
    "../data/normalized_fragments.npy"
)

print(f"Dataset shape: {X.shape}")

# Recreate and train SOM
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

som.random_weights_init(X)

som.train_random(
    X,
    5000
)

# Assign fragments to clusters
clusters = defaultdict(list)

for fragment_id, fragment in enumerate(X):

    winner = som.winner(fragment)

    clusters[winner].append(
        fragment_id
    )

print()
print("Cluster statistics:")
print("-" * 50)

for cluster_id, members in sorted(
    clusters.items()
):
    print(
        f"Cluster {cluster_id}: "
        f"{len(members)} fragments"
    )

# Save cluster assignments
cluster_info = {}

for cluster_id, members in clusters.items():

    cluster_info[str(cluster_id)] = members

np.save(
    "../models/cluster_assignments.npy",
    cluster_info
)

# Plot cluster patterns
weights = som.get_weights()

for row in range(som_rows):

    for col in range(som_cols):

        neuron_pattern = weights[row, col]

        cluster_size = len(
            clusters.get(
                (row, col),
                []
            )
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        fig.patch.set_facecolor(
            BACKGROUND
        )

        ax.set_facecolor(
            BACKGROUND
        )

        ax.plot(
            neuron_pattern,
            color=PATTERN_COLOR,
            linewidth=2
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4,
            color=GRID
        )

        ax.set_title(
            f"Cluster ({row},{col}) | "
            f"Members: {cluster_size}",
            color=TEXT
        )

        ax.set_xlabel(
            "Day",
            color=TEXT
        )

        ax.set_ylabel(
            "Normalized price",
            color=TEXT
        )

        ax.tick_params(
            colors=TEXT
        )

        for spine in ax.spines.values():
            spine.set_color(
                "#666666"
            )

        plt.tight_layout()

        plt.savefig(
            f"../plots/clusters/cluster_{row}_{col}.png",
            dpi=300,
            facecolor=fig.get_facecolor()
        )

        plt.close()

print()
print(
    f"Used clusters: "
    f"{len(clusters)} / "
    f"{som_rows * som_cols}"
)

cluster_sizes = []

cluster_names = []

for row in range(som_rows):
    for col in range(som_cols):

        size = len(
            clusters.get(
                (row, col),
                []
            )
        )

        cluster_sizes.append(size)
        cluster_names.append(f"{row},{col}")

fig, ax = plt.subplots(figsize=(14, 6))

fig.patch.set_facecolor(BACKGROUND)
ax.set_facecolor(BACKGROUND)

ax.bar(
    cluster_names,
    cluster_sizes
)

ax.set_title(
    "Distribution of Fragments Across Clusters",
    color=TEXT
)

ax.tick_params(colors=TEXT)

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "../plots/clusters_distribution.png",
    dpi=300,
    facecolor=fig.get_facecolor()
)