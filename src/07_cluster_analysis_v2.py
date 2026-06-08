import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from minisom import MiniSom

Path("../plots/clusters_v2").mkdir(
    parents=True,
    exist_ok=True
)

BACKGROUND = "#111111"
GRID = "#444444"
TEXT = "#ffffff"

PATTERN_COLOR = "#33ccff"

X = np.load(
    "../data/normalized_fragments_v2.npy"
)

weights = np.load(
    "../models/som_weights_v2.npy"
)

som_rows = weights.shape[0]
som_cols = weights.shape[1]

som = MiniSom(
    x=som_rows,
    y=som_cols,
    input_len=20,
    sigma=1.5,
    learning_rate=0.5
)

som._weights = weights

clusters = defaultdict(list)

for fragment_id, fragment in enumerate(X):

    winner = som.winner(fragment)

    clusters[winner].append(
        fragment_id
    )

print()
print("Cluster statistics")
print("-" * 50)

for cluster_id, members in sorted(
    clusters.items()
):
    print(
        f"{cluster_id}: "
        f"{len(members)} fragments"
    )

cluster_info = {}

for cluster_id, members in clusters.items():

    cluster_info[
        str(cluster_id)
    ] = members

np.save(
    "../models/cluster_assignments_v2.npy",
    cluster_info
)

for row in range(som_rows):

    for col in range(som_cols):

        pattern = weights[row, col]

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
            pattern,
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
            f"Cluster ({row},{col}) | Members: {cluster_size}",
            color=TEXT
        )

        ax.set_xlabel(
            "Trading Day",
            color=TEXT
        )

        ax.set_ylabel(
            "Normalized Price",
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
            f"../plots/clusters_v2/cluster_{row}_{col}.png",
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