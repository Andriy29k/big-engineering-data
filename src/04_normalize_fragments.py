import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Create output directories
Path("../plots/normalized").mkdir(parents=True, exist_ok=True)

BACKGROUND = "#111111"
GRID = "#444444"
RAW_LINE = "#ff9933"
NORMALIZED_LINE = "#33ccff"
TEXT = "#ffffff"

# Load fragments created in previous step
fragments = np.load("../data/fragments.npy")

normalized_fragments = []
fragment_metadata = []

# Normalize each fragment independently
for fragment in fragments:

    xmin = np.min(fragment)
    xmax = np.max(fragment)

    # Protection against division by zero
    if xmax == xmin:
        normalized_fragment = np.zeros_like(fragment)
    else:
        normalized_fragment = (
            (fragment - xmin)
            / (xmax - xmin)
        )

    normalized_fragments.append(
        normalized_fragment
    )

    fragment_metadata.append(
        {
            "xmin": float(xmin),
            "xmax": float(xmax)
        }
    )

normalized_fragments = np.array(
    normalized_fragments
)

# Save normalized data
np.save(
    "../data/normalized_fragments.npy",
    normalized_fragments
)

# Save metadata
np.save(
    "../data/fragment_metadata.npy",
    fragment_metadata
)

print(
    f"Normalized fragments shape: "
    f"{normalized_fragments.shape}"
)

# Fragments for visualization
fragment_ids = [0, 50, 100, 200]

fragment_ids = [
    idx
    for idx in fragment_ids
    if idx < len(normalized_fragments)
]

# Plot raw vs normalized
for fragment_id in fragment_ids:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    fig.patch.set_facecolor(
        BACKGROUND
    )

    for ax in axes:
        ax.set_facecolor(
            BACKGROUND
        )

    # Original fragment
    axes[0].plot(
        fragments[fragment_id],
        color=RAW_LINE,
        linewidth=2
    )

    axes[0].set_title(
        f"Original Fragment {fragment_id}",
        color=TEXT
    )

    # Normalized fragment
    axes[1].plot(
        normalized_fragments[fragment_id],
        color=NORMALIZED_LINE,
        linewidth=2
    )

    axes[1].set_title(
        f"Normalized Fragment {fragment_id}",
        color=TEXT
    )

    for ax in axes:

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4,
            color=GRID
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
        f"../plots/normalized/fragment_{fragment_id}_comparison.png",
        dpi=300,
        facecolor=fig.get_facecolor()
    )

    plt.close()

print("Normalization completed.")