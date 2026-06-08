import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

Path("../plots/normalization_v2").mkdir(
    parents=True,
    exist_ok=True
)

BACKGROUND = "#111111"
GRID = "#444444"
TEXT = "#ffffff"

RAW_COLOR = "#ff9933"
NORMALIZED_COLOR = "#33cc66"

# load fragments
fragments = np.load(
    "../data/fragments.npy"
)

normalized_fragments = []
fragment_metadata = []

for fragment in fragments:

    xmax = np.max(fragment)

    if xmax == 0:
        normalized_fragment = np.zeros_like(fragment)
    else:
        normalized_fragment = fragment / xmax

    normalized_fragments.append(
        normalized_fragment
    )

    fragment_metadata.append(
        {
            "xmax": float(xmax)
        }
    )

normalized_fragments = np.array(
    normalized_fragments,
    dtype=np.float64
)

# save normalized dataset
np.save(
    "../data/normalized_fragments_v2.npy",
    normalized_fragments
)

# save denormalization parameters
np.save(
    "../data/fragment_metadata_v2.npy",
    fragment_metadata
)

print(
    f"Normalized dataset shape: "
    f"{normalized_fragments.shape}"
)

fragment_ids = [
    0,
    50,
    100,
    200
]

fragment_ids = [
    idx
    for idx in fragment_ids
    if idx < len(normalized_fragments)
]

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

    axes[0].plot(
        fragments[fragment_id],
        color=RAW_COLOR,
        linewidth=2
    )

    axes[0].set_title(
        f"Original Fragment {fragment_id}",
        color=TEXT
    )

    axes[1].plot(
        normalized_fragments[
            fragment_id
        ],
        color=NORMALIZED_COLOR,
        linewidth=2
    )

    axes[1].set_ylim(0, 1)

    axes[1].set_title(
        "Zero-Max Normalization",
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
        f"../plots/normalization_v2/fragment_{fragment_id}.png",
        dpi=300,
        facecolor=fig.get_facecolor()
    )

    plt.close()

print()
print(
    "Normalization V2 completed."
)
print(
    "Saved to normalized_fragments_v2.npy"
)