import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# 4 trading weeks
WINDOW_SIZE = 20

# shift exactly 1 trading week
STEP = 5

BACKGROUND = "#111111"
GRID = "#444444"
LINE = "#3399ff"
TEXT = "#ffffff"

# create output directory
Path("../plots/fragments").mkdir(
    parents=True,
    exist_ok=True
)

# load Brent dataset
df = pd.read_csv(
    "../data/brent_oil_16y.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values(
    "Date"
).reset_index(drop=True)

# keep only weekdays
df = df[
    df["Date"].dt.dayofweek < 5
].reset_index(drop=True)

close_prices = df["Close"].values
dates = df["Date"].values

print(
    f"Trading days in dataset: "
    f"{len(close_prices)}"
)

fragments = []
fragment_dates = []

# create overlapping fragments
for start in range(
    0,
    len(close_prices) - WINDOW_SIZE + 1,
    STEP
):

    fragment = close_prices[
        start:start + WINDOW_SIZE
    ]

    fragments.append(
        fragment
    )

    fragment_dates.append(
        dates[start]
    )

fragments = np.array(
    fragments,
    dtype=np.float64
)

fragment_dates = np.array(
    fragment_dates
)

print(
    f"Total fragments: "
    f"{len(fragments)}"
)

print(
    f"Fragment shape: "
    f"{fragments.shape}"
)

# save fragments
np.save(
    "../data/fragments.npy",
    fragments
)

# save start dates
np.save(
    "../data/fragment_dates.npy",
    fragment_dates
)

# fragments for visualization
fragment_ids = [
    0,
    50,
    100,
    200,
    300
]

fragment_ids = [
    idx
    for idx in fragment_ids
    if idx < len(fragments)
]

for fragment_id in fragment_ids:

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
        range(1, WINDOW_SIZE + 1),
        fragments[fragment_id],
        color=LINE,
        linewidth=2,
        marker="o",
        markersize=3
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35,
        color=GRID
    )

    ax.set_title(
        f"Fragment {fragment_id}\n"
        f"Start: {str(fragment_dates[fragment_id])[:10]}",
        color=TEXT,
        fontsize=14
    )

    ax.set_xlabel(
        "Trading day inside fragment",
        color=TEXT
    )

    ax.set_ylabel(
        "Close price (USD)",
        color=TEXT
    )

    ax.tick_params(
        colors=TEXT
    )

    for spine in ax.spines.values():
        spine.set_color(
            "#555555"
        )

    plt.tight_layout()

    plt.savefig(
        f"../plots/fragments/fragment_{fragment_id}.png",
        dpi=300,
        facecolor=fig.get_facecolor()
    )

    plt.close()

print()
print(
    "Fragments saved to "
    "../data/fragments.npy"
)

print(
    "Dates saved to "
    "../data/fragment_dates.npy"
)