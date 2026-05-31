import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# create output directories if they not exist
Path("../plots/fragments").mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 21
STEP = 7

BACKGROUND = "#111111"
GRID = "#444444"
LINE = "#3399ff"
TEXT = "#ffffff"

df = pd.read_csv("../data/brent_oil_16y.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
close_prices = df["Close"].values
fragments = []

# create overlapping 21 day fragments with 7 day step
for start in range(
    0,
    len(close_prices) - WINDOW_SIZE + 1,
    STEP
):
    fragment = close_prices[start:start + WINDOW_SIZE]
    fragments.append(fragment)

fragments = np.array(fragments)

print(f"Total fragments: {len(fragments)}")
print(f"Fragment shape: {fragments.shape}")

np.save("../data/fragments.npy", fragments)

# several fragments for visualization
fragment_ids = [0, 50, 100, 200]

# remove invalid indexes if dataset is smaller
fragment_ids = [
    idx
    for idx in fragment_ids
    if idx < len(fragments)
]

for fragment_id in fragment_ids:

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    ax.plot(
        fragments[fragment_id],
        color=LINE,
        linewidth=2
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
        color=GRID
    )

    ax.set_title(
        f"Fragment {fragment_id}",
        color=TEXT,
        fontsize=14
    )

    ax.set_xlabel(
        "Day inside fragment",
        color=TEXT
    )

    ax.set_ylabel(
        "Close price (USD)",
        color=TEXT
    )

    ax.tick_params(colors=TEXT)

    for spine in ax.spines.values():
        spine.set_color("#666666")

    plt.tight_layout()

    plt.savefig(
        f"../plots/fragments/fragment_{fragment_id}.png",
        dpi=300,
        facecolor=fig.get_facecolor()
    )

    plt.close()

print("Fragments saved successfully.")