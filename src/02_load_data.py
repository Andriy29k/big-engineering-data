import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/brent_oil_16y.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.set_index("Date")

fig, ax = plt.subplots(figsize=(16, 8))

fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")

ax.plot(
    df.index,
    df["Close"],
    color="#3399ff",
    linewidth=1.3,
    label="Brent Close Price"
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3,
    color="white"
)

ax.set_title(
    "Brent Oil Prices (2010-2026)",
    fontsize=18,
    color="white",
    pad=15
)

ax.set_xlabel("Date", fontsize=12, color="white")
ax.set_ylabel("Price (USD)", fontsize=12, color="white")

ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("#666666")

legend = ax.legend(
    facecolor="#222222",
    edgecolor="#444444",
    fontsize=11
)

for text in legend.get_texts():
    text.set_color("white")

plt.tight_layout()

plt.savefig(
    "../plots/raw_series.png",
    dpi=300,
    facecolor=fig.get_facecolor()
)

plt.show()