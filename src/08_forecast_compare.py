import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from minisom import MiniSom

Path("../plots/forecast").mkdir(parents=True, exist_ok=True)

BG         = "#0d0d0d"
PANEL      = "#141414"
GRID_COL   = "#2a2a2a"
TEXT       = "#e0e0e0"
MUTED      = "#666666"

PREFIX_C   = "#f0a500"   # gold  — перші 15 днів (відомі)
PATTERN_C  = "#aaaaaa"   # grey  — знайдений патерн (prefix частина)
V1_C       = "#33cc66"   # green — прогноз V1
V2_C       = "#3399ff"   # blue  — прогноз V2
REAL_C     = "#ff4444"   # red   — реальні значення

WINDOW_SIZE  = 20
PREFIX_SIZE  = 15
POSTFIX_SIZE = 5

prefix_days   = np.arange(1, PREFIX_SIZE + 1)
forecast_days = np.arange(PREFIX_SIZE + 1, WINDOW_SIZE + 1)
all_days      = np.arange(1, WINDOW_SIZE + 1)


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.grid(True, linestyle="--", alpha=0.25, color=GRID_COL)
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#333333")


def divider(ax):
    """Вертикальна лінія між prefix і forecast зоною."""
    ax.axvline(x=PREFIX_SIZE, color="#555555",
               linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvspan(PREFIX_SIZE, WINDOW_SIZE,
               alpha=0.05, color="#ffffff")


# Завантаження і пошук патерну — V1
X1        = np.load("../data/normalized_fragments_v1.npy")
meta1     = np.load("../data/fragment_metadata_v1.npy", allow_pickle=True)
weights1  = np.load("../models/som_weights_v1.npy")

som1 = MiniSom(x=weights1.shape[0], y=weights1.shape[1],
               input_len=WINDOW_SIZE, sigma=1.5, learning_rate=0.5)
som1._weights = weights1

test1         = X1[-1]
prefix1       = test1[:PREFIX_SIZE]
real_post1_n  = test1[PREFIX_SIZE:]
xmin1         = float(meta1[-1]["xmin"])
xmax1         = float(meta1[-1]["xmax"])

best_dist1, best_pat1, best_cl1 = np.inf, None, None
for row in range(weights1.shape[0]):
    for col in range(weights1.shape[1]):
        pat = weights1[row, col]
        d   = np.linalg.norm(prefix1 - pat[:PREFIX_SIZE])
        if d < best_dist1:
            best_dist1, best_pat1, best_cl1 = d, pat, (row, col)

# Денормалізація V1: x = norm * (xmax - xmin) + xmin
rng1          = xmax1 - xmin1
prefix1_real  = prefix1          * rng1 + xmin1
forecast1_real= best_pat1[PREFIX_SIZE:] * rng1 + xmin1
real1_real    = real_post1_n     * rng1 + xmin1
pattern1_real = best_pat1        * rng1 + xmin1

mae1  = np.mean(np.abs(real1_real - forecast1_real))
rmse1 = np.sqrt(np.mean((real1_real - forecast1_real) ** 2))
mape1 = np.mean(np.abs((real1_real - forecast1_real) / real1_real)) * 100

# Завантаження і пошук патерну — V2
X2        = np.load("../data/normalized_fragments_v2.npy")
meta2     = np.load("../data/fragment_metadata_v2.npy", allow_pickle=True)
weights2  = np.load("../models/som_weights_v2.npy")

som2 = MiniSom(x=weights2.shape[0], y=weights2.shape[1],
               input_len=WINDOW_SIZE, sigma=1.5, learning_rate=0.5)
som2._weights = weights2

test2         = X2[-1]
prefix2       = test2[:PREFIX_SIZE]
real_post2_n  = test2[PREFIX_SIZE:]
xmax2         = float(meta2[-1]["xmax"])

best_dist2, best_pat2, best_cl2 = np.inf, None, None
for row in range(weights2.shape[0]):
    for col in range(weights2.shape[1]):
        pat = weights2[row, col]
        d   = np.linalg.norm(prefix2 - pat[:PREFIX_SIZE])
        if d < best_dist2:
            best_dist2, best_pat2, best_cl2 = d, pat, (row, col)

# Денормалізація V2 (x = norm * xmax)
prefix2_real  = prefix2               * xmax2
forecast2_real= best_pat2[PREFIX_SIZE:] * xmax2
real2_real    = real_post2_n           * xmax2
pattern2_real = best_pat2              * xmax2

mae2  = np.mean(np.abs(real2_real - forecast2_real))
rmse2 = np.sqrt(np.mean((real2_real - forecast2_real) ** 2))
mape2 = np.mean(np.abs((real2_real - forecast2_real) / real2_real)) * 100

print()
print("═" * 42)
print("  V1  Min-Max Normalization")
print("═" * 42)
print(f"  Cluster  : {best_cl1}")
print(f"  Distance : {best_dist1:.6f}")
print(f"  MAE      : {mae1:.4f} USD")
print(f"  RMSE     : {rmse1:.4f} USD")
print(f"  MAPE     : {mape1:.2f} %")
print()
print("═" * 42)
print("  V2  Zero-Max Normalization")
print("═" * 42)
print(f"  Cluster  : {best_cl2}")
print(f"  Distance : {best_dist2:.6f}")
print(f"  MAE      : {mae2:.4f} USD")
print(f"  RMSE     : {rmse2:.4f} USD")
print(f"  MAPE     : {mape2:.2f} %")
print("═" * 42)


# PLOT 1 — Full picture V1
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG)
style_ax(ax,
         title=f"V1 (Min-Max) — Prefix → Pattern → Forecast vs Real\n"
               f"Cluster {best_cl1}  |  MAE={mae1:.2f}  RMSE={rmse1:.2f}  MAPE={mape1:.1f}%",
         xlabel="Trading Day", ylabel="Brent Oil Price (USD)")

divider(ax)

# Місток: остання точка prefix з'єднує prefix з forecast/real
bridge1      = prefix1_real[-1]
days_bridge  = np.concatenate([[PREFIX_SIZE], forecast_days])
fore1_bridge = np.concatenate([[bridge1], forecast1_real])
real1_bridge = np.concatenate([[bridge1], real1_real])

ax.plot(prefix_days,  prefix1_real,               color=PREFIX_C,  lw=2.5, marker="o", ms=5, label="Prefix (15 днів)")
ax.plot(prefix_days,  pattern1_real[:PREFIX_SIZE], color=PATTERN_C, lw=1.5,
        ls=":", alpha=0.6, label=f"SOM pattern {best_cl1} (prefix)")
ax.plot(days_bridge,  fore1_bridge,                color=V1_C,     lw=2.5, marker="o", ms=7, label="Прогноз V1",
        markevery=list(range(1, len(days_bridge))))
ax.plot(days_bridge,  real1_bridge,                color=REAL_C,   lw=2.5, marker="s", ms=7, label="Реальні значення",
        markevery=list(range(1, len(days_bridge))))

ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT, fontsize=10)
plt.tight_layout()
plt.savefig("../plots/forecast/v1_full_picture.png", dpi=300, facecolor=BG)
plt.close()
print("Saved: v1_full_picture.png")


# Full picture V2
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG)
style_ax(ax,
         title=f"V2 (Zero-Max) — Prefix → Pattern → Forecast vs Real\n"
               f"Cluster {best_cl2}  |  MAE={mae2:.2f}  RMSE={rmse2:.2f}  MAPE={mape2:.1f}%",
         xlabel="Trading Day", ylabel="Brent Oil Price (USD)")

divider(ax)

# Місток: остання точка prefix з'єднує prefix з forecast/real
bridge2      = prefix2_real[-1]
fore2_bridge = np.concatenate([[bridge2], forecast2_real])
real2_bridge = np.concatenate([[bridge2], real2_real])

ax.plot(prefix_days,  prefix2_real,               color=PREFIX_C,  lw=2.5, marker="o", ms=5, label="Prefix (15 днів)")
ax.plot(prefix_days,  pattern2_real[:PREFIX_SIZE], color=PATTERN_C, lw=1.5,
        ls=":", alpha=0.6, label=f"SOM pattern {best_cl2} (prefix)")
ax.plot(days_bridge,  fore2_bridge,                color=V2_C,     lw=2.5, marker="o", ms=7, label="Прогноз V2",
        markevery=list(range(1, len(days_bridge))))
ax.plot(days_bridge,  real2_bridge,                color=REAL_C,   lw=2.5, marker="s", ms=7, label="Реальні значення",
        markevery=list(range(1, len(days_bridge))))

ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT, fontsize=10)
plt.tight_layout()
plt.savefig("../plots/forecast/v2_full_picture.png", dpi=300, facecolor=BG)
plt.close()
print("Saved: v2_full_picture.png")


# Zoom: forecast zone, V1 vs V2 vs Real
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
style_ax(ax,
         title="Forecast Zone (Days 16–20) — V1 vs V2 vs Real",
         xlabel="", ylabel="Brent Oil Price (USD)")

ax.plot(forecast_days, forecast1_real, color=V1_C,   lw=2.5, marker="o", ms=9, label="Прогноз V1 (Min-Max)")
ax.plot(forecast_days, forecast2_real, color=V2_C,   lw=2.5, marker="D", ms=9, label="Прогноз V2 (Zero-Max)")
ax.plot(forecast_days, real1_real,     color=REAL_C, lw=2.5, marker="s", ms=9, label="Реальні значення")

ax.fill_between(forecast_days, forecast1_real, real1_real, alpha=0.10, color=V1_C)
ax.fill_between(forecast_days, forecast2_real, real2_real, alpha=0.10, color=V2_C)

for d, f1, f2, r in zip(forecast_days, forecast1_real, forecast2_real, real1_real):
    ax.annotate(f"{f1:.1f}", (d, f1), textcoords="offset points",
                xytext=(-18, 4), color=V1_C, fontsize=8)
    ax.annotate(f"{f2:.1f}", (d, f2), textcoords="offset points",
                xytext=(6,  4), color=V2_C, fontsize=8)
    ax.annotate(f"{r:.1f}",  (d, r),  textcoords="offset points",
                xytext=(0, -16), color=REAL_C, fontsize=8, ha="center")

ax.set_xticks(forecast_days)
ax.set_xticklabels([f"Day {d}" for d in forecast_days], color=MUTED)
ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT, fontsize=10)
plt.tight_layout()
plt.savefig("../plots/forecast/forecast_zoom_comparison.png", dpi=300, facecolor=BG)
plt.close()
print("Saved: forecast_zoom_comparison.png")


# Daily absolute error: V1 vs V2
err1 = np.abs(real1_real - forecast1_real)
err2 = np.abs(real2_real - forecast2_real)
day_labels = [f"Day {d}" for d in forecast_days]
x = np.arange(len(forecast_days))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
style_ax(ax,
         title="Absolute Error per Day — V1 vs V2",
         ylabel="Absolute Error (USD)")

bars1 = ax.bar(x - w/2, err1, width=w, color=V1_C, alpha=0.85, label=f"V1  MAE={mae1:.2f}")
bars2 = ax.bar(x + w/2, err2, width=w, color=V2_C, alpha=0.85, label=f"V2  MAE={mae2:.2f}")

for bar, v in zip(bars1, err1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v:.2f}", ha="center", va="bottom", color=V1_C, fontsize=9)
for bar, v in zip(bars2, err2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v:.2f}", ha="center", va="bottom", color=V2_C, fontsize=9)

ax.axhline(mae1, color=V1_C,   lw=1, ls="--", alpha=0.5)
ax.axhline(mae2, color=V2_C,   lw=1, ls="--", alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(day_labels, color=MUTED)
ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT, fontsize=10)
plt.tight_layout()
plt.savefig("../plots/forecast/daily_error_v1_v2.png", dpi=300, facecolor=BG)
plt.close()
print("Saved: daily_error_v1_v2.png")


# Pattern match: нормалізований простір V1 і V2 поруч
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("SOM Pattern Match — Normalized Space", color=TEXT, fontsize=13)

# V1
style_ax(axes[0], title=f"V1 Min-Max  |  Cluster {best_cl1}  |  dist={best_dist1:.4f}",
         xlabel="Trading Day", ylabel="Normalized Price")
axes[0].plot(prefix_days, prefix1, color=PREFIX_C, lw=2.5, marker="o", ms=4, label="Test prefix")
axes[0].plot(all_days,    best_pat1, color=V1_C,   lw=2,   ls="--",            label=f"SOM pattern")
axes[0].axvline(x=PREFIX_SIZE, color="#555555", lw=1, ls=":")
axes[0].axvspan(PREFIX_SIZE, WINDOW_SIZE, alpha=0.05, color="#ffffff")
axes[0].legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT, fontsize=9)

# V2
style_ax(axes[1], title=f"V2 Zero-Max  |  Cluster {best_cl2}  |  dist={best_dist2:.4f}",
         xlabel="Trading Day", ylabel="Normalized Price")
axes[1].plot(prefix_days, prefix2, color=PREFIX_C, lw=2.5, marker="o", ms=4, label="Test prefix")
axes[1].plot(all_days,    best_pat2, color=V2_C,   lw=2,   ls="--",            label=f"SOM pattern")
axes[1].axvline(x=PREFIX_SIZE, color="#555555", lw=1, ls=":")
axes[1].axvspan(PREFIX_SIZE, WINDOW_SIZE, alpha=0.05, color="#ffffff")
axes[1].legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT, fontsize=9)

plt.tight_layout()
plt.savefig("../plots/forecast/pattern_match_v1_v2.png", dpi=300, facecolor=BG)
plt.close()
print("Saved: pattern_match_v1_v2.png")


print()
print("All plots saved to ../plots/forecast/")