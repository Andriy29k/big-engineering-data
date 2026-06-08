# SOM Time Series Forecasting

Self-Organizing Map (Kohonen) based forecasting for financial time series.  
The model learns typical 20-day price patterns, clusters them, and uses the closest pattern prefix to forecast the next 5 trading days.

---

## Project Structure

```
project/
├── data/                          # .gitignore: *.npy (generated)
│   ├── brent_oil_16y.csv          # raw dataset (downloaded once)
│   ├── fragments.npy              # sliding window fragments
│   ├── fragment_dates.npy
│   ├── normalized_fragments_v1.npy
│   ├── normalized_fragments_v2.npy
│   ├── fragment_metadata_v1.npy   # xmin, xmax per fragment
│   └── fragment_metadata_v2.npy   # xmax per fragment
├── models/
│   ├── som_weights_v1.npy
│   ├── som_weights_v2.npy
│   ├── cluster_assignments_v1.npy
│   └── cluster_assignments_v2.npy
├── plots/                         # .gitignore: *.png (generated)
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_load_data.py
│   ├── 03_create_fragments.py
│   ├── 04_normalize_fragments_v1.py
│   ├── 04_normalize_fragments_v2.py
│   ├── 05_train_som_v1.py
│   ├── 05_train_som_v2.py
│   ├── 07_cluster_analysis_v1.py
│   ├── 07_cluster_analysis_v2.py
│   └── 08_forecast_compare.py
├── .gitignore
└── README.md
```

---

## .gitignore

```gitignore
*.png
*.npy
```

Generated files (weights, fragments, plots) are excluded — they are reproducible by running the scripts in order.

---

## Tunable Parameters

### 1. Data Source — `01_download_data.py`

```python
ticker = "BZ=F"          # Brent Crude Oil futures
start  = "2010-01-01"    # dataset start date
end    = "2026-06-07"    # dataset end date
interval = "1d"          # candle interval
```

**`ticker`** — any symbol supported by Yahoo Finance:

| Ticker | Asset |
|--------|-------|
| `BZ=F` | Brent Crude Oil |
| `CL=F` | WTI Crude Oil |
| `GC=F` | Gold |
| `EURUSD=X` | EUR/USD |
| `BTC-USD` | Bitcoin |
| `AAPL` | Apple stock |

**`start` / `end`** — longer history gives more fragments → richer SOM.  
Minimum recommended: 3–4 years (≥700 trading days).

**`interval`** — `"1d"` (daily), `"1wk"` (weekly), `"1h"` (hourly, limited history).  
Changing interval changes the meaning of WINDOW_SIZE: with `"1wk"`, a window of 20 = 20 weeks ≈ 5 months.

---

### 2. Fragmentation — `03_create_fragments.py`

```python
WINDOW_SIZE = 20   # fragment length in trading days
STEP = 5           # sliding step in trading days
```

**`WINDOW_SIZE`** — total length of each fragment (prefix + postfix).  
Defines how much history the model "sees" and how far ahead it forecasts.

| WINDOW_SIZE | PREFIX_SIZE | Forecast horizon |
|-------------|-------------|-----------------|
| 20 | 15 | 5 days (~1 week) |
| 30 | 20 | 10 days (~2 weeks) |
| 40 | 30 | 10 days |

> Must match `input_len` in MiniSom and `PREFIX_SIZE + POSTFIX_SIZE` in `08_forecast_compare.py`.

**`STEP`** — sliding window step. Smaller step → more fragments → slower training but richer dataset.

| STEP | ~Fragments (16y data) |
|------|-----------------------|
| 1 | ~4000 |
| 5 | ~630 (current) |
| 10 | ~320 |

---

### 3. Forecast Split — `08_forecast_compare.py`

```python
PREFIX_SIZE  = 15   # known days used for pattern matching
POSTFIX_SIZE = 5    # days to forecast (quality control zone)
```

**`PREFIX_SIZE`** — how many known days are used to find the closest cluster.  
More = more context for matching, but less forecasted.

**`POSTFIX_SIZE`** — how many days ahead to forecast.  
Must satisfy: `PREFIX_SIZE + POSTFIX_SIZE == WINDOW_SIZE`.

---

### 4. SOM Architecture — `05_train_som_v1.py` / `v2`

```python
som_rows = 6
som_cols = 5

som = MiniSom(
    x=som_rows,
    y=som_cols,
    input_len=20,
    sigma=1.5,
    learning_rate=0.5,
    random_seed=42
)

som.train_random(X, 5000)
```

**`som_rows × som_cols`** — map size = number of clusters.

| Map size | Clusters | Use case |
|----------|----------|----------|
| 4×4 | 16 | small dataset, coarse grouping |
| 6×5 | 30 | current — balanced |
| 8×6 | 48 | richer dataset, finer patterns |
| 10×7 | 70 | large dataset only |

Rule of thumb: `rows × cols ≈ 5 × sqrt(N)`, where N = number of fragments.  
With ~630 fragments: `5 × sqrt(630) ≈ 125` — so 30 clusters is actually on the conservative side.

**`sigma`** — initial neighborhood radius. Defines how many neighboring neurons are pulled toward the winner during training.

| sigma | Effect |
|-------|--------|
| 0.5–1.0 | Narrow — neurons specialize quickly, less smooth map |
| 1.5 | Current — balanced |
| 2.0–3.0 | Wide — smoother map, neighbors strongly correlated |

**`learning_rate`** — how strongly weights move toward each input vector.

| learning_rate | Effect |
|---------------|--------|
| 0.1–0.3 | Slow, stable convergence |
| 0.5 | Current — standard |
| 0.7–1.0 | Fast but may overshoot |

**`random_seed`** — fixes randomness for reproducibility. Change to any integer to get a different (but deterministic) map.

**`train_random(X, N)`** — number of training iterations.

| Iterations | Effect |
|------------|--------|
| 1000 | Fast, rough map |
| 5000 | Current |
| 10000 | Better convergence, especially for large maps |
| 50000 | Diminishing returns unless map is large |

---

## Recommended Configurations to Test

### Config A — Current baseline
```python
# Fragmentation
WINDOW_SIZE = 20
STEP = 5
PREFIX_SIZE = 15

# SOM
som_rows, som_cols = 6, 5
sigma = 1.5
learning_rate = 0.5
iterations = 5000
```
**Result:** V2 MAPE = 3.60%

---

### Config B — Larger map + more iterations ⭐ try this first
```python
WINDOW_SIZE = 20
STEP = 5
PREFIX_SIZE = 15

som_rows, som_cols = 10, 7   # 70 clusters
sigma = 1.5
learning_rate = 0.5
iterations = 10000
```
More clusters → finer pattern separation → potentially better prefix match.

---

### Config C — Longer context window
```python
WINDOW_SIZE = 30     # update in 03, 05, 08
STEP = 5
PREFIX_SIZE = 20     # 4 trading weeks of context
# POSTFIX_SIZE = 10  # 2 weeks forecast

som_rows, som_cols = 8, 6
sigma = 1.5
learning_rate = 0.5
iterations = 10000
```
More context for matching but forecasts 10 days — harder task, expect higher MAPE.

---

### Config D — Denser fragments
```python
WINDOW_SIZE = 20
STEP = 2             # was 5 — 2.5x more fragments
PREFIX_SIZE = 15

som_rows, som_cols = 10, 8   # more clusters for more data
sigma = 2.0
learning_rate = 0.5
iterations = 15000
```
More training data, smoother map. Slower to run.

---

### Config E — Conservative, small map
```python
WINDOW_SIZE = 20
STEP = 5
PREFIX_SIZE = 15

som_rows, som_cols = 4, 4   # 16 clusters — very coarse
sigma = 1.0
learning_rate = 0.3
iterations = 3000
```
Useful as a lower bound to confirm that more clusters actually help.

---

## Pipeline Execution Order

```bash
python 01_download_data.py        # download CSV (once)
python 02_load_data.py            # visualize raw series
python 03_create_fragments.py     # create sliding windows
python 04_normalize_fragments_v1.py
python 04_normalize_fragments_v2.py
python 05_train_som_v1.py         # train SOM on V1
python 05_train_som_v2.py         # train SOM on V2
python 07_cluster_analysis_v1.py  # analyze clusters
python 07_cluster_analysis_v2.py
python 08_forecast_compare.py     # forecast + metrics + plots
```

> When changing `WINDOW_SIZE` or `STEP`, re-run everything from `03` onward.  
> When changing only SOM parameters, re-run from `05` onward.

---

## Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| MAE | `mean(\|real - forecast\|)` | Average error in USD |
| RMSE | `sqrt(mean((real - forecast)²))` | Penalizes large errors more |
| MAPE | `mean(\|real - forecast\| / real) × 100%` | Scale-independent, % error |
| Distance | `‖prefix_test - prefix_pattern‖₂` | Quality of pattern match (normalized space) |

MAPE < 5% is generally considered good for financial forecasting.  
RMSE > MAE always — the gap between them shows how "spiky" the errors are across the 5 forecast days.