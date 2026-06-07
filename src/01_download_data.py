import yfinance as yf

# Can be changed to any other currency, stock, or commodity ticker supported by Yahoo Finance
ticker = "BZ=F"

df = yf.download(
    ticker,
    start="2010-01-01",
    end="2026-05-27",
    interval="1d",
    auto_adjust=True
)

if hasattr(df.columns, "levels"):
    df.columns = df.columns.get_level_values(0)

df.to_csv("../data/brent_oil_16y.csv")

print(df.head())
print(f"Saved {len(df)} rows")