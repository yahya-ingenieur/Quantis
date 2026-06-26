"""
src/data.py — Data acquisition, cleaning, feature engineering, and tagging.

Pipeline:
    download (yfinance) -> save raw -> clean -> add features/tags -> combine -> save processed

Outputs:
    data/raw/{TICKER}.csv     one untouched OHLCV download per ticker (audit trail)
    data/processed/dataset.csv  one combined, cleaned, long-format table for pooled training

Design notes:
    - Windowing / sequence "slicing" is intentionally NOT done here; window size is a
      swept hyperparameter, so it lives in train.py. This module emits a flat tidy table.
    - We keep auto_adjust=False so both `Close` and `Adj Close` are available; all
      returns are computed from `Adj Close` (dividend/split adjusted).
    - Cleaning DROPS rows with a missing Adj Close rather than forward-filling. Filling
      a non-traded day and then taking a log-return off the filled price would invent a
      fake zero-return, corrupting the next-day target.

Run standalone to (re)build the data artifacts:
    python src/data.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The traded basket (pooled training across all of these).
TICKERS = ["AAPL", "META", "MSFT", "TSLA", "AMZN", "XOM", "JNJ", "JPM"]

# Shariah-compliance tagging.
#   AAPL & META  -> shariah     (per current DJIM US factsheet)
#   JPM          -> conventional (interest-based banking)
#   everything else -> unclassified (not asserting a status we haven't verified)
SHARIAH_LABELS = {
    "AAPL": "shariah",
    "META": "shariah",
    "JPM": "conventional",
    "MSFT": "unclassified",
    "TSLA": "unclassified",
    "AMZN": "unclassified",
    "XOM": "unclassified",
    "JNJ": "unclassified",
}

# How much history to request. yfinance returns max available if a stock is
# younger than this (e.g. META ~2012, TSLA ~2010), which is fine for pooling.
PERIOD = "10y"

# Stable integer id per ticker so the pooled model can tell stocks apart
# (cheap feature explicitly called for in the blueprint).
TICKER_IDS = {ticker: idx for idx, ticker in enumerate(TICKERS)}

# Paths resolved relative to the project root (parent of src/), so the module
# behaves the same whether run as `python src/data.py` or imported elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Canonical column order we standardise every download to.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_ticker(ticker: str, period: str = PERIOD) -> pd.DataFrame:
    """Download daily OHLCV for one ticker via yfinance.

    Returns a DataFrame with a `Date` column plus the OHLCV columns. We keep
    auto_adjust=False so `Adj Close` (used for returns) and raw `Close` are both
    present. Raises ValueError if yfinance returns nothing.
    """
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}).")

    # yfinance can return a MultiIndex on the columns when given a single ticker;
    # flatten it down to the plain OHLCV field names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()  # move the DatetimeIndex into a `Date` column

    # Keep only the columns we expect, in a known order (defensive against
    # yfinance field reordering / extras).
    keep = ["Date"] + [c for c in OHLCV_COLUMNS if c in df.columns]
    df = df[keep]

    return df


def save_raw(df: pd.DataFrame, ticker: str) -> Path:
    """Write an untouched per-ticker download to data/raw/{TICKER}.csv.

    This is the audit trail — preserved exactly as downloaded, before any
    cleaning or feature engineering.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.csv"
    df.to_csv(path, index=False)
    return path


def download_all(tickers: list[str] = TICKERS) -> dict[str, pd.DataFrame]:
    """Download + save raw OHLCV for every ticker in the basket.

    Skips (with a warning) any symbol that fails to download so one bad ticker
    doesn't abort the whole run. Returns {ticker: raw DataFrame}.
    """
    raw: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = download_ticker(ticker)
        except Exception as exc:  # network / delisted / empty response
            print(f"  [warn] skipping {ticker}: {exc}")
            continue
        save_raw(df, ticker)
        raw[ticker] = df
        print(f"  [ok]   {ticker}: {len(df)} rows "
              f"({df['Date'].min().date()} -> {df['Date'].max().date()})")
    return raw


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative per-ticker cleaning, with no look-ahead leakage.

    Steps:
      - ensure `Date` is datetime and sort ascending
      - drop duplicate dates (keep first)
      - coerce OHLCV columns to numeric
      - DROP rows with a missing Adj Close (NOT forward-filled — filling a
        non-traded day would invent a fake zero log-return on the next step)

    Returns a cleaned copy with a fresh integer index.
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="first")

    for col in OHLCV_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Returns are built from Adj Close, so a missing Adj Close makes the row
    # unusable. Drop it rather than fill it.
    df = df.dropna(subset=["Adj Close"])

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature engineering + tagging
# --------------------------------------------------------------------------- #

def add_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add the modelling columns and tags for one (already cleaned) ticker.

    Adds:
      - log_return     : log(Adj Close_t / Adj Close_{t-1})  — the base series
      - target         : next-day log return (log_return.shift(-1)) — the label
      - ticker         : symbol string
      - ticker_id      : stable integer id for pooled training
      - shariah_status : shariah / conventional / unclassified

    Drops the first row (no prior price -> NaN return) and the last row
    (no next day -> NaN target), so every returned row is fully supervised.
    """
    df = df.copy()

    df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["target"] = df["log_return"].shift(-1)  # next-day log return

    df["ticker"] = ticker
    df["ticker_id"] = TICKER_IDS[ticker]
    df["shariah_status"] = SHARIAH_LABELS.get(ticker, "unclassified")

    # Remove rows that can't be fully supervised (head: NaN return, tail: NaN target).
    df = df.dropna(subset=["log_return", "target"]).reset_index(drop=True)

    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_dataset(tickers: list[str] = TICKERS) -> pd.DataFrame:
    """Run the full pipeline and return one combined long-format DataFrame.

    download_all -> clean each -> add_features each -> concat, sorted by
    (ticker, Date). Each row is one trading day for one ticker, fully labelled.
    """
    raw = download_all(tickers)

    frames: list[pd.DataFrame] = []
    for ticker, df in raw.items():
        featured = add_features(clean(df), ticker)
        if not featured.empty:
            frames.append(featured)

    if not frames:
        raise RuntimeError("No data could be built for any ticker.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return combined


def save_processed(df: pd.DataFrame, name: str = "dataset.csv") -> Path:
    """Write the combined dataset to data/processed/{name}."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_csv(path, index=False)
    return path


def _summarise(df: pd.DataFrame) -> None:
    """Print a short human-readable summary of the built dataset."""
    print("\nDataset summary")
    print("-" * 40)
    print(f"total rows : {len(df)}")
    print(f"date range : {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print("\nrows per ticker:")
    print(df.groupby("ticker").size().to_string())
    print("\nrows per shariah_status:")
    print(df.groupby("shariah_status").size().to_string())


def main() -> None:
    """Build and persist the data artifacts when run standalone."""
    print(f"Building Quantis dataset for {len(TICKERS)} tickers...")
    df = build_dataset()
    out = save_processed(df)
    _summarise(df)
    print(f"\nSaved processed dataset -> {os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
