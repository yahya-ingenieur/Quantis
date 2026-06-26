"""
src/sentiment_experiment.py — Does news sentiment improve next-day forecasting?

A self-contained, pooled multi-stock side-experiment (does NOT modify the locked
modules; it reuses their public functions). Runs on local files + small per-ticker
yfinance pulls, so it's reliable.

  1. Read daily sentiment scores (data/processed/sentiment.csv) for whichever
     tickers were collected/scored.
  2. For each ticker, pull daily prices over the sentiment window and build the
     same features as data.py (log_return, Volume, next-day target), then merge
     the sentiment scores (trading days with no news get a neutral 0). Pool all
     tickers with a ticker_id (the LSTM's embedding tells them apart).
  3. Train the LSTM (model.py) under matched conditions for each feature set,
     across several SEEDS (mean +/- std, not a single lucky run):
        baseline   : [log_return, Volume]
        + VADER    : [log_return, Volume, vader_score]
        + FinBERT  : [log_return, Volume, finbert_score]   (if scored)
  4. Compare on the identical pooled test windows + a Diebold-Mariano test on the
     baseline-vs-+FinBERT forecast errors (significance, not just a point delta).

Windowing/splitting/scaling is delegated to train.build_split_windows, so the
no-leakage logic is identical to the main pipeline.

Outputs: outputs/sentiment_results.json + plots/sentiment_experiment.png

Run (after `python -m src.sentiment fetch` and `... score`):
    python -m src.sentiment_experiment
"""

from __future__ import annotations

import json

import torch  # before pandas (WinError 1114)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import compute_metrics
from src.model import LSTMForecaster
from src.train import (OUTPUTS_DIR, PROCESSED_DIR, PROJECT_ROOT, SPLIT,
                       build_split_windows, set_seed, train_model)

SENTIMENT_PATH = PROCESSED_DIR / "sentiment.csv"
RESULTS_PATH = OUTPUTS_DIR / "sentiment_results.json"
PLOTS_DIR = OUTPUTS_DIR / "plots"

WINDOW = 10
SEEDS = [42, 0, 1, 2, 3]
CONFIG = {"hidden_size": 64, "embedding_dim": 8, "dropout": 0.2,
          "lr": 1e-3, "epochs": 50, "batch_size": 64, "clip_norm": 0.1}


def load_combined(sent: pd.DataFrame):
    """Pull prices for every ticker in `sent` and merge daily sentiment.

    Returns (combined_df, sentiment_cols, n_tickers). combined_df has the columns
    train.build_split_windows expects: Date, ticker, ticker_id, log_return,
    Volume, target, plus the sentiment columns.
    """
    tickers = sorted(sent["ticker"].unique())
    tid = {t: i for i, t in enumerate(tickers)}
    sent_cols = [c for c in ("vader_score", "finbert_score") if c in sent.columns]
    start = (sent["Date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (sent["Date"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    frames = []
    for t in tickers:
        px = yf.download(t, start=start, end=end, interval="1d",
                         auto_adjust=False, progress=False)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        px = px.reset_index()[["Date", "Adj Close", "Volume"]].dropna()
        px["Date"] = pd.to_datetime(px["Date"]).dt.normalize()
        px["log_return"] = np.log(px["Adj Close"] / px["Adj Close"].shift(1))
        px["target"] = px["log_return"].shift(-1)
        st = sent.loc[sent["ticker"] == t, ["Date"] + sent_cols]
        px = px.merge(st, on="Date", how="left")
        px[sent_cols] = px[sent_cols].fillna(0.0)        # no news -> neutral
        px["ticker"] = t
        px["ticker_id"] = tid[t]
        px = px.dropna(subset=["log_return", "target"])
        frames.append(px)
        print(f"  {t}: {len(px)} trading days", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    return combined, sent_cols, len(tickers)


def _loader(arrays: dict, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(arrays["X"]), torch.from_numpy(arrays["id"]),
                       torch.from_numpy(arrays["y"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def run_config(df: pd.DataFrame, feature_cols: list[str], n_tickers: int, seed: int):
    """Train the pooled LSTM on one feature set at one seed; return (metrics, pred, y)."""
    set_seed(seed)
    data = build_split_windows(df, feature_cols, WINDOW, SPLIT)
    model = LSTMForecaster(n_features=len(feature_cols), n_tickers=n_tickers,
                           hidden_size=CONFIG["hidden_size"],
                           embedding_dim=CONFIG["embedding_dim"],
                           dropout=CONFIG["dropout"])
    tl = _loader(data["train"], CONFIG["batch_size"], True, seed)
    vl = _loader(data["val"], CONFIG["batch_size"], False, seed)
    train_model(model, tl, vl, CONFIG, clip=True)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(data["test"]["X"]),
                     torch.from_numpy(data["test"]["id"])).numpy()
    return compute_metrics(data["test"]["y"], pred), pred, data["test"]["y"]


def diebold_mariano(e1: np.ndarray, e2: np.ndarray):
    """DM test on squared-error loss (1-step). d = e1^2 - e2^2; returns (stat, p)."""
    d = e1.astype(np.float64) ** 2 - e2.astype(np.float64) ** 2
    n = len(d)
    var = np.var(d, ddof=0) / n
    if var <= 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(var)
    return float(stat), float(2 * (1 - student_t.cdf(abs(stat), df=n - 1)))


def plot_comparison(agg: dict, names: list[str], n_tickers: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, [agg[n]["rmse_mean"] for n in names],
                yerr=[agg[n]["rmse_std"] for n in names], capsize=5, color="steelblue")
    axes[0].set_title(f"Test RMSE (mean ± std, {len(SEEDS)} seeds)")
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[1].bar(names, [agg[n]["dir_mean"] for n in names],
                yerr=[agg[n]["dir_std"] for n in names], capsize=5, color="seagreen")
    axes[1].axhline(0.5, color="r", ls="--", lw=1, label="chance"); axes[1].legend()
    axes[1].set_title("Directional accuracy (mean ± std)")
    axes[1].tick_params(axis="x", labelrotation=15)
    fig.suptitle(f"Does news sentiment improve forecasting? (pooled, {n_tickers} stocks)")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "sentiment_experiment.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sent = pd.read_csv(SENTIMENT_PATH, parse_dates=["Date"])
    print(f"Sentiment covers {sent['ticker'].nunique()} tickers: "
          f"{sorted(sent['ticker'].unique())}")
    df, sent_cols, n_tickers = load_combined(sent)
    print(f"pooled: {len(df)} rows across {n_tickers} tickers")

    configs = [("baseline", ["log_return", "Volume"])]
    if "vader_score" in sent_cols:
        configs.append(("+ VADER", ["log_return", "Volume", "vader_score"]))
    if "finbert_score" in sent_cols:
        configs.append(("+ FinBERT", ["log_return", "Volume", "finbert_score"]))
    names = [n for n, _ in configs]

    seed_metrics = {n: [] for n in names}
    preds = {}
    print(f"Training {len(names)} configs x {len(SEEDS)} seeds...")
    for si, seed in enumerate(SEEDS):
        for name, cols in configs:
            m, pred, y = run_config(df, cols, n_tickers, seed)
            seed_metrics[name].append(m)
            if si == 0:
                preds[name] = (pred, y)
        print(f"  seed {seed} done", flush=True)

    agg = {}
    for name in names:
        rmses = [m["rmse"] for m in seed_metrics[name]]
        dirs = [m["directional_acc"] for m in seed_metrics[name]]
        agg[name] = {"rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                     "dir_mean": float(np.mean(dirs)), "dir_std": float(np.std(dirs))}

    dm = {}
    y0 = preds["baseline"][1]
    e_base = y0 - preds["baseline"][0]
    for name in names[1:]:
        stat, p = diebold_mariano(e_base, preds[name][1] - preds[name][0])
        dm[name] = {"dm_stat": stat, "p_value": p, "significant_5pct": bool(p < 0.05)}

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"tickers": sorted(sent["ticker"].unique()), "n_tickers": n_tickers,
                   "window": WINDOW, "n_rows": len(df), "seeds": SEEDS, "config": CONFIG,
                   "aggregate": agg, "diebold_mariano_vs_baseline": dm}, f, indent=2)
    plot_comparison(agg, names, n_tickers)

    base = agg["baseline"]["rmse_mean"]
    print(f"\nResults (pooled {n_tickers} stocks, mean ± std over {len(SEEDS)} seeds):")
    for name in names:
        a = agg[name]
        line = f"  {name:10s} RMSE {a['rmse_mean']:.5e} ± {a['rmse_std']:.1e}  " \
               f"dir {a['dir_mean']:.3f} ± {a['dir_std']:.3f}"
        if name in dm:
            delta = (a["rmse_mean"] - base) / base * 100
            sig = "SIGNIFICANT" if dm[name]["significant_5pct"] else "not significant"
            line += f"  | {delta:+.1f}% vs base, DM p={dm[name]['p_value']:.3f} ({sig})"
        print(line)
    print(f"\nSaved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)} + plots/sentiment_experiment.png")


if __name__ == "__main__":
    main()
