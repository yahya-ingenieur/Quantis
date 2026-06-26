"""
src/feature_ablation.py — Does richer feature engineering help? (ablation study)

A controlled ablation that does NOT modify the locked modules and does NOT change
the main results. It reuses train.py's pipeline (same windowing, splitting,
scaling, model, training loop) and compares two feature sets on the identical
pooled samples, across several seeds:

    baseline : [log_return, Volume]
    + technical indicators : baseline + [RSI(14), SMA-ratio(20),
                                         rolling-vol(20), momentum(10)]

The indicators are computed per ticker from past data only (backward-looking
rolling windows), so there is no look-ahead leakage. We report mean +/- std RMSE
and directional accuracy, plus a Diebold-Mariano test on the baseline-vs-richer
forecast errors.

Run (after the main pipeline has produced dataset.csv + training_history.json):
    python -m src.feature_ablation
"""

from __future__ import annotations

import json

import torch

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import compute_metrics
from src.model import LSTMForecaster
from src.train import (FEATURE_COLS, HISTORY_PATH, OUTPUTS_DIR, PROJECT_ROOT, SPLIT,
                       build_split_windows, load_data, set_seed, train_model)

SEEDS = [42, 0, 1, 2, 3]
INDICATORS = ["rsi14", "sma_ratio", "roll_vol20", "mom10"]
RESULTS_PATH = OUTPUTS_DIR / "feature_ablation_results.json"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-ticker, backward-looking technical indicators, then drop warm-up NaNs.

    All indicators use only data up to day t (rolling windows), so using them to
    predict day t+1's return introduces no leakage.
    """
    frames = []
    for _, g in df.groupby("ticker"):
        g = g.sort_values("Date").copy()
        price = g["Adj Close"]
        delta = price.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        g["rsi14"] = 100 - 100 / (1 + rs)                          # momentum oscillator
        g["sma_ratio"] = price / price.rolling(20).mean() - 1       # distance from SMA20
        g["roll_vol20"] = g["log_return"].rolling(20).std()        # recent volatility
        g["mom10"] = g["log_return"].rolling(10).sum()             # 10-day momentum
        frames.append(g)
    out = pd.concat(frames).dropna(subset=INDICATORS).reset_index(drop=True)
    return out


def _loader(arrays: dict, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(arrays["X"]), torch.from_numpy(arrays["id"]),
                       torch.from_numpy(arrays["y"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def run_seed(df: pd.DataFrame, feature_cols: list[str], config: dict,
             n_tickers: int, seed: int):
    """Train the pooled LSTM on one feature set at one seed; return (metrics, pred, y)."""
    set_seed(seed)
    data = build_split_windows(df, feature_cols, config["window"], SPLIT)
    # Build directly so n_features matches the (possibly richer) feature set —
    # train.build_model hardcodes len(FEATURE_COLS)=2.
    model = LSTMForecaster(n_features=len(feature_cols), n_tickers=n_tickers,
                           hidden_size=config["hidden_size"],
                           embedding_dim=config["embedding_dim"],
                           dropout=config["dropout"])
    tl = _loader(data["train"], config["batch_size"], True, seed)
    vl = _loader(data["val"], config["batch_size"], False, seed)
    train_model(model, tl, vl, config, clip=True)
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


def main() -> None:
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)["config_used"]
    df = add_indicators(load_data())
    n_tickers = int(df["ticker_id"].max()) + 1

    sets = {"baseline": list(FEATURE_COLS),
            "+ indicators": list(FEATURE_COLS) + INDICATORS}
    print(f"rows after indicators: {len(df)} | configs: {list(sets)}")

    seed_metrics = {k: [] for k in sets}
    preds = {}
    for si, seed in enumerate(SEEDS):
        for name, cols in sets.items():
            m, pred, y = run_seed(df, cols, config, n_tickers, seed)
            seed_metrics[name].append(m)
            if si == 0:
                preds[name] = (pred, y)
        print(f"  seed {seed} done", flush=True)

    agg = {}
    for name in sets:
        rmses = [m["rmse"] for m in seed_metrics[name]]
        dirs = [m["directional_acc"] for m in seed_metrics[name]]
        agg[name] = {"rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                     "dir_mean": float(np.mean(dirs)), "dir_std": float(np.std(dirs))}

    y0 = preds["baseline"][1]
    stat, p = diebold_mariano(y0 - preds["baseline"][0], y0 - preds["+ indicators"][0])
    dm = {"dm_stat": stat, "p_value": p, "significant_5pct": bool(p < 0.05)}

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"indicators": INDICATORS, "seeds": SEEDS, "config": config,
                   "aggregate": agg, "diebold_mariano_vs_baseline": dm}, f, indent=2)

    names = list(sets)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(names, [agg[n]["rmse_mean"] for n in names],
                yerr=[agg[n]["rmse_std"] for n in names], capsize=5, color="steelblue")
    axes[0].set_title(f"Test RMSE (mean ± std, {len(SEEDS)} seeds)")
    axes[1].bar(names, [agg[n]["dir_mean"] for n in names],
                yerr=[agg[n]["dir_std"] for n in names], capsize=5, color="seagreen")
    axes[1].axhline(0.5, color="r", ls="--", lw=1, label="chance"); axes[1].legend()
    axes[1].set_title("Directional accuracy (mean ± std)")
    fig.suptitle("Feature ablation: baseline vs + technical indicators")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "feature_ablation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    base = agg["baseline"]["rmse_mean"]
    print(f"\nResults (mean ± std over {len(SEEDS)} seeds):")
    for name in names:
        a = agg[name]
        line = f"  {name:14s} RMSE {a['rmse_mean']:.5e} ± {a['rmse_std']:.1e}  " \
               f"dir {a['dir_mean']:.3f} ± {a['dir_std']:.3f}"
        if name != "baseline":
            delta = (a["rmse_mean"] - base) / base * 100
            sig = "SIGNIFICANT" if dm["significant_5pct"] else "not significant"
            line += f"  | {delta:+.1f}% vs base, DM p={dm['p_value']:.3f} ({sig})"
        print(line)
    print(f"\nSaved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)} + plots/feature_ablation.png")


if __name__ == "__main__":
    main()
