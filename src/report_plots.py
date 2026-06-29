"""
src/report_plots.py — Extra figures for the report and presentation.

Standalone helper that turns artifacts already produced by the pipeline
(outputs/training_history.json, data/processed/dataset.csv) into the figures
the main evaluate.py run does NOT generate but that a deep-learning report
still wants: training dynamics, the gradient-clipping experiment, the
hyperparameter search, and exploratory data analysis (EDA).

It only READS existing files — no model, no torch, no retraining — so it never
touches the locked modules. Run after train.py/evaluate.py have produced their
outputs:

    python -m src.report_plots

New figures written to outputs/plots/:
    training_loss_curves.png        LSTM train vs validation loss per epoch
    gradient_clipping_comparison.png  with vs without clipping (+ clip activity)
    hyperparameter_search.png       candidate configs ranked by validation loss
    lstm_vs_gru_training.png        LSTM vs GRU validation-loss curves
    eda_price_series.png            normalised price history, all tickers
    eda_return_correlation.png      correlation heatmap of daily returns
    eda_return_distributions.png    per-ticker daily-return distributions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
HISTORY_PATH = OUTPUTS_DIR / "training_history.json"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"

ACCENT = "#1A8F5F"
ACCENT2 = "#E08A2B"
MUTED = "#888888"


def _save(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def _cfg_label(cfg: dict) -> str:
    return f"w{cfg['window']}·h{cfg['hidden_size']}·d{cfg['dropout']}"


# --------------------------------------------------------------------------- #
# Training-dynamics figures (from training_history.json)
# --------------------------------------------------------------------------- #

def plot_training_curves(hist: dict) -> None:
    lstm = hist["final_models"]["lstm"]
    epochs = np.arange(1, len(lstm["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, lstm["train_loss"], label="train loss", color=ACCENT, lw=2)
    ax.plot(epochs, lstm["val_loss"], label="validation loss", color=ACCENT2, lw=2)
    best = int(np.argmin(lstm["val_loss"]))
    ax.scatter([best + 1], [lstm["val_loss"][best]], color=ACCENT2, zorder=5,
               label=f"best val (epoch {best + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("LSTM training vs validation loss")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, "training_loss_curves.png")


def plot_gradient_clipping(hist: dict) -> None:
    gc = hist["gradient_clipping"]
    wc, nc = gc["with_clip"], gc["without_clip"]
    epochs = np.arange(1, len(wc["val_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, nc["val_loss"], label="without clipping", color=MUTED, lw=2)
    ax1.plot(epochs, wc["val_loss"], label="with clipping", color=ACCENT, lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation MSE")
    ax1.set_title("Gradient clipping: validation loss")
    ax1.legend()
    ax1.grid(alpha=0.25)

    frac = np.array(wc["grad_clip_frac"]) * 100
    ax2.bar(epochs, frac, color=ACCENT, alpha=0.85, label="% of steps clipped")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("% of steps clipped", color=ACCENT)
    ax2.set_title("When clipping actually engages")
    ax2b = ax2.twinx()
    ax2b.plot(epochs, wc["grad_norm_max"], color=ACCENT2, lw=2, marker="o",
              ms=3, label="max grad norm")
    ax2b.axhline(hist["config_used"]["clip_norm"], color="red", ls="--", lw=1,
                 label=f"clip threshold = {hist['config_used']['clip_norm']}")
    ax2b.set_ylabel("max gradient norm", color=ACCENT2)
    lines = ax2.get_legend_handles_labels()[0] + ax2b.get_legend_handles_labels()[0]
    labels = ax2.get_legend_handles_labels()[1] + ax2b.get_legend_handles_labels()[1]
    ax2b.legend(lines, labels, fontsize=8, loc="upper right")
    fig.suptitle("Gradient-clipping experiment (clipping mostly acts in early training)")
    fig.tight_layout()
    _save(fig, "gradient_clipping_comparison.png")


def plot_hyperparameter_search(hist: dict) -> None:
    cands = hist["config_selection"]["candidates"]
    labels = [_cfg_label(c["config"]) for c in cands]
    vals = [c["best_val_loss"] for c in cands]
    colors = [ACCENT if c.get("winner") else "#B8C2CC" for c in cands]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors)
    for bar, v, c in zip(bars, vals, cands):
        tag = "  ← winner" if c.get("winner") else ""
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3e}{tag}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Best validation MSE")
    ax.set_xlabel("Configuration (window · hidden · dropout)")
    ax.set_title("Hyperparameter search — lowest validation loss wins")
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "hyperparameter_search.png")


def plot_lstm_vs_gru_training(hist: dict) -> None:
    lstm = hist["final_models"]["lstm"]["val_loss"]
    gru = hist["final_models"]["gru"]["val_loss"]
    epochs = np.arange(1, len(lstm) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, lstm, label=f"LSTM (best {min(lstm):.3e})", color=ACCENT, lw=2)
    ax.plot(epochs, gru, label=f"GRU (best {min(gru):.3e})", color=ACCENT2, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_title("LSTM vs GRU — validation loss over training")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, "lstm_vs_gru_training.png")


# --------------------------------------------------------------------------- #
# EDA figures (from dataset.csv)
# --------------------------------------------------------------------------- #

def plot_price_series(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="Date", columns="ticker", values="Adj Close")
    normed = pivot / pivot.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in normed.columns:
        ax.plot(normed.index, normed[col], lw=1.4, label=col)
    ax.set_ylabel("Price (indexed to 100 at start)")
    ax.set_title("Normalised price history — all tickers")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "eda_price_series.png")


def plot_return_correlation(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="Date", columns="ticker", values="log_return")
    corr = pivot.corr()
    tickers = list(corr.columns)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tickers)), tickers, rotation=45, ha="right")
    ax.set_yticks(range(len(tickers)), tickers)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    ax.set_title("Daily-return correlation across tickers")
    fig.tight_layout()
    _save(fig, "eda_return_correlation.png")


def plot_return_distributions(df: pd.DataFrame) -> None:
    tickers = sorted(df["ticker"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, t in zip(axes.ravel(), tickers):
        r = df.loc[df["ticker"] == t, "log_return"].to_numpy()
        ax.hist(r, bins=60, color=ACCENT, alpha=0.8)
        ax.axvline(np.mean(r), color=ACCENT2, ls="--", lw=1,
                   label=f"μ={np.mean(r):.4f}\nσ={np.std(r):.4f}")
        ax.set_title(t)
        ax.legend(fontsize=7)
    fig.suptitle("Daily log-return distributions by ticker")
    fig.tight_layout()
    _save(fig, "eda_return_distributions.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"{HISTORY_PATH} not found — run train.py first.")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found — run data.py first.")

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        hist = json.load(f)
    df = pd.read_csv(DATASET_PATH, parse_dates=["Date"])

    print("Generating report/presentation figures...")
    plot_training_curves(hist)
    plot_gradient_clipping(hist)
    plot_hyperparameter_search(hist)
    plot_lstm_vs_gru_training(hist)
    plot_price_series(df)
    plot_return_correlation(df)
    plot_return_distributions(df)
    print(f"\nDone -> {PLOTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
