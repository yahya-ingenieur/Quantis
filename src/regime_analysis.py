"""
src/regime_analysis.py — Direction accuracy by market regime.

Splits the test set into "trending" (|20-day rolling return| > 0.5 × rolling σ)
vs "choppy" periods and reports directional accuracy in each. This turns the flat
~50% overall figure into a nuanced finding.

Outputs: outputs/regime_analysis_results.json + plots/regime_analysis.png

Run (after training + evaluation):
    python -m src.regime_analysis
"""

from __future__ import annotations

import json

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.train import OUTPUTS_DIR, CHECKPOINTS_DIR, DATASET_PATH, SPLIT, load_data
from src.evaluate import (
    load_model, load_windows, reconstruct_test_metadata,
    compute_metrics, PLOTS_DIR,
)

ROLLING_WINDOW = 20          # trading days for regime classification
TREND_THRESHOLD_FRAC = 0.5   # |rolling_mean| > frac × rolling_std → trending


def _regime(rolling_mean: float, rolling_std: float) -> str:
    if rolling_std <= 0:
        return "choppy"
    return "trending" if abs(rolling_mean) > TREND_THRESHOLD_FRAC * rolling_std else "choppy"


def run() -> dict:
    w = load_windows()
    X, y, ids = w["X_test"], w["y_test"], w["id_test"]
    df = load_data(DATASET_PATH)
    meta = reconstruct_test_metadata(df, w["window"])

    model, _ = load_model(CHECKPOINTS_DIR / "lstm.pt")
    with torch.no_grad():
        pred = model(torch.from_numpy(X), torch.from_numpy(ids)).numpy()

    # Build per-ticker rolling stats on the FULL series, then index by test endpoint date.
    regime_labels = []
    for ticker in sorted(df["ticker"].unique()):
        g = (df[df["ticker"] == ticker]
             .sort_values("Date")
             .reset_index(drop=True))
        n = len(g)
        val_end = int((SPLIT[0] + SPLIT[1]) * n)
        chunk = g.iloc[val_end:].reset_index(drop=True)
        cn = len(chunk)

        # Full-series rolling mean & std of log_return
        rolling_mean = g["log_return"].rolling(ROLLING_WINDOW).mean()
        rolling_std  = g["log_return"].rolling(ROLLING_WINDOW).std()
        date_to_idx  = {d: i for i, d in enumerate(g["Date"])}

        for j in range(w["window"] - 1, cn - 1):
            ep = chunk["Date"].iloc[j]
            gi = date_to_idx.get(ep)
            if gi is None or pd.isna(rolling_mean.iloc[gi]):
                regime_labels.append("unknown")
            else:
                regime_labels.append(
                    _regime(float(rolling_mean.iloc[gi]),
                            float(rolling_std.iloc[gi]) if not pd.isna(rolling_std.iloc[gi]) else 0.0))

    regime_labels = np.array(regime_labels)

    # Direction accuracy per regime
    results: dict = {}
    for regime in ["trending", "choppy", "unknown"]:
        mask = regime_labels == regime
        if mask.sum() == 0:
            continue
        m = compute_metrics(y[mask], pred[mask])
        results[regime] = {
            "n": int(mask.sum()),
            "fraction_of_test": float(mask.mean()),
            "directional_acc": m["directional_acc"],
            "binom_p_value": m["binom_p_value"],
            "rmse": m["rmse"],
        }

    results["overall"] = {"n": int(len(y)), **compute_metrics(y, pred)}

    # --- Plot --------------------------------------------------------------- #
    plot_regimes = [r for r in ("trending", "choppy") if r in results]
    dir_accs = [results[r]["directional_acc"] for r in plot_regimes]
    ns        = [results[r]["n"] for r in plot_regimes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bars = axes[0].bar(plot_regimes, dir_accs, color=["steelblue", "darkorange"])
    axes[0].axhline(0.5, color="red", ls="--", lw=1, label="chance (50%)")
    axes[0].set_ylim(0, 0.70)
    for bar, acc, n_r in zip(bars, dir_accs, ns):
        axes[0].text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
                     f"{acc:.1%}\n(n={n_r})", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Directional accuracy by market regime")
    axes[0].set_ylabel("Directional accuracy")
    axes[0].legend()

    axes[1].pie(
        ns,
        labels=[f"{r}\n({n_r} samples)" for r, n_r in zip(plot_regimes, ns)],
        colors=["steelblue", "darkorange"],
        autopct="%1.0f%%",
        startangle=90,
    )
    axes[1].set_title(
        f"Test-set regime distribution\n"
        f"({ROLLING_WINDOW}-day rolling return, threshold = {TREND_THRESHOLD_FRAC}σ)")

    fig.suptitle("Regime analysis: does market trend affect directional accuracy?")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "regime_analysis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    out_path = OUTPUTS_DIR / "regime_analysis_results.json"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Directional accuracy by regime:")
    for r, v in results.items():
        print(f"  {r:10s}: {v['directional_acc']:.1%}  "
              f"(n={v['n']}, p={v['binom_p_value']:.3f})")
    print(f"Saved -> {out_path}")
    return results


if __name__ == "__main__":
    run()
