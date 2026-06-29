"""
src/robustness.py — Multi-seed robustness check (LSTM vs GRU).

Addresses the "trained only once" weakness: the main pipeline trains each model
on a single seed, so the headline RMSE / directional numbers could be partly
luck. This retrains the winning config across several seeds and reports the
mean +/- std, so every comparison is defensible.

It REUSES train.py's functions (build_split_windows, build_model, train_model,
set_seed) and evaluate.py's compute_metrics — it does not modify any locked
module. The only difference from train.train_run is that the seed (model init
AND batch-shuffle order) is varied instead of fixed.

Run:
    python -m src.robustness
"""

from __future__ import annotations

import json

import torch

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import compute_metrics
from src.train import (FEATURE_COLS, HISTORY_PATH, OUTPUTS_DIR, PROJECT_ROOT, SPLIT,
                       WINDOWS_PATH, build_model, build_split_windows, load_data,
                       set_seed, train_model)

SEEDS = [42, 0, 1, 2, 3]
RESULTS_PATH = OUTPUTS_DIR / "robustness_results.json"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def make_loader(arrays: dict, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(arrays["X"]), torch.from_numpy(arrays["id"]),
                       torch.from_numpy(arrays["y"]))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def run_seed(kind: str, config: dict, data: dict, n_tickers: int, seed: int) -> dict:
    """Train one model at a given seed and return its test metrics."""
    set_seed(seed)
    model = build_model(kind, config, n_tickers)
    tl = make_loader(data["train"], config["batch_size"], True, seed)
    vl = make_loader(data["val"], config["batch_size"], False, seed)
    train_model(model, tl, vl, config, clip=True)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(data["test"]["X"]),
                     torch.from_numpy(data["test"]["id"])).numpy()
    return compute_metrics(data["test"]["y"], pred)


def main() -> None:
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)["config_used"]
    df = load_data()
    n_tickers = int(df["ticker_id"].max()) + 1
    data = build_split_windows(df, FEATURE_COLS, config["window"], SPLIT)
    print(f"winner config: window={config['window']} hidden={config['hidden_size']} "
          f"dropout={config['dropout']} | test windows={len(data['test']['y'])}")

    per_seed = {"LSTM": [], "GRU": []}
    for kind, key in (("lstm", "LSTM"), ("gru", "GRU")):
        for seed in SEEDS:
            m = run_seed(kind, config, data, n_tickers, seed)
            per_seed[key].append({"seed": seed, "rmse": m["rmse"],
                                  "directional_acc": m["directional_acc"],
                                  "binom_p_value": m["binom_p_value"]})
            print(f"  {key} seed={seed}: RMSE={m['rmse']:.5e} dir={m['directional_acc']:.3f}",
                  flush=True)

    agg = {}
    for key in ("LSTM", "GRU"):
        rmses = [r["rmse"] for r in per_seed[key]]
        dirs = [r["directional_acc"] for r in per_seed[key]]
        agg[key] = {"rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                    "dir_mean": float(np.mean(dirs)), "dir_std": float(np.std(dirs))}

    # for reference lines in the plot
    z = np.load(WINDOWS_PATH, allow_pickle=True)
    yb = z["y_test"].astype(float)
    base = {"persistence": float(np.sqrt(np.mean((yb - z["persistence_test_pred"]) ** 2))),
            "linreg": float(np.sqrt(np.mean((yb - z["linreg_test_pred"]) ** 2)))}

    results = {"config": config, "seeds": SEEDS, "per_seed": per_seed,
               "aggregate": agg, "baseline_rmse": base}
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.boxplot([[r["rmse"] for r in per_seed["LSTM"]],
                 [r["rmse"] for r in per_seed["GRU"]]])
    ax1.set_xticks([1, 2]); ax1.set_xticklabels(["LSTM", "GRU"])
    ax1.axhline(base["persistence"], color="red", ls="--", lw=1, label="persistence")
    ax1.axhline(base["linreg"], color="orange", ls="--", lw=1, label="linreg")
    ax1.set_ylabel("Test RMSE"); ax1.set_title(f"RMSE across {len(SEEDS)} seeds")
    ax1.legend(fontsize=8)
    ax2.boxplot([[r["directional_acc"] for r in per_seed["LSTM"]],
                 [r["directional_acc"] for r in per_seed["GRU"]]])
    ax2.set_xticks([1, 2]); ax2.set_xticklabels(["LSTM", "GRU"])
    ax2.axhline(0.5, color="red", ls="--", lw=1, label="chance")
    ax2.set_ylabel("Directional accuracy"); ax2.set_title("Directional accuracy across seeds")
    ax2.legend(fontsize=8)
    fig.suptitle("Multi-seed robustness — LSTM vs GRU")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "robustness_seeds.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\nAggregate (mean +/- std):")
    for key in ("LSTM", "GRU"):
        a = agg[key]
        print(f"  {key}: RMSE {a['rmse_mean']:.5e} +/- {a['rmse_std']:.1e} | "
              f"dir {a['dir_mean']:.3f} +/- {a['dir_std']:.3f}")
    print(f"baselines: persistence RMSE={base['persistence']:.5e}, linreg={base['linreg']:.5e}")
    print(f"saved -> {RESULTS_PATH.relative_to(PROJECT_ROOT)} + plots/robustness_seeds.png")


if __name__ == "__main__":
    main()
