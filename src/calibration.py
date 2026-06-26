"""
src/calibration.py — Empirical coverage of MC-dropout 95% confidence intervals.

Runs the trained LSTM's MC-dropout on every test window and checks what fraction
of true realized returns fall within the 95% CI. If the band is well-calibrated,
empirical coverage should be ~95%. Deviations reveal over/under-confidence.

Outputs: outputs/calibration_results.json + plots/calibration_coverage.png

Run (after training + evaluation):
    python -m src.calibration
"""

from __future__ import annotations

import json

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.train import OUTPUTS_DIR, CHECKPOINTS_DIR
from src.evaluate import load_model, load_windows, PLOTS_DIR
from src.model import mc_dropout_predict

N_PASSES = 200
BATCH = 256  # process test windows in chunks to avoid memory spikes


def run() -> dict:
    w = load_windows()
    X, y, ids = w["X_test"], w["y_test"], w["id_test"]
    n = len(y)

    model, _ = load_model(CHECKPOINTS_DIR / "lstm.pt")

    all_mean  = np.zeros(n, dtype=np.float32)
    all_lower = np.zeros(n, dtype=np.float32)
    all_upper = np.zeros(n, dtype=np.float32)

    print(f"Running {N_PASSES} MC-dropout passes on {n} test samples (batch={BATCH})…")
    torch.manual_seed(0)
    for i in range(0, n, BATCH):
        x_b  = torch.from_numpy(X[i : i + BATCH])
        id_b = torch.from_numpy(ids[i : i + BATCH])
        m, lo, hi, _ = mc_dropout_predict(model, x_b, id_b,
                                          n_passes=N_PASSES, ci=0.95)
        all_mean[i  : i + BATCH] = m.numpy()
        all_lower[i : i + BATCH] = lo.numpy()
        all_upper[i : i + BATCH] = hi.numpy()
        print(f"  batch {i // BATCH + 1}/{(n + BATCH - 1) // BATCH} done", flush=True)

    # Empirical coverage: fraction of true values inside [lower, upper]
    inside = (y >= all_lower) & (y <= all_upper)
    coverage = float(inside.mean())
    widths = all_upper - all_lower

    # Coverage in 10 equal-count bins sorted by band width (wide vs narrow)
    order = np.argsort(widths)
    bin_size = n // 10
    bin_cov, bin_w = [], []
    for b in range(10):
        idx = order[b * bin_size : (b + 1) * bin_size]
        bin_cov.append(float(inside[idx].mean()))
        bin_w.append(float(widths[idx].mean()))

    result = {
        "n_samples": n,
        "n_passes": N_PASSES,
        "nominal_ci": 0.95,
        "empirical_coverage": coverage,
        "mean_band_width_log_return": float(widths.mean()),
        "coverage_gap": float(coverage - 0.95),
        "well_calibrated": bool(abs(coverage - 0.95) < 0.05),
        "bin_analysis": {"mean_width": bin_w, "coverage": bin_cov},
    }

    # --- Plot ---------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(["Inside 95% band", "Outside 95% band"],
                [coverage, 1.0 - coverage],
                color=["steelblue", "salmon"])
    axes[0].axhline(0.95, color="red", ls="--", lw=1.2, label="95% nominal target")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Fraction of test samples")
    axes[0].set_title(f"Empirical coverage = {coverage:.1%}  (nominal 95%)")
    axes[0].legend()
    for bar, val in zip(axes[0].patches, [coverage, 1.0 - coverage]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=10)

    axes[1].plot(bin_w, bin_cov, "o-", color="steelblue", label="empirical per bin")
    axes[1].axhline(0.95, color="red", ls="--", lw=1, label="95% target")
    axes[1].set_xlabel("Mean band width in bin (log-return units)")
    axes[1].set_ylabel("Coverage in bin")
    axes[1].set_title("Coverage vs uncertainty band width\n(wider band → should cover more)")
    axes[1].legend()

    fig.suptitle(f"MC-dropout calibration  ·  {N_PASSES} passes  ·  {n} test samples")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "calibration_coverage.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "calibration_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nEmpirical coverage : {coverage:.1%}  (nominal 95%)")
    print(f"Coverage gap       : {coverage - 0.95:+.1%}")
    print(f"Well-calibrated    : {result['well_calibrated']}")
    print(f"Saved -> {out_path}")
    return result


if __name__ == "__main__":
    run()
