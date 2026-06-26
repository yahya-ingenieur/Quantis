"""
src/evaluate.py — Metrics, backtest, group comparison, and plots.

Loads the test windows (data/processed/windows.npz) and the trained checkpoints
(checkpoints/lstm.pt, gru.pt), then produces every evaluation number and figure:

  1. Load windows + checkpoints (+ training_history.json).
  2. Core metrics (MAE, RMSE, directional accuracy, binomial test vs 50%) for the
     LSTM, GRU, naive persistence and linear regression — one comparable table.
  3. Per-ticker RMSE breakdown.
  4. Monte Carlo dropout forecast + confidence band on sample test examples.
  5. Shariah illustration: AAPL/META vs JPM (illustrative, not powered).
  6. FinTech layer: rolling volatility, historical VaR, threshold signal,
     long-only backtest vs buy-and-hold (cumulative return + Sharpe).
  7. Permutation feature importance on the LSTM.
  8. All blueprint plots that belong here -> outputs/plots/.
  9. Every plotted number -> outputs/evaluation_results.json.

windows.npz has no dates / prices / tags, so reconstruct_test_metadata() replays
train.py's exact per-ticker split + n-2 windowing on dataset.csv to recover them,
asserting the realized returns match y_test (alignment is verified, not assumed).
"""

from __future__ import annotations

import json

# torch must be imported before pandas on this machine (WinError 1114 otherwise).
import torch

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import r2_score

from src.model import GRUForecaster, LSTMForecaster, mc_dropout_predict
from src.train import (
    CHECKPOINTS_DIR,
    DATASET_PATH,
    FEATURE_COLS,
    HISTORY_PATH,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    SPLIT,
    WINDOWS_PATH,
    load_data,
)

PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_PATH = OUTPUTS_DIR / "evaluation_results.json"

# Trading days per year, for annualising volatility and Sharpe.
TRADING_DAYS = 252

# Per-trade transaction cost (5 bps = 0.05%), charged on every position change
# (entry/exit). A realistic frictions assumption so the backtest isn't optimistic.
COST_BPS = 0.0005

# Signal threshold as a fraction of each ticker's OWN train-period return std.
# Rationale: a fixed t=0 would go long on any positive prediction, including
# predictions smaller than the stock's daily noise -> almost always invested.
# Scaling to half a standard deviation makes the threshold adapt to each ticker's
# volatility, so the strategy only commits when the predicted move is meaningful
# relative to that stock's typical day, and genuinely goes flat otherwise.
SIGNAL_THRESHOLD_FRAC = 0.5

# Reproducible permutation importance.
PERM_SEED = 0
PERM_REPEATS = 5

# Representative ticker for the single-name signal-overlay chart.
OVERLAY_TICKER = "AAPL"


# --------------------------------------------------------------------------- #
# JSON helper
# --------------------------------------------------------------------------- #

def jsonable(o):
    """Recursively convert numpy/pandas scalars and arrays to plain Python."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.date().isoformat()
    return o


def save_fig(fig, name: str) -> None:
    """Save a figure into outputs/plots/ and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Section 1 — Load windows, checkpoints, metadata
# --------------------------------------------------------------------------- #

def load_windows() -> dict:
    """Load the pooled test arrays + baseline predictions from windows.npz."""
    z = np.load(WINDOWS_PATH, allow_pickle=True)
    return {
        "X_test": z["X_test"].astype(np.float32),
        "y_test": z["y_test"].astype(np.float32),
        "id_test": z["id_test"].astype(np.int64),
        "persistence_test_pred": z["persistence_test_pred"].astype(np.float32),
        "linreg_test_pred": z["linreg_test_pred"].astype(np.float32),
        "window": int(z["window"]),
        "n_tickers": int(z["n_tickers"]),
        "feature_cols": [str(c) for c in z["feature_cols"]],
    }


def load_model(path):
    """Rebuild a model from a checkpoint and load its weights (eval mode)."""
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    cls = LSTMForecaster if ckpt["model_type"] == "lstm" else GRUForecaster
    model = cls(
        n_features=len(ckpt["feature_cols"]),
        n_tickers=ckpt["n_tickers"],
        hidden_size=cfg["hidden_size"],
        embedding_dim=cfg["embedding_dim"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def model_predict(model, X: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Deterministic test predictions (always resets to eval mode)."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X), torch.from_numpy(ids))
    return pred.numpy()


def reconstruct_test_metadata(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Replay train.py's per-ticker split + n-2 windowing to recover, aligned 1:1
    with the pooled test windows: ticker, tags, dates, price, realized return.

    Iterates tickers in sorted order (matching pandas groupby in train.py) and
    test-chunk endpoints j in [window-1, cn-2], exactly as make_windows does.
    """
    rows = []
    for ticker in sorted(df["ticker"].unique()):
        g = df[df["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        n = len(g)
        val_end = int((SPLIT[0] + SPLIT[1]) * n)
        chunk = g.iloc[val_end:].reset_index(drop=True)
        cn = len(chunk)
        for j in range(window - 1, cn - 1):  # n-2 cap, same as training
            rows.append({
                "ticker": ticker,
                "ticker_id": int(chunk["ticker_id"].iloc[j]),
                "shariah_status": chunk["shariah_status"].iloc[j],
                "endpoint_date": chunk["Date"].iloc[j],
                "realization_date": chunk["Date"].iloc[j + 1],
                "adj_close": float(chunk["Adj Close"].iloc[j]),
                "realized_return": float(chunk["target"].iloc[j]),
                "last_return": float(chunk["log_return"].iloc[j]),
            })
    return pd.DataFrame(rows)


def train_target_std(df: pd.DataFrame) -> dict:
    """Per-ticker std of next-day return over the TRAIN portion (no leakage)."""
    out = {}
    for ticker in sorted(df["ticker"].unique()):
        g = df[df["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        train_end = int(SPLIT[0] * len(g))
        out[ticker] = float(np.nanstd(g["target"].iloc[:train_end].to_numpy()))
    return out


# --------------------------------------------------------------------------- #
# Section 2 — Core metrics (+ binomial significance)
# --------------------------------------------------------------------------- #

def compute_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """MAE, RMSE, directional accuracy and a two-sided binomial test vs 50%.

    Zero-target days are excluded from the directional count (ambiguous sign).
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

    mask = y != 0
    correct = np.sign(pred[mask]) == np.sign(y[mask])
    n = int(mask.sum())
    k = int(correct.sum())
    dir_acc = k / n if n else float("nan")
    p_value = binomtest(k, n, 0.5, alternative="two-sided").pvalue if n else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_acc": dir_acc,
        "n_directional": n,
        "n_correct": k,
        "binom_p_value": float(p_value),
    }


# --------------------------------------------------------------------------- #
# Section 6 helpers — VaR, volatility, signal, backtest
# --------------------------------------------------------------------------- #

def historical_var(returns: np.ndarray, level: float = 0.95) -> float:
    """Historical VaR: the loss at the (1-level) empirical quantile (positive)."""
    return float(-np.quantile(returns, 1.0 - level))


def rolling_volatility(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling std of returns (trailing `window` days)."""
    return pd.Series(returns).rolling(window).std().to_numpy()


def backtest_ticker(pred: np.ndarray, realized: np.ndarray, threshold: float) -> dict:
    """Long-only threshold strategy vs buy-and-hold for one ticker.

    Position = 1 (long) only when predicted return > threshold, else flat.
    Returns simple daily returns NET of transaction costs (COST_BPS charged on each
    position change), so cumulative growth and Sharpe reflect realistic frictions.
    """
    position = (pred > threshold).astype(np.float64)              # long-only
    realized_simple = np.exp(realized) - 1.0                       # log -> simple

    # Charge COST_BPS whenever the position changes (enter or exit); start flat.
    prev = np.concatenate(([0.0], position[:-1]))
    turnover = np.abs(position - prev)
    cost = COST_BPS * turnover
    strat_simple = position * realized_simple - cost

    def sharpe(r):
        sd = np.std(r)
        return float(np.mean(r) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    return {
        "position": position,
        "strat_simple": strat_simple,           # net of costs
        "bh_simple": realized_simple,
        "cum_strat": float(np.prod(1 + strat_simple) - 1),
        "cum_bh": float(np.prod(1 + realized_simple) - 1),
        "sharpe_strat": sharpe(strat_simple),
        "sharpe_bh": sharpe(realized_simple),
        "frac_long": float(position.mean()),
        "n_trades": int(turnover.sum()),
        "cost_drag": float(cost.sum()),         # total return lost to costs
    }


# --------------------------------------------------------------------------- #
# Section 7 — Permutation feature importance
# --------------------------------------------------------------------------- #

def permutation_importance(model, X, ids, y, feature_cols) -> dict:
    """Increase in test MSE when each input is shuffled across samples.

    Shuffles each sequence feature in turn, plus ticker_id (the embedding input),
    averaged over PERM_REPEATS shuffles. Larger ΔMSE = more important.
    """
    rng = np.random.default_rng(PERM_SEED)
    baseline = float(np.mean((model_predict(model, X, ids) - y) ** 2))

    importances = {}
    for k, name in enumerate(feature_cols):
        deltas = []
        for _ in range(PERM_REPEATS):
            Xp = X.copy()
            Xp[:, :, k] = X[rng.permutation(len(X)), :, k]
            mse = float(np.mean((model_predict(model, Xp, ids) - y) ** 2))
            deltas.append(mse - baseline)
        importances[name] = float(np.mean(deltas))

    # ticker_id / embedding input.
    deltas = []
    for _ in range(PERM_REPEATS):
        ids_p = ids[rng.permutation(len(ids))]
        mse = float(np.mean((model_predict(model, X, ids_p) - y) ** 2))
        deltas.append(mse - baseline)
    importances["ticker_id"] = float(np.mean(deltas))

    return {"baseline_mse": baseline, "importance": importances}


# --------------------------------------------------------------------------- #
# Plotting (Section 8)
# --------------------------------------------------------------------------- #

def plot_pred_vs_actual(y, lstm_pred, gru_pred):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pred, name in zip(axes, (lstm_pred, gru_pred), ("LSTM", "GRU")):
        ax.scatter(y, pred, s=6, alpha=0.3)
        lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("Actual next-day log return")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}: predicted vs actual  (R²={r2_score(y, pred):.3f})")
    fig.tight_layout()
    save_fig(fig, "predicted_vs_actual.png")


def plot_residuals(y, lstm_pred):
    resid = y - lstm_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(lstm_pred, resid, s=6, alpha=0.3)
    axes[0].axhline(0, color="r", lw=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("LSTM residuals vs predicted")
    axes[1].hist(resid, bins=60)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("LSTM residual distribution")
    fig.tight_layout()
    save_fig(fig, "residuals.png")


def plot_directional_accuracy(metrics_table):
    names = list(metrics_table.keys())
    accs = [metrics_table[n]["directional_acc"] for n in names]
    pvals = [metrics_table[n]["binom_p_value"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color="steelblue")
    ax.axhline(0.5, color="r", ls="--", lw=1, label="chance (50%)")
    for bar, acc, p in zip(bars, accs, pvals):
        star = " *" if p < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
                f"{acc:.3f}{star}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Directional accuracy vs baselines (* = p<0.05 vs chance)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "directional_accuracy.png")


def plot_lstm_vs_gru(metrics_table):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ("rmse", "directional_acc"),
                              ("Test RMSE (lower=better)", "Directional accuracy")):
        vals = [metrics_table["LSTM"][key], metrics_table["GRU"][key]]
        ax.bar(["LSTM", "GRU"], vals, color=["steelblue", "darkorange"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        if key == "directional_acc":
            ax.axhline(0.5, color="r", ls="--", lw=1)
    fig.suptitle("LSTM vs GRU (test set)")
    fig.tight_layout()
    save_fig(fig, "lstm_vs_gru.png")


def plot_per_ticker_rmse(per_ticker):
    tickers = list(per_ticker.keys())
    lstm_rmse = [per_ticker[t]["lstm_rmse"] for t in tickers]
    gru_rmse = [per_ticker[t]["gru_rmse"] for t in tickers]
    x = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, lstm_rmse, 0.4, label="LSTM", color="steelblue")
    ax.bar(x + 0.2, gru_rmse, 0.4, label="GRU", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("RMSE")
    ax.set_title("Per-ticker test RMSE")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "per_ticker_rmse.png")


def plot_mc_dropout(mc):
    labels = mc["tickers"]
    x = np.arange(len(labels))
    mean = np.array(mc["mean"])
    lower = np.array(mc["lower"])
    upper = np.array(mc["upper"])
    actual = np.array(mc["actual"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, mean, yerr=[mean - lower, upper - mean], fmt="o",
                capsize=4, label="MC mean ± 95% band", color="steelblue")
    ax.scatter(x, actual, marker="x", color="red", s=60, label="actual")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Next-day log return")
    ax.set_title("Monte Carlo dropout forecast + confidence band")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "mc_dropout_band.png")


def plot_shariah(shariah):
    groups = ["shariah (AAPL/META)", "conventional (JPM)"]
    metrics = [("mean_return", "Mean daily return"),
               ("volatility_annual", "Volatility (annual)"),
               ("lstm_rmse", "LSTM RMSE"),
               ("directional_acc", "Directional acc")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (key, title) in zip(axes, metrics):
        vals = [shariah["shariah"][key], shariah["conventional"][key]]
        ax.bar(groups, vals, color=["seagreen", "slategray"])
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Shariah illustration — illustrative two-stock example, "
                 "NOT a statistically powered comparison")
    fig.tight_layout()
    save_fig(fig, "shariah_illustration.png")


def plot_var_distributions(var_info):
    tickers = list(var_info.keys())
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, t in zip(axes.ravel(), tickers):
        returns = np.array(var_info[t]["returns"])
        var = var_info[t]["var_95"]
        ax.hist(returns, bins=40, color="steelblue", alpha=0.8)
        ax.axvline(-var, color="red", ls="--", lw=1.2, label=f"VaR95={var:.3f}")
        ax.set_title(t)
        ax.legend(fontsize=8)
    fig.suptitle("Return distributions with 95% historical VaR cutoff")
    fig.tight_layout()
    save_fig(fig, "return_distribution_var.png")


def plot_signal_overlay(dates, price, position, ticker):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, price, color="black", lw=1, label=f"{ticker} price")
    long_days = position == 1
    ax.scatter(dates[long_days], price[long_days], marker="^", color="green",
               s=25, label="long signal", zorder=3)
    ax.scatter(dates[~long_days], price[~long_days], marker="v", color="lightgray",
               s=12, label="flat", zorder=2)
    ax.set_ylabel("Adj Close")
    ax.set_title(f"{ticker}: buy/hold/sell signal overlay (test period)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "signal_overlay.png")


def plot_cumulative(port):
    dates = port["dates"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, port["equity_strat"], label=f"strategy (Sharpe {port['sharpe_strat']:.2f})",
            color="steelblue")
    ax.plot(dates, port["equity_bh"], label=f"buy & hold (Sharpe {port['sharpe_bh']:.2f})",
            color="darkorange")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_ylabel("Growth of 1 (equal-weight portfolio)")
    ax.set_title("Cumulative return: threshold strategy vs buy-and-hold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "cumulative_return.png")


def plot_permutation(perm):
    imp = perm["importance"]
    names = list(imp.keys())
    vals = [imp[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color="steelblue")
    ax.set_xlabel("Increase in test MSE when shuffled")
    ax.set_title("LSTM permutation feature importance")
    fig.tight_layout()
    save_fig(fig, "permutation_importance.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    # --- Section 1: load ---------------------------------------------------- #
    w = load_windows()
    X, y, ids = w["X_test"], w["y_test"], w["id_test"]
    df = load_data(DATASET_PATH)
    meta = reconstruct_test_metadata(df, w["window"])

    # Verify the reconstruction lines up exactly with the saved windows.
    assert len(meta) == len(y), f"meta {len(meta)} vs y {len(y)}"
    assert np.allclose(meta["realized_return"].to_numpy(), y, atol=1e-5), "return mismatch"
    assert np.array_equal(meta["ticker_id"].to_numpy(), ids), "ticker_id mismatch"

    lstm, _ = load_model(CHECKPOINTS_DIR / "lstm.pt")
    gru, _ = load_model(CHECKPOINTS_DIR / "gru.pt")
    lstm_pred = model_predict(lstm, X, ids)
    gru_pred = model_predict(gru, X, ids)

    ticker_arr = meta["ticker"].to_numpy()
    status_arr = meta["shariah_status"].to_numpy()
    date_arr = meta["endpoint_date"].to_numpy()
    price_arr = meta["adj_close"].to_numpy()

    results = {"config": {"window": w["window"], "n_tickers": w["n_tickers"],
                          "feature_cols": w["feature_cols"],
                          "signal_threshold_frac": SIGNAL_THRESHOLD_FRAC,
                          "transaction_cost_bps": COST_BPS * 1e4}}

    # --- Section 2: core metrics table -------------------------------------- #
    metrics_table = {
        "LSTM": compute_metrics(y, lstm_pred),
        "GRU": compute_metrics(y, gru_pred),
        "persistence": compute_metrics(y, w["persistence_test_pred"]),
        "linreg": compute_metrics(y, w["linreg_test_pred"]),
    }
    results["metrics"] = metrics_table
    print("[2] core metrics:")
    for name, m in metrics_table.items():
        print(f"    {name:12s} RMSE={m['rmse']:.5e} dir={m['directional_acc']:.3f} "
              f"p={m['binom_p_value']:.3g}")

    # --- Section 3: per-ticker RMSE ----------------------------------------- #
    per_ticker = {}
    for t in sorted(np.unique(ticker_arr)):
        m = ticker_arr == t
        per_ticker[t] = {
            "lstm_rmse": float(np.sqrt(np.mean((y[m] - lstm_pred[m]) ** 2))),
            "gru_rmse": float(np.sqrt(np.mean((y[m] - gru_pred[m]) ** 2))),
            "n": int(m.sum()),
        }
    results["per_ticker_rmse"] = per_ticker

    # --- Section 4: MC dropout on one example per ticker -------------------- #
    sel_idx = [int(np.where(ticker_arr == t)[0][np.sum(ticker_arr == t) // 2])
               for t in sorted(np.unique(ticker_arr))]
    sel_idx = np.array(sel_idx)
    mean, lower, upper, std = mc_dropout_predict(
        lstm, torch.from_numpy(X[sel_idx]), torch.from_numpy(ids[sel_idx]),
        n_passes=50, ci=0.95)
    mc = {
        "tickers": [str(ticker_arr[i]) for i in sel_idx],
        "mean": mean.numpy(), "lower": lower.numpy(),
        "upper": upper.numpy(), "std": std.numpy(),
        "actual": y[sel_idx],
    }
    results["mc_dropout"] = mc

    # --- Section 5: Shariah illustration ------------------------------------ #
    shariah = {}
    for label, status in (("shariah", "shariah"), ("conventional", "conventional")):
        m = status_arr == status
        shariah[label] = {
            "tickers": sorted(set(ticker_arr[m].tolist())),
            "mean_return": float(np.mean(y[m])),
            "volatility_annual": float(np.std(y[m]) * np.sqrt(TRADING_DAYS)),
            "lstm_rmse": float(np.sqrt(np.mean((y[m] - lstm_pred[m]) ** 2))),
            "directional_acc": compute_metrics(y[m], lstm_pred[m])["directional_acc"],
            "n": int(m.sum()),
        }
    shariah["note"] = ("Illustrative two-stock example (AAPL/META vs JPM); "
                       "NOT a statistically powered comparison.")
    results["shariah_illustration"] = shariah

    # --- Section 6: VaR, volatility, signal, backtest ----------------------- #
    thr_std = train_target_std(df)
    var_info, backtests = {}, {}
    strat_cols, bh_cols, port_dates = [], [], None
    for t in sorted(np.unique(ticker_arr)):
        m = ticker_arr == t
        order = np.argsort(date_arr[m])           # ensure chronological
        ret = y[m][order]
        pred_t = lstm_pred[m][order]
        threshold = SIGNAL_THRESHOLD_FRAC * thr_std[t]

        var_info[t] = {
            "returns": ret,
            "var_95": historical_var(ret, 0.95),
            "mean_rolling_vol": float(np.nanmean(rolling_volatility(ret))),
        }
        bt = backtest_ticker(pred_t, ret, threshold)
        backtests[t] = {k: v for k, v in bt.items()
                        if k not in ("position", "strat_simple", "bh_simple")}
        backtests[t]["threshold"] = float(threshold)
        strat_cols.append(bt["strat_simple"])
        bh_cols.append(bt["bh_simple"])
        if port_dates is None:
            port_dates = date_arr[m][order]

    results["var"] = {t: {"var_95": var_info[t]["var_95"],
                          "mean_rolling_vol": var_info[t]["mean_rolling_vol"]}
                      for t in var_info}
    results["backtest_per_ticker"] = backtests

    # Equal-weight portfolio (test windows are date-aligned across the balanced panel).
    lengths = {len(c) for c in strat_cols}
    assert len(lengths) == 1, f"unequal test lengths, cannot align portfolio: {lengths}"
    strat_mat = np.vstack(strat_cols)
    bh_mat = np.vstack(bh_cols)
    port_strat = strat_mat.mean(axis=0)
    port_bh = bh_mat.mean(axis=0)
    port = {
        "dates": port_dates,
        "equity_strat": np.cumprod(1 + port_strat),
        "equity_bh": np.cumprod(1 + port_bh),
        "cum_strat": float(np.prod(1 + port_strat) - 1),
        "cum_bh": float(np.prod(1 + port_bh) - 1),
        "sharpe_strat": float(np.mean(port_strat) / np.std(port_strat) * np.sqrt(TRADING_DAYS)),
        "sharpe_bh": float(np.mean(port_bh) / np.std(port_bh) * np.sqrt(TRADING_DAYS)),
    }
    results["portfolio_backtest"] = {k: v for k, v in port.items()
                                     if k not in ("dates", "equity_strat", "equity_bh")}
    print(f"[6] portfolio: strat cum={port['cum_strat']:.3f} (Sharpe {port['sharpe_strat']:.2f}) "
          f"vs B&H cum={port['cum_bh']:.3f} (Sharpe {port['sharpe_bh']:.2f})")

    # --- Section 7: permutation importance ---------------------------------- #
    perm = permutation_importance(lstm, X, ids, y, w["feature_cols"])
    results["permutation_importance"] = perm
    print(f"[7] permutation importance: {perm['importance']}")

    # --- R² for the report -------------------------------------------------- #
    results["r2"] = {"LSTM": float(r2_score(y, lstm_pred)),
                     "GRU": float(r2_score(y, gru_pred))}

    # --- Section 8: plots --------------------------------------------------- #
    plot_pred_vs_actual(y, lstm_pred, gru_pred)
    plot_residuals(y, lstm_pred)
    plot_directional_accuracy(metrics_table)
    plot_lstm_vs_gru(metrics_table)
    plot_per_ticker_rmse(per_ticker)
    plot_mc_dropout(mc)
    plot_shariah(shariah)
    plot_var_distributions(var_info)
    plot_permutation(perm)

    # signal overlay for the representative ticker
    m = ticker_arr == OVERLAY_TICKER
    order = np.argsort(date_arr[m])
    threshold = SIGNAL_THRESHOLD_FRAC * thr_std[OVERLAY_TICKER]
    position = (lstm_pred[m][order] > threshold).astype(int)
    plot_signal_overlay(date_arr[m][order], price_arr[m][order], position, OVERLAY_TICKER)
    plot_cumulative(port)

    # --- Section 9: results JSON -------------------------------------------- #
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(results), f, indent=2)

    print("\nSaved:")
    print(f"  {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  11 figures -> {PLOTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
