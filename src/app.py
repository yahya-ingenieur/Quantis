"""
src/app.py — Quantis Streamlit demo.

A single scrolling page with a light, approachable marketplace aesthetic: hero
header, ticker sidebar with a Shariah-status pill, KPI metric row, an interactive
Plotly price chart with a one-day-ahead Monte-Carlo-dropout forecast (point +
uncertainty range), and an honest "Model Transparency" card sourced from the
evaluation JSON. Jargon terms carry plain-language popovers so the page stays
readable for someone with zero ML or finance background.

Under the hood it only LOADS local artifacts — checkpoints, processed data, and
evaluation_results.json — and runs MC-dropout inference on the latest available
window. It never retrains and makes no network calls.

Run:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
import os
# Ensure the repo root is in sys.path so `from src.xxx import` works on
# Streamlit Cloud (which runs the file directly, not from the project root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from math import exp

# torch before pandas/streamlit on this machine (WinError 1114 otherwise).
import torch

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from src.data import SHARIAH_LABELS, TICKERS
from src.train import CHECKPOINTS_DIR, DATASET_PATH, FEATURE_COLS, OUTPUTS_DIR, SPLIT, load_data
from src.evaluate import RESULTS_PATH, load_model, load_windows, reconstruct_test_metadata
from src.model import mc_dropout_predict
try:
    from src.sentiment import live_sentiment, load_finbert
    _SENTIMENT_AVAILABLE = True
except Exception:
    _SENTIMENT_AVAILABLE = False

# How much recent history to pull for a live refresh (enough bars to build one
# window after dropping the first NaN log-return).
LIVE_PERIOD = "4mo"

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

GREEN, AMBER = "#2DD4A8", "#E0A33E"
UP, DOWN, NEUTRAL = "#2DD4A8", "#F1666E", "#8B98A6"
CARD_BG = "#141B24"
GRID = "#232E3A"
TEXT = "#E8EDF2"
N_HISTORY = 150        # recent trading days shown on the chart
N_PASSES = 200         # MC-dropout forward passes for the demo forecast

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root{
  --bg:#0A0E13; --card:#141B24; --border:#232E3A;
  --green:#2DD4A8; --green-dark:#1FA888; --green-tint:rgba(45,212,168,.13); --amber:#E0A33E;
  --up:#2DD4A8; --down:#F1666E; --neutral:#8B98A6;
  --text:#E8EDF2; --muted:#93A1B0;
  --glass-top:rgba(28,37,49,.72); --glass-bot:rgba(16,22,30,.72);
  --hairline:rgba(255,255,255,.06);
}
.stApp{ background:
    radial-gradient(1200px 620px at 0% -10%, rgba(45,212,168,.20) 0%, rgba(45,212,168,0) 52%),
    radial-gradient(1200px 680px at 100% -6%, rgba(91,143,249,.15) 0%, rgba(91,143,249,0) 52%),
    var(--bg);
  color:var(--text); font-family:'Inter',sans-serif; }
/* visible-but-subtle graph-paper grid, fading toward the bottom */
.stApp::before{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
  background-size:40px 40px;
  -webkit-mask-image:linear-gradient(180deg, #000 0%, rgba(0,0,0,.55) 50%, transparent 92%);
  mask-image:linear-gradient(180deg, #000 0%, rgba(0,0,0,.55) 50%, transparent 92%); }
#MainMenu, header, footer{ visibility:hidden; }
.block-container{ padding-top:1.7rem; max-width:1160px; position:relative; z-index:1; }

/* hero */
.hero{ padding:6px 0 14px 0; margin-bottom:8px; }
.hero h1{ font-family:'Plus Jakarta Sans',sans-serif; font-size:3.8rem; font-weight:800;
  margin:0; letter-spacing:-2px; line-height:1;
  background:linear-gradient(96deg,#7CF0CE 0%,var(--green) 42%,var(--amber) 130%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  filter:drop-shadow(0 6px 30px rgba(45,212,168,.28)); }
.hero p{ color:var(--muted); font-size:1.12rem; margin:12px 0 2px 0; letter-spacing:.2px; }
.taghint{ color:var(--muted); font-size:.82rem; margin:16px 0 4px 2px; }

/* section + card headings, with an accent bar */
.sec-title{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.24rem;
  color:var(--text); margin:2px 0 2px 0; display:flex; align-items:center; gap:11px;
  letter-spacing:-.3px; }
.sec-title::before{ content:""; width:4px; height:19px; border-radius:3px; flex:none;
  background:linear-gradient(180deg,var(--green),var(--amber)); box-shadow:0 0 12px rgba(45,212,168,.5); }
.lead{ font-size:1.03rem; line-height:1.62; color:var(--text); margin:8px 0 12px 0; }
.body{ font-size:.95rem; line-height:1.7; color:var(--text); margin:2px 0; }
.body b{ color:var(--green); }
.dateline{ font-size:.95rem; color:var(--muted); margin:8px 0 14px 0;
  font-family:'JetBrains Mono',monospace; }
.dateline b{ color:var(--green); font-weight:600; }
.cap{ color:var(--muted); font-size:.9rem; line-height:1.6; margin:10px 2px 0 2px; }

/* bordered containers -> glass cards with hairline accent + hover lift */
div[data-testid="stVerticalBlockBorderWrapper"]{ position:relative;
  background:linear-gradient(180deg,var(--glass-top),var(--glass-bot));
  -webkit-backdrop-filter:blur(16px) saturate(125%); backdrop-filter:blur(16px) saturate(125%);
  border:1px solid var(--hairline) !important; border-radius:18px; padding:10px 10px;
  box-shadow:0 12px 44px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.04);
  transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
div[data-testid="stVerticalBlockBorderWrapper"]:hover{ transform:translateY(-2px);
  border-color:rgba(45,212,168,.26) !important;
  box-shadow:0 18px 56px rgba(0,0,0,.55), 0 0 0 1px rgba(45,212,168,.10); }
div[data-testid="stVerticalBlockBorderWrapper"]::before{ content:""; position:absolute;
  top:0; left:20px; right:20px; height:1px;
  background:linear-gradient(90deg,transparent,rgba(45,212,168,.5),transparent); }

/* st.metric -> glass KPI cards with top accent + hover */
div[data-testid="stMetric"]{ position:relative; overflow:hidden;
  background:linear-gradient(180deg,rgba(26,35,47,.66),rgba(15,21,29,.66));
  -webkit-backdrop-filter:blur(12px); backdrop-filter:blur(12px);
  border:1px solid var(--hairline); border-radius:15px; padding:16px 18px 13px 18px;
  box-shadow:0 8px 28px rgba(0,0,0,.38);
  transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
div[data-testid="stMetric"]:hover{ transform:translateY(-3px);
  border-color:rgba(45,212,168,.24); box-shadow:0 14px 36px rgba(0,0,0,.5); }
div[data-testid="stMetric"]::after{ content:""; position:absolute; top:0; left:0; right:0;
  height:2px; background:linear-gradient(90deg,var(--green),rgba(45,212,168,0) 75%); }
div[data-testid="stMetricLabel"] p{ color:var(--muted); font-weight:500; font-size:.8rem;
  letter-spacing:.3px; text-transform:uppercase; }
div[data-testid="stMetricValue"]{ font-family:'JetBrains Mono',monospace; font-weight:600;
  font-size:1.8rem; color:var(--text); letter-spacing:-.5px; font-feature-settings:"tnum"; }

/* refresh button -> accent button */
div[data-testid="stButton"] button{ background:var(--green-tint);
  border:1px solid rgba(45,212,168,.35); color:var(--green); font-weight:600;
  border-radius:11px; transition:all .2s ease; }
div[data-testid="stButton"] button:hover{ background:rgba(45,212,168,.2);
  border-color:var(--green); transform:translateY(-1px);
  box-shadow:0 6px 18px rgba(45,212,168,.18); }

/* popover triggers -> quiet glass info pills */
div[data-testid="stPopover"] button{ background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.08); border-radius:999px; color:var(--muted);
  font-size:.8rem; padding:4px 13px; font-weight:500; transition:all .2s ease; }
div[data-testid="stPopover"] button:hover{ border-color:rgba(45,212,168,.5);
  color:var(--green); background:var(--green-tint); }

/* selectbox -> subtle glass */
div[data-baseweb="select"] > div{ background:rgba(255,255,255,.03) !important;
  border-radius:11px !important; }

/* pills */
.pill{ display:inline-block; font-size:.78rem; font-weight:600; padding:6px 13px;
  border-radius:999px; }
.pill-shariah{ color:var(--green); background:var(--green-tint);
  border:1px solid rgba(45,212,168,.35); box-shadow:0 0 16px rgba(45,212,168,.12); }
.pill-conv{ color:var(--text); background:#1E2630; border:1px solid var(--border); }
.pill-unc{ color:var(--muted); background:#1A212A; border:1px dashed var(--border); }
.side-h{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.0rem;
  color:var(--text); margin-bottom:4px; }
section[data-testid="stSidebar"]{ background:#0E141B; border-right:1px solid var(--border); }
</style>
"""

PILL = {
    "shariah": '<span class="pill pill-shariah">● Shariah-Compliant</span>',
    "conventional": '<span class="pill pill-conv">● Conventional</span>',
    "unclassified": '<span class="pill pill-unc">○ Unclassified</span>',
}

# Plain-language popover copy (kept in one place for easy editing).
GLOSSARY = {
    "LSTM": "A type of AI model that's good at spotting patterns in sequences of "
            "past data, like a stock's daily prices.",
    "GRU": "A simpler, faster cousin of the LSTM. We train both and compare them "
           "to see which predicts better here.",
    "MCD": "Instead of one guess, the model makes ~200 slightly different guesses "
           "and reports how much they agree — tighter agreement means more confident.",
    "VAR": "Value at Risk: on a bad day, how much this stock has historically "
           "dropped — our figure is its worst 5% of trading days.",
    "FORECAST": "The forecast isn't one certain number. The model checks itself "
                "~200 times, and the range shows how much those checks agree — "
                "tighter means more confident.",
    "SIGNAL": "We only flag BUY or SELL when the predicted move is bigger than this "
              "stock's typical daily wiggle; otherwise we say HOLD.",
    "DIRACC": "Directional accuracy: how often the model correctly calls tomorrow "
              "as up vs. down. 50% is a coin flip.",
    "PVAL": "p-value: the chance you'd see this result by luck alone. A high p-value "
            "(here ~0.6) means we can't rule out that it's just guessing on direction.",
    "SHARPE": "Sharpe ratio: return earned for the amount of risk taken — higher is "
              "better, and it lets you compare strategies fairly even if one is riskier.",
}


# --------------------------------------------------------------------------- #
# Cached loaders (data = st.cache_data, model = st.cache_resource)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_data(DATASET_PATH)


@st.cache_data(show_spinner=False)
def get_results() -> dict:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_windows() -> dict:
    return load_windows()


@st.cache_data(show_spinner=False)
def get_robustness() -> dict:
    with open(OUTPUTS_DIR / "robustness_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_sentiment_results() -> dict:
    p = OUTPUTS_DIR / "sentiment_results.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_ablation_results() -> dict:
    p = OUTPUTS_DIR / "feature_ablation_results.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_calibration_results() -> dict:
    p = OUTPUTS_DIR / "calibration_results.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_regime_results() -> dict:
    p = OUTPUTS_DIR / "regime_analysis_results.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_meta(window: int) -> pd.DataFrame:
    return reconstruct_test_metadata(get_dataset(), window)


@st.cache_resource(show_spinner=False)
def get_model():
    model, _ = load_model(CHECKPOINTS_DIR / "lstm.pt")
    return model


@st.cache_data(show_spinner=False)
def get_forecast(ticker: str) -> dict:
    """One-day-ahead MC-dropout forecast from the ticker's latest test window."""
    w = get_windows()
    meta = get_meta(w["window"])
    idx = np.where(meta["ticker"].to_numpy() == ticker)[0]
    last = int(idx[-1])  # most recent window for this ticker

    x = w["X_test"][last][None]                       # (1, window, n_features)
    tid = torch.tensor([int(w["id_test"][last])])     # (1,)

    torch.manual_seed(0)                              # stable demo forecast
    mean, lower, upper, std = mc_dropout_predict(
        get_model(), torch.from_numpy(x), tid, n_passes=N_PASSES, ci=0.95)
    mean, lower, upper, std = (float(mean), float(lower), float(upper), float(std))

    row = meta.iloc[last]
    last_price = float(row["adj_close"])
    return {
        "last_price": last_price,
        "last_move_pct": (exp(float(row["last_return"])) - 1) * 100,
        "last_date": pd.Timestamp(row["endpoint_date"]).strftime("%b %d, %Y"),
        "forecast_date": pd.Timestamp(row["realization_date"]).strftime("%b %d, %Y"),
        "pred_ret": mean,
        "pred_ret_pct": (exp(mean) - 1) * 100,
        "pred_price": last_price * exp(mean),
        "lower_price": last_price * exp(lower),
        "upper_price": last_price * exp(upper),
        "std_pct": std * 100,
    }


def history_series(ticker: str, window: int):
    meta = get_meta(window)
    sub = meta[meta["ticker"] == ticker].tail(N_HISTORY)
    return (pd.to_datetime(sub["endpoint_date"]).to_numpy(),
            sub["adj_close"].to_numpy())


def _train_scaler(ticker: str):
    """Reproduce train.py's per-ticker StandardScaler (fit on the TRAIN portion).

    The fitted scaler was never persisted, so to scale a fresh live window the
    SAME way the model was trained, we re-fit it on this ticker's training rows
    from the frozen dataset.csv — identical data and split fraction as training.
    """
    g = get_dataset()
    g = g[g["ticker"] == ticker].sort_values("Date")
    train_end = int(SPLIT[0] * len(g))
    scaler = StandardScaler().fit(g[FEATURE_COLS].iloc[:train_end].to_numpy())
    ticker_id = int(g["ticker_id"].iloc[0])
    return scaler, ticker_id


def fetch_live_forecast(ticker: str) -> dict:
    """Live one-session-ahead forecast: pull fresh daily bars from yfinance,
    build + scale the latest window exactly as in training, run MC dropout.

    Raises on network/empty/insufficient data so the caller can fall back to the
    frozen snapshot. No retraining — inference only.
    """
    window = get_windows()["window"]
    scaler, ticker_id = _train_scaler(ticker)

    raw = yf.download(ticker, period=LIVE_PERIOD, interval="1d",
                      auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise ValueError("yfinance returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index().dropna(subset=["Adj Close"]).sort_values("Date")
    raw["log_return"] = np.log(raw["Adj Close"] / raw["Adj Close"].shift(1))
    raw = raw.dropna(subset=["log_return"]).reset_index(drop=True)
    if len(raw) < window:
        raise ValueError("not enough recent data to build a window")

    win = raw.iloc[-window:]
    feats = scaler.transform(win[FEATURE_COLS].to_numpy()).astype(np.float32)

    torch.manual_seed(0)
    mean, lower, upper, std = mc_dropout_predict(
        get_model(), torch.from_numpy(feats[None]), torch.tensor([ticker_id]),
        n_passes=N_PASSES, ci=0.95)
    mean, lower, upper, std = (float(mean), float(lower), float(upper), float(std))

    last_price = float(win["Adj Close"].iloc[-1])
    last_ts = pd.Timestamp(win["Date"].iloc[-1])
    fcast_ts = last_ts + pd.tseries.offsets.BDay(1)  # next business day (holidays approx.)
    hist = raw.iloc[-N_HISTORY:]
    return {
        "mode": "LIVE",
        "last_price": last_price,
        "last_move_pct": (exp(float(win["log_return"].iloc[-1])) - 1) * 100,
        "last_date": last_ts.strftime("%b %d, %Y"),
        "forecast_date": fcast_ts.strftime("%b %d, %Y"),
        "pred_ret": mean,
        "pred_ret_pct": (exp(mean) - 1) * 100,
        "pred_price": last_price * exp(mean),
        "lower_price": last_price * exp(lower),
        "upper_price": last_price * exp(upper),
        "std_pct": std * 100,
        "hist_dates": pd.to_datetime(hist["Date"]).to_numpy(),
        "hist_prices": hist["Adj Close"].to_numpy(),
        "last_ts": last_ts,
        "forecast_ts": fcast_ts,
    }


@st.cache_data(ttl=1200, show_spinner="Fetching latest market data…")
def live_forecast_cached(ticker: str) -> dict:
    """Auto-refresh wrapper: caches the live forecast ~20 min so an on-load
    fetch doesn't hit the network on every rerun. The Refresh button clears it."""
    return fetch_live_forecast(ticker)


@st.cache_resource(show_spinner=False)
def get_finbert():
    """Load FinBERT once per session. Returns None if transformers unavailable."""
    if not _SENTIMENT_AVAILABLE:
        return None
    try:
        return load_finbert()
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner="Fetching live news…")
def live_sentiment_cached(ticker: str) -> dict:
    """Recent-headline FinBERT vs VADER sentiment for a ticker (cached ~30 min)."""
    if not _SENTIMENT_AVAILABLE:
        raise RuntimeError("sentiment module unavailable")
    return live_sentiment(ticker, finbert=get_finbert())


def all_tickers_summary(results: dict) -> list:
    """Snapshot signal summary for every ticker using frozen test forecasts."""
    rows = []
    for t in TICKERS:
        fc_t = get_forecast(t)
        thr = results["backtest_per_ticker"][t]["threshold"]
        var = results["var"][t]["var_95"]
        if fc_t["pred_ret"] > thr:
            sig = "BUY"
        elif fc_t["pred_ret"] < -thr:
            sig = "SELL"
        else:
            sig = "HOLD"
        rows.append({
            "ticker": t,
            "shariah": SHARIAH_LABELS.get(t, "unclassified"),
            "price": fc_t["last_price"],
            "pred_pct": fc_t["pred_ret_pct"],
            "signal": sig,
            "var_95": var,
        })
    return rows


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Quantis", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>Quantis</h1>
      <p>AI-Powered Return Forecasting with Quantified Uncertainty</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero info-pills: each opens a plain-language explanation.
st.markdown('<div class="taghint">New to these terms? Tap to learn more 👇</div>',
            unsafe_allow_html=True)
t1, t2, t3, t4, _ = st.columns([1, 1, 2, 2, 3])
with t1:
    with st.popover("LSTM  ⓘ"):
        st.markdown(GLOSSARY["LSTM"])
with t2:
    with st.popover("GRU  ⓘ"):
        st.markdown(GLOSSARY["GRU"])
with t3:
    with st.popover("Monte Carlo Dropout  ⓘ"):
        st.markdown(GLOSSARY["MCD"])
with t4:
    with st.popover("Historical VaR  ⓘ"):
        st.markdown(GLOSSARY["VAR"])

# --- Stock selector + live refresh (on the main page, always visible) ------- #
sel, badge, refresh_col = st.columns([1.5, 1.7, 1.6])
with sel:
    ticker = st.selectbox("Choose a stock", TICKERS, index=0)
with badge:
    status = SHARIAH_LABELS.get(ticker, "unclassified")
    st.markdown(f'<div style="margin-top:1.95rem">{PILL[status]}</div>',
                unsafe_allow_html=True)
with refresh_col:
    st.markdown('<div style="margin-top:1.55rem"></div>', unsafe_allow_html=True)
    refresh = st.button("🔄 Refresh now")
if refresh:
    live_forecast_cached.clear()  # force a fresh pull on this run

# --- Load for selected ticker (auto-live on load, snapshot fallback) -------- #
results = get_results()
try:
    fc, mode = live_forecast_cached(ticker), "LIVE"
except Exception as exc:  # network / empty / insufficient data
    fc, mode = get_forecast(ticker), "SNAPSHOT"
    if refresh:
        st.warning(f"Couldn't fetch live data ({exc}). Showing the saved snapshot.")
var_95 = results["var"][ticker]["var_95"]
threshold = results["backtest_per_ticker"][ticker]["threshold"]

if mode == "LIVE":
    st.caption(f"🟢 LIVE — auto-updated from yfinance (cached ~20 min) · latest close "
               f"{fc['last_date']}. Today's bar may be partial during market hours; "
               "next-session date approximate around holidays. Not investment advice.")
else:
    st.caption("⚪ Live data unavailable — showing saved snapshot. "
               "Educational demo, not investment advice.")

if fc["pred_ret"] > threshold:
    signal, sig_delta, sig_color = "BUY", "▲ long", "normal"
elif fc["pred_ret"] < -threshold:
    signal, sig_delta, sig_color = "SELL", "▼ avoid", "inverse"
else:
    signal, sig_delta, sig_color = "HOLD", "— flat", "off"

# --- Date line -------------------------------------------------------------- #
st.markdown(
    f'<div class="dateline">Last close: <b>{fc["last_date"]}</b> &nbsp;→&nbsp; '
    f'Forecasting: <b>{fc["forecast_date"]}</b> &nbsp;·&nbsp; {ticker}</div>',
    unsafe_allow_html=True,
)

# --- KPI row ---------------------------------------------------------------- #
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Current Price", f"${fc['last_price']:,.2f}", f"{fc['last_move_pct']:+.2f}%")
with k2:
    st.metric("Forecast (next day)", f"${fc['pred_price']:,.2f}", f"{fc['pred_ret_pct']:+.2f}%")
    with st.popover("ⓘ How sure is it?"):
        st.markdown(GLOSSARY["FORECAST"])
with k3:
    st.metric("Signal", signal, sig_delta, delta_color=sig_color)
    with st.popover("ⓘ How is this decided?"):
        st.markdown(GLOSSARY["SIGNAL"])
with k4:
    st.metric("VaR · 95% (1d)", f"-{var_95 * 100:.2f}%", "historical", delta_color="off")
    with st.popover("ⓘ What's VaR?"):
        st.markdown(GLOSSARY["VAR"])

# --- Main chart card -------------------------------------------------------- #
window = get_windows()["window"]
if mode == "LIVE":
    hist_x, hist_y = fc["hist_dates"], fc["hist_prices"]
    fcast_x = fc["forecast_ts"]
else:
    hist_x, hist_y = history_series(ticker, window)
    meta = get_meta(window)
    fcast_x = pd.Timestamp(meta[meta["ticker"] == ticker].iloc[-1]["realization_date"])
last_x = pd.Timestamp(hist_x[-1])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_x, y=hist_y, mode="lines", name="Price history",
    line=dict(color=GREEN, width=2.2),
    hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>"))
# dotted connector last close -> forecast
fig.add_trace(go.Scatter(
    x=[last_x, fcast_x], y=[fc["last_price"], fc["pred_price"]], mode="lines",
    line=dict(color=AMBER, width=1.5, dash="dot"), showlegend=False, hoverinfo="skip"))
# shaded uncertainty ribbon: triangular fan from last close to forecast range
fig.add_trace(go.Scatter(
    x=[last_x, fcast_x, fcast_x, last_x],
    y=[fc["last_price"], fc["upper_price"], fc["lower_price"], fc["last_price"]],
    fill="toself", fillcolor="rgba(224,163,62,0.13)",
    line=dict(color="rgba(224,163,62,0.30)", width=0.8),
    name="95% uncertainty band", hoverinfo="skip"))
# forecast point (MC-dropout mean)
fig.add_trace(go.Scatter(
    x=[fcast_x], y=[fc["pred_price"]], mode="markers", name="Forecast (next session)",
    marker=dict(color=AMBER, size=11, line=dict(color=CARD_BG, width=1.5)),
    hovertemplate=("Forecast %{x|%b %d, %Y}<br>Mean: $%{y:.2f}<br>"
                   "95%% band: $" + f"{fc['lower_price']:.2f} – ${fc['upper_price']:.2f}"
                   "<extra></extra>")))

fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=TEXT), height=470,
    margin=dict(l=12, r=12, t=90, b=12),
    title=dict(text=f"{ticker} — Price & One-Day-Ahead Forecast",
               font=dict(family="Plus Jakarta Sans", size=18, color=TEXT)),
    legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom",
                bgcolor="rgba(0,0,0,0)", itemclick=False, itemdoubleclick=False),
    hovermode="x unified",
    xaxis=dict(gridcolor=GRID, showspikes=True, spikecolor=GREEN,
               spikethickness=1, spikemode="across"),
    yaxis=dict(gridcolor=GRID, title="Adj Close (USD)", zeroline=False))

with st.container(border=True):
    st.plotly_chart(fig, width="stretch",
                    config={"displaylogo": False, "scrollZoom": True})
    cap, btn = st.columns([5, 1])
    with cap:
        st.markdown(
            '<div class="cap">The amber dot is the predicted price for the next session; '
            'the shaded ribbon fans out from the last close and shows the model\'s 95% '
            'uncertainty band — a narrower ribbon means more confident, wider means less. '
            'Note: MC-dropout captures model uncertainty, not full market randomness.</div>',
            unsafe_allow_html=True)
    with btn:
        with st.popover("ⓘ About this range"):
            st.markdown(GLOSSARY["MCD"])

# --- All-stocks comparison view --------------------------------------------- #
with st.container(border=True):
    st.markdown('<div class="sec-title">Market Overview — All Stocks</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="body" style="margin-bottom:10px">Snapshot signals from the test-set '
        'forecasts for every stock in the portfolio. Switch the ticker above to see the '
        'live chart and news for any row.</div>',
        unsafe_allow_html=True)

    summary = all_tickers_summary(results)
    _sig_style = {
        "BUY":  "background:rgba(45,212,168,.15);color:#2DD4A8;border:1px solid rgba(45,212,168,.4);",
        "SELL": "background:rgba(241,102,110,.15);color:#F1666E;border:1px solid rgba(241,102,110,.4);",
        "HOLD": "background:rgba(139,152,166,.10);color:#8B98A6;border:1px solid rgba(139,152,166,.3);",
    }
    _shr_badge = {
        "shariah":       '<span style="color:#2DD4A8;font-size:.75rem">● Shariah</span>',
        "conventional":  '<span style="color:#8B98A6;font-size:.75rem">○ Conv.</span>',
        "unclassified":  '<span style="color:#8B98A6;font-size:.75rem">○ —</span>',
    }
    _rows_html = ""
    for _row in summary:
        _pct = _row["pred_pct"]
        _pc  = "#2DD4A8" if _pct > 0 else ("#F1666E" if _pct < 0 else "#8B98A6")
        _sig = _row["signal"]
        _rows_html += (
            f'<tr style="border-top:1px solid rgba(255,255,255,.05)">'
            f'<td style="padding:9px 14px;font-weight:600;color:#E8EDF2;font-size:.9rem">{_row["ticker"]}</td>'
            f'<td style="padding:9px 14px;text-align:center">{_shr_badge[_row["shariah"]]}</td>'
            f'<td style="padding:9px 14px;text-align:right;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.86rem;color:#E8EDF2">${_row["price"]:,.2f}</td>'
            f'<td style="padding:9px 14px;text-align:center">'
            f'<span style="padding:3px 11px;border-radius:999px;font-size:.76rem;font-weight:700;'
            f'{_sig_style[_sig]}">{_sig}</span></td>'
            f'<td style="padding:9px 14px;text-align:right;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.86rem;color:{_pc}">{_pct:+.2f}%</td>'
            f'<td style="padding:9px 14px;text-align:right;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.86rem;color:#8B98A6">{_row["var_95"] * 100:.2f}%</td>'
            f'</tr>'
        )
    _th = ("text-align:{a};padding:5px 14px;color:#93A1B0;font-size:.70rem;"
           "text-transform:uppercase;letter-spacing:.5px;font-weight:500")
    st.markdown(
        f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;'
        f'font-family:\'Inter\',sans-serif"><thead><tr>'
        f'<th style="{_th.format(a="left")}">Ticker</th>'
        f'<th style="{_th.format(a="center")}">Status</th>'
        f'<th style="{_th.format(a="right")}">Last Price</th>'
        f'<th style="{_th.format(a="center")}">Signal</th>'
        f'<th style="{_th.format(a="right")}">Predicted Move</th>'
        f'<th style="{_th.format(a="right")}">VaR 95%</th>'
        f'</tr></thead><tbody>{_rows_html}</tbody></table></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="cap" style="margin-top:10px">Signals use the test-set snapshot (not '
        'live data). BUY/SELL only when predicted move exceeds that stock\'s noise '
        'threshold; otherwise HOLD. VaR is the worst 5% of historical daily returns.</div>',
        unsafe_allow_html=True)

# --- Live News Sentiment card (FinBERT vs VADER) ---------------------------- #
with st.container(border=True):
    st.markdown('<div class="sec-title">Live News Sentiment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Recent headlines scored by <b>FinBERT</b> (a transformer '
                'fine-tuned on financial text) vs <b>VADER</b> (a general-purpose lexicon). '
                'They often disagree — FinBERT tends to read market news more cautiously.</div>',
                unsafe_allow_html=True)
    try:
        ls = live_sentiment_cached(ticker)
    except Exception:
        ls = None

    if ls and ls["n"] > 0:
        def _sent_meta(score):
            if score > 0.05:
                return "bullish", "normal"
            if score < -0.05:
                return "bearish", "inverse"
            return "neutral", "off"

        fb_lbl, fb_col = _sent_meta(ls["finbert_mean"])
        vd_lbl, vd_col = _sent_meta(ls["vader_mean"])
        s1, s2, s3 = st.columns(3)
        s1.metric("FinBERT", f"{ls['finbert_mean']:+.2f}", fb_lbl, delta_color=fb_col)
        s2.metric("VADER", f"{ls['vader_mean']:+.2f}", vd_lbl, delta_color=vd_col)
        s3.metric("Model agreement", f"{ls['agreement']:.0%}",
                  f"{ls['n']} headlines", delta_color="off")

        def _chip(v):
            c = "var(--up)" if v > 0.05 else ("var(--down)" if v < -0.05 else "var(--neutral)")
            return f'<span style="color:{c};font-weight:600">{v:+.2f}</span>'

        items = "".join(
            f'<div style="margin:8px 0;font-size:.92rem;color:var(--text)">{h["title"]}'
            f'<br><span style="color:var(--muted);font-size:.8rem">FinBERT {_chip(h["finbert"])}'
            f' &nbsp;·&nbsp; VADER {_chip(h["vader"])}'
            f' &nbsp;·&nbsp; {h.get("source") or ""}</span></div>'
            for h in ls["headlines"][:5]
        )
        st.markdown('<div class="cap">Most recent headlines:</div>', unsafe_allow_html=True)
        st.markdown(items, unsafe_allow_html=True)
    else:
        st.caption("Live sentiment unavailable right now (no recent headlines, or the "
                   "news API is unreachable / rate-limited).")

# --- Model Transparency card ------------------------------------------------ #
m = results["metrics"]["LSTM"]
pb = results["portfolio_backtest"]

with st.container(border=True):
    st.markdown('<div class="sec-title">Model Transparency</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lead">In plain English: Quantis is good at estimating '
        '<b>how big</b> a price move might be and giving a range of likely outcomes — '
        'but, like most short-term forecasts, it <b>can\'t reliably predict which '
        'direction</b> prices will move tomorrow. We\'d rather show that honestly '
        'than pretend otherwise.</div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div class="body">• <b>Direction:</b> it calls tomorrow\'s up/down correctly '
        f'<b>{m["directional_acc"] * 100:.1f}%</b> of the time — about a coin flip — and '
        f'we tested it (<b>p = {m["binom_p_value"]:.2f}</b>): not better than chance.</div>',
        unsafe_allow_html=True)
    d1, d2, _ = st.columns([1.5, 1.1, 3])
    with d1:
        with st.popover("ⓘ Directional accuracy"):
            st.markdown(GLOSSARY["DIRACC"])
    with d2:
        with st.popover("ⓘ p-value"):
            st.markdown(GLOSSARY["PVAL"])

    st.markdown(
        '<div class="body">• <b>Where it shines:</b> estimating the size of a move and '
        'giving a range of likely outcomes, shown on the chart above.</div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div class="body">• <b>Backtest:</b> trading on its signals over the test '
        f'window returned <b>{pb["cum_strat"] * 100:+.1f}%</b> (Sharpe '
        f'{pb["sharpe_strat"]:.2f}) vs. <b>{pb["cum_bh"] * 100:+.1f}%</b> just holding '
        f'(Sharpe {pb["sharpe_bh"]:.2f}). Without a reliable direction call, simply '
        f'holding won — and we report that plainly.</div>',
        unsafe_allow_html=True)
    s1, _ = st.columns([1.3, 4])
    with s1:
        with st.popover("ⓘ Sharpe ratio"):
            st.markdown(GLOSSARY["SHARPE"])

# --- Research Findings expander --------------------------------------------- #
with st.expander("📊 Research Findings — What the experiments showed", expanded=False):
    st.markdown(
        '<div class="body" style="margin-bottom:14px">These findings come from separate '
        'experiments run outside the live demo. They are reproducible and form the '
        'academic core of the project.</div>',
        unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Model Stability (5 seeds)**")
        _rob = get_robustness()
        _r_agg = _rob.get("aggregate", {})
        if _r_agg:
            _l = _r_agg["LSTM"]
            _g = _r_agg["GRU"]
            st.markdown(
                f"LSTM RMSE: `{_l['rmse_mean']:.5f}` ± `{_l['rmse_std']:.2e}`  \n"
                f"GRU RMSE:  `{_g['rmse_mean']:.5f}` ± `{_g['rmse_std']:.2e}`")
        st.markdown(
            "Both models are consistent across 5 random seeds. The variation is smaller "
            "than the 5th decimal place, so results do not depend on initialization luck.")

    with col_b:
        st.markdown("**Does news sentiment help?**")
        _sent = get_sentiment_results()
        if _sent:
            _sdm = _sent.get("diebold_mariano_vs_baseline", {})
            _vp  = _sdm.get("+ VADER",   {}).get("p_value", None)
            _fp  = _sdm.get("+ FinBERT", {}).get("p_value", None)
            if _vp is not None:
                st.markdown(
                    f"VADER p = `{_vp:.3f}` — not significant  \n"
                    f"FinBERT p = `{_fp:.3f}` — not significant")
        st.markdown(
            "Tested pooled across 5 stocks × 5 seeds with a Diebold-Mariano test. "
            "Adding daily news sentiment scores does not significantly improve forecasting. "
            "A null result is a genuine finding, not a failure.")

    with col_c:
        st.markdown("**Feature ablation (RSI, SMA, momentum)**")
        _abl = get_ablation_results()
        if _abl:
            _base_r = _abl["aggregate"]["baseline"]["rmse_mean"]
            _indic_r = _abl["aggregate"]["+ indicators"]["rmse_mean"]
            _delta_pct = (_indic_r - _base_r) / _base_r * 100
            _dm_p = _abl["diebold_mariano_vs_baseline"]["p_value"]
            st.markdown(
                f"Δ RMSE = `{_delta_pct:+.3f}%`  \n"
                f"DM p = `{_dm_p:.3f}` (significant at 5%)")
        st.markdown(
            "Statistically significant difference — but the effect is ~0.003%. "
            "A textbook illustration of **statistical significance ≠ practical significance**.")

    st.markdown("---")
    col_d, col_e = st.columns(2)

    with col_d:
        st.markdown("**MC-Dropout Calibration (200 passes, full test set)**")
        _cal = get_calibration_results()
        if _cal:
            _cov = _cal["empirical_coverage"]
            st.markdown(
                f"Empirical coverage: `{_cov:.1%}` &nbsp;(nominal 95%)\n\n"
                f"The 95% band actually contains only **{_cov:.1%}** of realized returns. "
                "MC-dropout measures *parameter* uncertainty — how much the model's "
                "weights could vary — but not the irreducible randomness in financial "
                "markets. So the bands look tight but do not reflect true market risk. "
                "This is a genuine limitation of the approach, and reporting it honestly "
                "is more rigorous than ignoring it.",
                unsafe_allow_html=True)

    with col_e:
        st.markdown("**Regime analysis: trending vs choppy markets**")
        _reg = get_regime_results()
        if _reg and "trending" in _reg and "choppy" in _reg:
            _t_acc = _reg["trending"]["directional_acc"]
            _c_acc = _reg["choppy"]["directional_acc"]
            _t_n   = _reg["trending"]["n"]
            _c_n   = _reg["choppy"]["n"]
            st.markdown(
                f"Trending markets: `{_t_acc:.1%}` (n={_t_n})  \n"
                f"Choppy markets:   `{_c_acc:.1%}` (n={_c_n})")
        st.markdown(
            "Splitting the test set by market regime (20-day rolling return vs its σ) "
            "shows no meaningful difference in directional accuracy. The model does not "
            "improve in trending conditions — the coin-flip nature is regime-agnostic.")
