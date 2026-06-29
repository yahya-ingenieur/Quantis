# Quantis — AI Financial Intelligence Platform
<p align="center">
  <a href="https://quantis-ai-financial-intelligence-platform.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀_Live_App-00E676?style=for-the-badge&logo=streamlit&logoColor=black" alt="Launch Quantis Platform" width="350">
  </a>
</p>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

Quantis is a complete end-to-end deep learning system for next-day stock return forecasting. It trains LSTM and GRU models on 8 US equities, quantifies prediction uncertainty using Monte Carlo dropout, and delivers a live interactive dashboard that surfaces forecasts, news sentiment, risk metrics, and honest model transparency.

The project was built as a serious academic exercise — every methodological decision is documented, every result is reproducible across multiple random seeds, and the evaluation deliberately includes cases where the model *fails to beat baselines*, reported honestly rather than papered over.

---

## Table of Contents

1. [What Quantis Does](#what-quantis-does)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Training Pipeline](#training-pipeline)
5. [Experiments](#experiments)
6. [Results](#results)
7. [Live Dashboard](#live-dashboard)
8. [File Structure](#file-structure)
9. [Setup](#setup)
10. [Running the Pipeline](#running-the-pipeline)
11. [Academic Notes & Limitations](#academic-notes--limitations)

---

## What Quantis Does

At a high level, Quantis answers a single question every trading day for each of 8 stocks: *"Given the past 10 sessions of price and volume, what is the probability distribution over tomorrow's return?"*

It answers this not with a single point estimate, but with:

- A **mean predicted log-return** (converted to a predicted price)
- A **95% Monte Carlo confidence interval** showing how much the model's individual forward passes disagree
- A **trading signal** (BUY / SELL / HOLD) that only fires when the predicted move exceeds the stock's own historical noise floor — avoiding spurious signals on coin-flip predictions
- A **live news sentiment score** from FinBERT and VADER on the most recent headlines

These are all displayed in a real-time Streamlit dashboard with a Bloomberg-inspired dark UI, live data pulled from yfinance, and an honest "Model Transparency" card that explicitly tells users where the model fails.

---

## Architecture

### Models: LSTM and GRU

Both models share an identical interface and are trained and evaluated in parallel, enabling a direct apples-to-apples comparison.

```
Input window (10 days × 2 features)
        ↓
  LSTM / GRU layers (hidden=128, dropout=0.3)
        ↓
  Final hidden state  ←──  Ticker embedding (dim=8)
        ↓
  Concatenated vector  →  Linear output head  →  predicted log-return (scalar)
```

**Key design decisions:**

**Ticker embedding (`nn.Embedding`):** Each of the 8 stocks is assigned a learnable 8-dimensional embedding vector. This is concatenated with the LSTM/GRU final hidden state before the output head. The rationale: a single model that shares temporal pattern-detection weights across all stocks, but learns a per-stock "personality" vector — capturing that TSLA is a fundamentally different asset from JNJ even if their return sequences look structurally similar. This is substantially more principled than training 8 independent models or ignoring the stock identity altogether.

**Dropout at inference (MC-Dropout):** During the live demo, dropout is *kept active* at inference time. The model runs 200 forward passes through the same input with different dropout masks each time, producing a distribution of predictions rather than a single number. The mean of this distribution is the point forecast; the 2.5th–97.5th percentiles form the 95% confidence band. This is a Bayesian approximation: the disagreement between passes reflects the model's epistemic uncertainty about the true parameters.

**No weight tying between LSTM and GRU:** Both models are trained independently from scratch with the same hyperparameters. Comparing their outputs gives a direct measure of how much the architectural choice of gating mechanism matters.

### Windowing and Feature Engineering

Input features: **log-return** and **Volume** (the minimal informative set — anything more risks spurious correlation on small data).

The lookback window of 10 trading days was selected through a joint hyperparameter search over three configurations (varying window, hidden size, and dropout together). The model sees:

```
X[t] = StandardScaler( [log_return, Volume] )_{t-9 : t}   shape: (10, 2)
y[t] = log_return_{t+1}                                     shape: (1,)
```

The scaler is fit **only on the training portion** of each ticker's data and applied without refitting to validation and test — preventing the subtle form of data leakage where future distribution knowledge seeps into feature scaling.

### Hyperparameter Search

Three joint configurations were explored (window, hidden size, and dropout varied together):

| Config | Window | Hidden size | Dropout | LR | Best val loss |
|--------|--------|-------------|---------|-----|---------------|
| 0 | 20 | 64 | 0.2 | 1e-3 | 3.766e-4 |
| 1 | 30 | 64 | 0.2 | 1e-3 | 3.689e-4 |
| **2 (winner)** | **10** | **128** | **0.3** | **1e-3** | **3.663e-4** |

**Winner:** Config 2 (window=10, hidden=128, dropout=0.3, lr=1e-3). Selected by lowest validation loss — no test-set information used in this selection.

---

## Dataset

| Property | Value |
|----------|-------|
| Stocks | AAPL, META, MSFT, TSLA, AMZN, XOM, JNJ, JPM |
| Source | Yahoo Finance via yfinance (Adj Close, Volume) |
| Frequency | Daily OHLCV |
| Total rows | 20,096 (after cleaning) |
| Date range | 2016-06-21 – 2026-06-17 |
| Split | 70% train / 15% val / 15% test (chronological, no shuffle) |
| Test windows per ticker | 367 |
| Total test windows | 2,936 |

**Cleaning policy:** Rows with missing `Adj Close` are dropped entirely. No forward-fill, no interpolation, no synthetic gap-filling. The rationale: forward-filling fabricates artificial zero-return days, which would depress RMSE without corresponding to any real market information. The cost (slightly fewer training samples) is worth the integrity.

**Shariah classification:**
- Shariah-compliant: AAPL, META (per current DJIM US factsheet)
- Conventional: JPM (financial sector, interest-based operations)
- Unclassified: MSFT, TSLA, AMZN, XOM, JNJ (status not verified against a current screening methodology — labelled unclassified rather than asserting compliance we have not confirmed)

---

## Training Pipeline

### Chronological split

The 70/15/15 split is applied **per ticker** to each stock's individual time series, then the resulting windows are pooled. This prevents any temporal leakage — the model never sees data from a period that would be in another ticker's test set. The alternative (pooling first then splitting) would allow 2024 data from one stock to appear in the training set while 2022 data from another is in the test set — a subtle but real form of information leakage.

### n-2 label boundary

The test chunk for each ticker has length `cn`. Test windows are built for `j ∈ [window-1, cn-2]` — stopping one step before the end. The final row (`j = cn-1`) would require `target = chunk[cn]`, which doesn't exist. Without this cap, the last window would have a NaN target or wrap into the next split. The n-2 boundary is applied identically in both training and evaluation to ensure the test arrays align.

### Gradient clipping

All gradients are clipped at `max_norm = 0.1`. At this threshold, approximately 14% of training steps trigger the clip (gradient norms were profiled: median ~0.02, occasional spikes to 0.65). This prevents rare runaway updates in the later layers from destabilising the embedding table, which is shared across all tickers and therefore receives gradients from every batch.

### Loss and optimisation

- Loss: Mean Squared Error on the predicted vs realised log-return
- Optimiser: Adam (lr=1e-3, betas=(0.9, 0.999))
- Batch size: 64 (shuffled, within-split only — never mixing train/val samples)
- Epochs: 30 with early stopping on validation loss

### Baselines

Two baselines are trained on identical windows to establish a performance floor:

- **Persistence:** predict tomorrow's return = today's return (naive, no parameters)
- **Linear regression:** `sklearn.LinearRegression` fit on the flattened (10 × 2) window, trained on the training split only

The LSTM/GRU must beat these to demonstrate that the added complexity is warranted.

---

## Experiments

Quantis goes beyond single-run evaluation with five independent experiments, each addressing a distinct question.

### 1. Multi-Seed Robustness

**Question:** Are the results reproducible, or do they depend on a lucky random seed?

**Method:** Train both LSTM and GRU across 5 independent seeds (42, 0, 1, 2, 3), report mean ± std across seeds.

**Why this matters:** A single-seed result can look impressive purely by chance. Publishing mean ± std is the minimum standard for credible deep learning evaluation.

### 2. News Sentiment Experiment

**Question:** Does adding daily news sentiment scores to the input features improve next-day return forecasting?

**Method:** Headlines from FNSPID (37,853 headlines across AAPL, AMZN, MSFT, TSLA, XOM — sourced via byte-range binary search on a 23 GB file) are scored by:
- **FinBERT** (`ProsusAI/finbert`): a transformer fine-tuned on financial text, producing positive/negative/neutral probabilities
- **VADER** (`vaderSentiment`): a rule-based lexicon calibrated for social media and news text

Daily sentiment scores per ticker are merged with the price data. Three model variants are trained and compared using a **Diebold-Mariano test** (not just a point-difference comparison): baseline, +VADER, +FinBERT.

The experiment runs pooled across 5 stocks × 5 seeds to avoid spurious single-stock, single-seed results.

### 3. Feature Ablation

**Question:** Do standard technical indicators (RSI-14, SMA-ratio, 20-day rolling volatility, 10-day momentum) meaningfully improve the model?

**Method:** Train baseline `[log_return, Volume]` vs `[log_return, Volume, rsi14, sma_ratio, roll_vol20, mom10]` across 5 seeds. Compare with a Diebold-Mariano test.

### 4. MC-Dropout Calibration

**Question:** When the model reports a 95% confidence interval, does the true value actually fall inside it 95% of the time?

**Method:** Run 200 MC-dropout passes on all 2,936 test windows. Count the fraction of true realised returns that fall within `[lower_95, upper_95]`.

This is the *empirical coverage* — the correct way to evaluate whether an uncertainty estimate is well-calibrated.

### 5. Regime-Conditional Direction Accuracy

**Question:** Does the model predict direction better during trending market periods than during choppy, sideways periods?

**Method:** Classify each test window by market regime using a 20-day rolling mean return vs 0.5× its rolling standard deviation. Test windows where `|rolling_mean| > 0.5 × rolling_std` are labeled "trending"; the rest are "choppy". Report directional accuracy and binomial p-value in each group.

---

## Results

### Core metrics (single seed, 2,936 test windows, 8 stocks)

| Model | MAE | RMSE | Dir. Acc | Binom p |
|-------|-----|------|----------|---------|
| **LSTM** | 0.01444 | **0.02135** | 50.5% | 0.592 |
| **GRU** | 0.01434 | **0.02128** | 50.4% | 0.698 |
| Linear Regression | 0.01446 | 0.02140 | 49.9% | 0.926 |
| Persistence | 0.02054 | 0.03013 | 50.3% | 0.754 |

**Interpretation:** Both LSTM and GRU meaningfully beat persistence (29% RMSE improvement) and marginally beat linear regression (0.2–0.6% improvement). The directional accuracy is statistically indistinguishable from a coin flip (p > 0.05 for all models). This is an **honest result** — the model captures return *magnitude* well but cannot reliably predict *direction* on daily data, which is consistent with the efficient market hypothesis and with every serious academic study of short-horizon equity forecasting.

### Per-ticker RMSE

| Ticker | LSTM RMSE | GRU RMSE | VaR 95% | Sector |
|--------|-----------|----------|---------|--------|
| JNJ | 0.01205 | 0.01225 | 1.73% | Healthcare |
| JPM | 0.01611 | 0.01568 | 2.36% | Finance |
| MSFT | 0.01762 | 0.01728 | 2.66% | Technology |
| AAPL | 0.01873 | 0.01859 | 3.16% | Technology |
| XOM | 0.01628 | 0.01629 | 2.69% | Energy |
| AMZN | 0.02097 | 0.02084 | 3.11% | Technology |
| META | 0.02442 | 0.02435 | 3.39% | Technology |
| **TSLA** | **0.03577** | **0.03591** | **5.30%** | Technology |

RMSE correlates almost perfectly with each stock's historical volatility. JNJ (healthcare, low volatility) is easiest; TSLA (the most volatile stock in the universe at this scale) is hardest. This is the expected and correct behaviour: a model that reported the same RMSE across all tickers would be suspicious.

### Multi-Seed Robustness

| Model | RMSE mean | RMSE std | Dir. Acc mean | Dir. Acc std |
|-------|-----------|----------|---------------|--------------|
| **LSTM** | 0.02139 | ±0.00007 | 51.1% | ±0.9% |
| **GRU** | 0.02131 | ±0.00004 | 51.4% | ±0.9% |

The GRU is slightly more stable (tighter std). Both are robust — a variation of 7e-5 in RMSE across 5 seeds is negligible. The results are not artifacts of a single lucky initialisation.

### News Sentiment Experiment (5 stocks × 5 seeds, Diebold-Mariano)

| Feature set | RMSE mean | Dir. Acc mean | DM p-value | Significant? |
|-------------|-----------|---------------|------------|-------------|
| Baseline | 0.01933 | 48.0% | — | — |
| + VADER | 0.01949 | 52.3% | 0.194 | No |
| + FinBERT | 0.01951 | 48.9% | 0.073 | No |

**Interpretation:** Adding daily news sentiment makes the model *marginally worse* on RMSE, and any improvement in directional accuracy is not statistically significant. This is a robust null result — tested across 5 stocks and 5 seeds, not a single run. The Diebold-Mariano test (which accounts for autocorrelation in forecast errors) confirms there is no significant difference in predictive accuracy.

This null result is itself academically valuable: it rules out the naive hypothesis that "more information always helps" and is consistent with efficient market theory on daily time scales.

### Feature Ablation (technical indicators)

| Feature set | RMSE mean | DM p-value | Δ RMSE |
|-------------|-----------|------------|--------|
| Baseline `[log_return, Volume]` | 0.02141 | — | — |
| + RSI14, SMA-ratio, roll_vol20, mom10 | 0.02140 | 0.019 | −0.003% |

**Interpretation:** A textbook case of **statistical significance without practical significance**. The Diebold-Mariano test finds a significant difference (p = 0.019 < 0.05), but the actual improvement is 0.003% — below any meaningful threshold. This likely reflects the DM test having high power on 2,936 samples, detecting a real but economically irrelevant signal.

### MC-Dropout Calibration

| Metric | Value |
|--------|-------|
| Nominal CI | 95% |
| **Empirical coverage** | **20.6%** |
| Coverage gap | −74.4 pp |
| Well-calibrated | No |

**Interpretation:** The model's 95% confidence band actually contains the true return only 20.6% of the time. This is a fundamental limitation of MC-dropout applied to financial data: it captures *epistemic* uncertainty (how unsure the model is about its own weights) but not *aleatoric* uncertainty (the irreducible randomness in market returns). Market noise vastly dominates parameter uncertainty for daily return prediction. The bands look tight relative to actual return variability — they should not be used as a reliable risk measure. This finding is displayed explicitly in the app's Model Transparency card.

### Regime-Conditional Direction Accuracy

| Market Regime | Dir. Acc | n samples | Binom p |
|---------------|----------|-----------|---------|
| Trending | 51.3% | 113 | 0.851 |
| Choppy | 50.5% | 2,823 | 0.624 |
| **Overall** | **50.5%** | **2,936** | **0.592** |

**Interpretation:** The model shows no advantage during trending market periods. In both regimes, directional accuracy is statistically indistinguishable from chance (p >> 0.05). The coin-flip nature of short-horizon direction prediction is regime-agnostic.

### Portfolio Backtest (equal-weight, 5 bps per trade)

| Strategy | Cumulative return | Sharpe |
|----------|-------------------|--------|
| LSTM threshold strategy | −0.46% | −0.15 |
| Buy & Hold (8-stock equal weight) | +22.0% | +0.77 |

The strategy is unprofitable net of trading costs. Buy-and-hold dominates across all evaluation metrics. This result is expected given the directional accuracy near 50%, and is reported without softening.

---

## Live Dashboard

The Streamlit app (`src/app.py`) provides a full interactive experience:

**Main chart:** 150-day price history with a Bloomberg-inspired shaded uncertainty ribbon fanning from the last close to the MC-dropout forecast. The ribbon is triangular — zero width at the last known close, widening to `[lower_price, upper_price]` at the forecast date.

**KPI row:** Current price (last session), forecast price (next session with ± % change), trading signal (BUY/SELL/HOLD with adaptive noise threshold), and 1-day 95% historical VaR.

**Market Overview table:** All 8 stocks side-by-side in a glass-card table showing signal, predicted move, Shariah status badge, and VaR.

**Live News Sentiment card:** Fetches the 5 most recent headlines from NewsAPI, scores each with both FinBERT and VADER in real time, and reports model agreement.

**Model Transparency card:** Plain-English explanation of what the model does well (magnitude estimation) and what it cannot do (direction prediction), with the exact numbers from evaluation results.

**Research Findings panel:** A collapsible expander containing all 5 experiment results with exact numbers — model stability, sentiment null result, feature ablation, MC-dropout calibration, and regime analysis.

**Technical:** Live forecasts are cached for 20 minutes (`st.cache_data(ttl=1200)`). News sentiment is cached for 30 minutes. On network failure, the app falls back to a frozen test-set snapshot. The model checkpoint is loaded once and held in `st.cache_resource`.

---

## File Structure

```
Quantis/
│
├── src/
│   ├── data.py                  # Download OHLCV from yfinance, clean, feature-engineer
│   ├── model.py                 # LSTMForecaster, GRUForecaster, mc_dropout_predict()
│   ├── train.py                 # Windowing, splitting, scaling, training, checkpointing
│   ├── evaluate.py              # Metrics, VaR, backtest, permutation importance, 11 plots
│   ├── app.py                   # Streamlit live dashboard
│   │
│   ├── sentiment.py             # FNSPID headline fetch, FinBERT+VADER daily scoring, NewsAPI live
│   ├── sentiment_experiment.py  # Pooled 5-stock 5-seed sentiment ablation + DM test
│   ├── robustness.py            # 5-seed LSTM vs GRU stability experiment
│   ├── feature_ablation.py      # Technical indicator ablation + DM test
│   ├── calibration.py           # MC-dropout empirical coverage on full test set
│   ├── regime_analysis.py       # Directional accuracy by market regime
│   └── report_plots.py          # Additional figures from existing artifacts
│
├── main.py                      # Pipeline orchestrator (--all / --only / --from flags)
│
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis (all cells executed)
│
├── outputs/
│   ├── evaluation_results.json      # All metrics, VaR, backtest numbers
│   ├── robustness_results.json      # Multi-seed LSTM vs GRU
│   ├── sentiment_results.json       # Sentiment experiment results
│   ├── feature_ablation_results.json # Technical indicator ablation
│   ├── calibration_results.json     # MC-dropout coverage
│   ├── regime_analysis_results.json # Regime-conditional accuracy
│   ├── training_history.json        # Per-epoch loss curves + grad stats
│   └── plots/                       # All generated figures (PNG, 120 DPI)
│
├── checkpoints/
│   ├── lstm.pt                  # Trained LSTM checkpoint + config + feature list
│   └── gru.pt                   # Trained GRU checkpoint
│
├── data/
│   ├── raw/                     # Raw yfinance downloads + FNSPID headlines
│   └── processed/               # dataset.csv (20,096 rows), windows.npz, sentiment.csv
│
├── .streamlit/
│   └── config.toml              # Dark theme: Inter + Plus Jakarta Sans + JetBrains Mono
│
├── requirements.txt
├── .env                         # NEWSAPI_KEY (git-ignored, never committed)
├── .gitignore
├── RESULTS.md
└── README.md
```

---

## Setup

### Requirements

- Python 3.11 (tested; 3.10+ should work)
- CPU-only is sufficient. No GPU required.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Quantis.git
cd Quantis

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment variables (optional — only needed for live news sentiment)

Create a `.env` file in the project root:

```
NEWSAPI_KEY=your_key_here
```

Get a free key at [newsapi.org](https://newsapi.org). Without this key the app still runs fully — it falls back to cached sentiment scores.

---

## Running the Pipeline

### Full pipeline (recommended for reproducibility)

```bash
python src/data.py                       # Download + clean dataset (creates data/)
python -m src.train                      # Train LSTM + GRU, run baselines
python -m src.evaluate                   # Metrics, backtest, VaR, 11 plots
python -m src.report_plots               # Additional report figures
streamlit run src/app.py                 # Launch the live dashboard
```

### Or use the orchestrator

```bash
python main.py --all                     # Run all stages in sequence
python main.py --only evaluate           # Run one specific stage
python main.py --from evaluate           # Resume from a given stage
```

### Experiments (independent, run after training)

```bash
python -m src.robustness                 # 5-seed stability (LSTM vs GRU)
python -m src.feature_ablation           # Technical indicator ablation
python -m src.calibration                # MC-dropout empirical coverage
python -m src.regime_analysis            # Regime-conditional direction accuracy
```

### News sentiment (optional, requires FNSPID download ~23 GB)

```bash
python -m src.sentiment fetch            # Fetch headlines (FNSPID + timeout guard)
python -m src.sentiment score            # FinBERT + VADER daily scoring
python -m src.sentiment_experiment       # Pooled 5-stock 5-seed comparison + DM test
```

---

## Academic Notes & Limitations

### What the model does well

- **Magnitude estimation:** RMSE is 29% better than persistence and marginally better than linear regression. The model correctly captures that large-volatility stocks (TSLA) have larger absolute errors than low-volatility stocks (JNJ).
- **Reproducibility:** Results are stable across 5 independent random seeds (RMSE std < 1e-4). The findings are not luck.
- **Honest uncertainty:** The MC-dropout distribution is correctly wider for more ambiguous inputs, even if the absolute calibration is poor.

### What the model cannot do

- **Direction prediction:** 50.5% directional accuracy — statistically indistinguishable from a coin flip. This is not a failure of implementation; it is an expected result consistent with the semi-strong efficient market hypothesis and decades of academic literature on short-horizon return prediction.
- **MC-dropout calibration:** 20.6% empirical coverage vs 95% nominal. The confidence bands reflect *model parameter uncertainty* only, not the irreducible randomness of market returns (aleatoric uncertainty). They should not be used as risk bounds.
- **News sentiment:** Adding FinBERT or VADER scores does not improve forecasting accuracy (DM p = 0.073 / 0.194). Daily headline sentiment does not appear to carry information that isn't already in the price/volume sequence at this time scale.
- **Regime sensitivity:** The model is equally uninformative in trending and choppy markets. Market regimes do not unlock latent predictive ability.

### Why we report these limitations

Selecting a different evaluation window, a different seed, or a different baseline can make almost any model look good. We deliberately chose a broad evaluation — 8 stocks, 5 seeds, multiple baselines, significance tests — precisely to expose where the model fails. A project that shows only successes is either selecting results or working on an easy problem. Neither is academically interesting.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Deep learning | PyTorch 2.x |
| Data | yfinance, pandas, numpy |
| ML baselines | scikit-learn |
| Statistical tests | scipy (Diebold-Mariano, binomial test) |
| NLP / Sentiment | HuggingFace Transformers (FinBERT), vaderSentiment |
| Visualisation (report) | matplotlib |
| Visualisation (app) | Plotly |
| Dashboard | Streamlit 1.58 |
| Fonts | Inter, Plus Jakarta Sans, JetBrains Mono (via Google Fonts) |

---

*Results are fully reproducible by running the pipeline in order on any machine with Python 3.11 and the packages in `requirements.txt`. All random seeds are fixed. No external data is required beyond what yfinance downloads automatically.*
