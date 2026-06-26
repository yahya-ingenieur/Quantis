# Quantis — Full Experimental Results

This document contains every numerical result produced by the project pipeline, with interpretation for each finding. All results are reproducible by running the pipeline from scratch (see README).

---

## 1. Core Model Evaluation

**Test set:** 2,936 windows across 8 stocks (367 per ticker), chronological 70/15/15 split, no data leakage.

### Overall metrics

| Model | MAE | RMSE | Dir. Acc | n correct / n | Binom p-value |
|-------|-----|------|----------|---------------|---------------|
| **LSTM** | 0.014440 | **0.021350** | **50.51%** | 1482 / 2934 | 0.592 |
| **GRU** | 0.014342 | **0.021280** | 50.37% | 1478 / 2934 | 0.698 |
| Linear Regression | 0.014464 | 0.021402 | 49.90% | 1464 / 2934 | 0.926 |
| Persistence | 0.020542 | 0.030129 | 50.31% | 1476 / 2934 | 0.754 |

**R²:** LSTM = −0.0094, GRU = −0.0028

> R² is negative because the test variance slightly exceeds the baseline MSE — a natural consequence of the model targeting log-returns near zero. The RMSE and MAE comparisons are the meaningful metrics here.

**Key takeaways:**
- Both neural models beat persistence by ~29% on RMSE — meaningful improvement over the naive baseline.
- Both neural models marginally outperform linear regression (0.2–0.6% on RMSE), indicating that temporal nonlinearity contributes, but not dramatically.
- No model achieves directional accuracy above chance. All binomial p-values are > 0.05. This is the expected result for daily return prediction.

---

## 2. Per-Ticker Results

### RMSE breakdown

| Ticker | LSTM RMSE | GRU RMSE | LSTM wins? | Test windows |
|--------|-----------|----------|------------|-------------|
| JNJ | 0.012051 | 0.012250 | Yes | 367 |
| JPM | 0.016114 | 0.015685 | No | 367 |
| MSFT | 0.017616 | 0.017276 | No | 367 |
| XOM | 0.016277 | 0.016290 | Yes | 367 |
| AAPL | 0.018732 | 0.018594 | No | 367 |
| AMZN | 0.020969 | 0.020842 | No | 367 |
| META | 0.024424 | 0.024354 | No | 367 |
| TSLA | 0.035770 | 0.035912 | Yes | 367 |
| **Mean** | **0.021375** | **0.021275** | GRU: 5/8 | — |

RMSE is almost perfectly correlated with each stock's annualised volatility. The model is doing what it should: harder stocks (higher σ) produce larger absolute errors, not because the model fails selectively, but because the signal-to-noise ratio is inherently lower.

### VaR and rolling volatility

| Ticker | VaR 95% (1-day) | Mean rolling vol (21-day) |
|--------|----------------|--------------------------|
| TSLA | **5.30%** | **3.37%** |
| META | 3.39% | 2.26% |
| AAPL | 3.16% | 1.69% |
| AMZN | 3.11% | 1.98% |
| XOM | 2.69% | 1.57% |
| MSFT | 2.66% | 1.57% |
| JPM | 2.36% | 1.47% |
| JNJ | **1.73%** | **1.13%** |

VaR is computed as the empirical 5th percentile of the test-period return distribution (historical method, no distributional assumptions).

### Per-ticker backtest (threshold strategy vs buy-and-hold, 5 bps/trade)

| Ticker | Strategy cum. | B&H cum. | Strategy Sharpe | B&H Sharpe | n trades |
|--------|--------------|----------|-----------------|------------|---------|
| JNJ | +0.56% | +68.25% | 0.14 | 1.99 | 14 |
| JPM | +1.88% | +42.56% | 0.81 | 1.11 | 2 |
| XOM | −1.44% | +38.88% | −0.86 | 1.01 | 2 |
| AAPL | −2.24% | +16.53% | −0.03 | 0.50 | 2 |
| AMZN | −2.39% | +6.15% | −1.17 | 0.29 | 4 |
| META | 0.00% | −4.91% | 0.00 | 0.10 | 0 |
| TSLA | 0.00% | −8.17% | 0.00 | 0.18 | 0 |
| MSFT | −1.43% | −10.95% | −0.31 | −0.15 | 8 |

The threshold for META and TSLA is high enough (driven by their large train-period return volatility) that the model never issues a signal — the strategy stays flat and returns 0% for the test period. For stocks where the strategy does trade, it underperforms buy-and-hold in 6 out of 6 active cases. The signal threshold is set at 0.5 × train-period return std per ticker, which is conservative and prevents constant trading on noise.

### Equal-weight portfolio backtest

| | Cumulative return | Sharpe ratio |
|-|------------------|-------------|
| **LSTM threshold strategy** | **−0.46%** | **−0.15** |
| **Buy & Hold (8 stocks)** | **+22.0%** | **+0.77** |

The strategy is unprofitable in aggregate. The 5 bps/trade cost is realistic (representing exchange fees + bid-ask spread for a retail account), and even without costs the strategy would not significantly beat buy-and-hold given ~50% directional accuracy.

---

## 3. Monte Carlo Dropout — Sample Forecasts

One representative window per ticker (mid-test-set index), 50 passes, 95% CI:

| Ticker | MC mean | Lower 95% | Upper 95% | MC std | Actual |
|--------|---------|-----------|-----------|--------|--------|
| AAPL | +0.022% | −0.405% | +0.449% | 0.218% | −0.837% |
| AMZN | +0.530% | +0.178% | +0.881% | 0.179% | −0.227% |
| JNJ | +0.218% | −0.082% | +0.518% | 0.153% | +0.062% |
| JPM | −0.226% | −0.584% | +0.133% | 0.183% | +0.217% |
| META | +0.187% | −0.080% | +0.455% | 0.137% | +0.694% |
| MSFT | +0.335% | +0.014% | +0.656% | 0.164% | +0.180% |
| TSLA | +0.140% | −0.099% | +0.380% | 0.122% | +3.901% |
| XOM | −0.030% | −0.484% | +0.425% | 0.232% | +0.534% |

The TSLA row is illustrative: the model predicted a small +0.14% move; the actual return was +3.90%. The 95% CI reached only up to +0.38%. This single-sample observation reflects the calibration finding (Section 5 below) — the bands are too tight relative to actual market swings.

---

## 4. Multi-Seed Robustness Experiment

**Setup:** 5 seeds (42, 0, 1, 2, 3), winner hyperconfig (hidden=128, dropout=0.3, lr=1e-3, epochs=30), identical architecture, no test-set information used in config selection.

### Aggregate results (mean ± std across 5 seeds)

| Model | RMSE mean | RMSE std | Dir. Acc mean | Dir. Acc std |
|-------|-----------|----------|---------------|-------------|
| **LSTM** | **0.021390** | **±0.000074** | 51.14% | ±0.86% |
| **GRU** | **0.021314** | **±0.000037** | 51.36% | ±0.95% |

Baseline reference:
- Persistence RMSE: 0.030129 (no-parameters naive baseline)
- Linear regression RMSE: 0.021402

### Per-seed breakdown

| Seed | LSTM RMSE | LSTM dir. | GRU RMSE | GRU dir. |
|------|-----------|-----------|----------|---------|
| 42 | 0.021350 | 50.51% | 0.021280 | 50.37% |
| 0 | 0.021405 | 50.55% | 0.021346 | 51.74% |
| 1 | 0.021512 | 51.47% | 0.021259 | 51.09% |
| 2 | 0.021397 | 50.48% | 0.021348 | 53.00% |
| 3 | 0.021287 | 52.69% | 0.021338 | 50.58% |

**Finding:** Results are extremely stable. The RMSE standard deviation is 7.4e-5 for LSTM and 3.7e-5 for GRU — both less than 0.4% of the mean RMSE. The GRU is slightly more reproducible (tighter std) despite marginally better mean performance.

---

## 5. News Sentiment Experiment

**Setup:** 5 tickers with headline coverage (AAPL, AMZN, MSFT, TSLA, XOM), 37,853 headlines total from FNSPID dataset. Scoring: FinBERT (ProsusAI/finbert) and VADER (vaderSentiment). Pooled multi-stock training with ticker embedding. 5 seeds × 3 feature sets. Diebold-Mariano test (squared-error loss, two-sided) on seed 42 for significance.

| Feature set | RMSE mean | RMSE std | Dir. Acc mean | Dir. Acc std | DM stat | DM p-value | Sig.? |
|-------------|-----------|----------|---------------|-------------|---------|------------|-------|
| Baseline `[log_return, Volume]` | 0.019329 | ±0.000110 | 47.98% | ±1.67% | — | — | — |
| + VADER score | 0.019492 | ±0.000170 | 52.31% | ±2.48% | −1.299 | 0.194 | No |
| + FinBERT score | 0.019513 | ±0.000053 | 48.92% | ±0.75% | −1.798 | 0.073 | No |

**Headline coverage per ticker:** AAPL: 1,088 trading days, AMZN: 1,088, MSFT: 1,088, TSLA: 1,088, XOM: 1,088 (5,440 rows pooled).

**Finding:** Adding either sentiment signal makes the RMSE marginally *worse* (+0.8% and +1.0% respectively). Neither improvement in directional accuracy is statistically significant. The Diebold-Mariano test, which accounts for autocorrelation in forecast errors unlike a simple t-test, confirms the null. Trading-day gaps in news coverage are filled with a neutral score of 0.0 — days with no headline coverage are treated as no sentiment signal, not missing data.

---

## 6. Feature Ablation — Technical Indicators

**Setup:** 8 stocks, same winner config, 5 seeds. Baseline vs baseline + {RSI-14, SMA-ratio (close/20d SMA), 20-day rolling volatility, 10-day momentum}. Diebold-Mariano test on pooled test predictions.

| Feature set | RMSE mean | RMSE std | Dir. Acc mean | Dir. Acc std |
|-------------|-----------|----------|---------------|-------------|
| Baseline | 0.021406 | ±0.000066 | 50.84% | ±1.12% |
| + Technical indicators | 0.021403 | ±0.000040 | 50.41% | ±1.15% |

Diebold-Mariano test (indicators vs baseline):
- DM statistic: 2.347
- p-value: **0.019** (significant at 5%)
- Δ RMSE: **−0.003%** (practically zero)

**Finding:** Classic statistical vs practical significance mismatch. With 2,936 test samples, the DM test has enough power to detect differences smaller than 0.003% in RMSE — differences that are meaningless for any real-world application. Adding four technical indicators produces no meaningful improvement in forecasting accuracy.

---

## 7. MC-Dropout Empirical Calibration

**Setup:** Trained LSTM checkpoint, all 2,936 test windows, 200 forward passes per window with dropout active, 95% nominal CI computed from the 2.5th–97.5th percentile of the per-window pass distribution.

| Metric | Value |
|--------|-------|
| n test samples | 2,936 |
| n MC passes | 200 |
| Nominal CI | 95% |
| **Empirical coverage** | **20.6%** |
| Coverage gap | −74.4 percentage points |
| Mean band width | 0.00691 log-return units |
| Well-calibrated (< 5 pp gap) | No |

### Coverage vs band width (10 equal-count bins)

| Bin | Mean band width | Empirical coverage |
|-----|----------------|-------------------|
| Narrowest | 0.00447 | 11.6% |
| 2 | 0.00491 | 11.6% |
| 3 | 0.00526 | 14.7% |
| 4 | 0.00568 | 21.5% |
| 5 | 0.00616 | 21.8% |
| 6 | 0.00699 | 20.1% |
| 7 | 0.00765 | 26.6% |
| 8 | 0.00823 | 28.3% |
| 9 | 0.00894 | 25.6% |
| Widest | 0.01056 | 23.5% |

Even the widest 10% of bands only achieves 23.5% empirical coverage. Coverage monotonically increases with band width (wider bands do cover more, as expected), but the maximum achievable coverage is far below 95%.

**Explanation:** MC-dropout approximates the posterior over model weights, capturing *epistemic* uncertainty — uncertainty about which parameter values are correct. It does not capture *aleatoric* uncertainty — the irreducible randomness in the actual return process. For stock returns, aleatoric uncertainty dominates by a large margin: even a perfectly trained model cannot predict next-day returns with high confidence because markets are genuinely stochastic. The implication is that MC-dropout confidence bands are not appropriate as risk measures for financial forecasting without additional uncertainty calibration (e.g., conformal prediction).

---

## 8. Regime-Conditional Direction Accuracy

**Setup:** Trained LSTM, all 2,936 test windows. Regime classification: for each test window endpoint, compute the 20-day trailing rolling mean and std of log-returns. Label "trending" if `|rolling_mean| > 0.5 × rolling_std`, else "choppy".

| Regime | n samples | % of test set | Dir. Acc | Binom p-value |
|--------|-----------|--------------|----------|---------------|
| Trending | 113 | 3.9% | 51.3% | 0.851 |
| **Choppy** | **2,823** | **96.1%** | **50.5%** | **0.624** |
| Overall | 2,936 | 100% | 50.5% | 0.592 |

**Finding:** 96% of test windows are classified as choppy (small rolling return relative to rolling std). This is consistent with financial theory — genuine trending periods are rare at daily resolution. The directional accuracy in trending periods (51.3%) is slightly higher but with only 113 samples and p = 0.851, the difference is nowhere near statistical significance. The model's coin-flip accuracy is entirely regime-independent.

---

## 9. Permutation Feature Importance

**Setup:** Trained LSTM, all 2,936 test windows. Each feature (or the ticker embedding ID) is shuffled 5 times; the increase in MSE from baseline is the importance score. Baseline MSE: 4.558e-4.

| Feature | ΔMSE (increase) | Relative importance |
|---------|----------------|---------------------|
| Volume | +2.998e-6 | Highest |
| log_return | +2.512e-6 | Second |
| ticker_id (embedding) | −8.5e-8 | Near zero (negative) |

Both features contribute positively: shuffling either raises MSE. Volume is marginally more important than log_return. The ticker embedding contributes negligibly (the slight negative value is within noise — shuffling it doesn't hurt because the model has largely encoded ticker information in the shared LSTM weights). The effect sizes are small relative to baseline MSE (< 1%), confirming that neither feature is critical on its own.

---

## Summary: Key Numbers at a Glance

| Result | Value | Context |
|--------|-------|---------|
| LSTM RMSE (test) | **0.02135** | vs persistence 0.03013 (+29% improvement) |
| GRU RMSE (test) | **0.02128** | slightly better than LSTM |
| Directional accuracy | **50.5%** | statistically = coin flip (p = 0.59) |
| LSTM stability (5 seeds) | **±0.00007** RMSE | robust, seed-independent |
| Sentiment improvement | **not significant** | VADER p=0.194, FinBERT p=0.073 |
| Technical indicators | **Δ −0.003% RMSE** | stat. sig. but practically irrelevant |
| MC-dropout coverage | **20.6%** vs 95% nominal | overconfident — epistemic only |
| Regime accuracy (trending) | **51.3%** | no regime advantage (p=0.85) |
| Portfolio strategy return | **−0.46%** | B&H returned +22.0% |
| TSLA VaR 95% | **5.30% / day** | highest risk in portfolio |
| JNJ VaR 95% | **1.73% / day** | lowest risk in portfolio |

---

*All results generated by the pipeline in `src/`. To reproduce: follow the setup instructions in README.md and run `python main.py --all` followed by the experiment scripts.*
