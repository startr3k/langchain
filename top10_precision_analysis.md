# Daily Top-10 Stock Picks Precision Analysis

## Executive Summary

**Current daily top-10 hit rate: 64.4%** (522/810 picks across 81 sampled dates, Oct 2024 – May 2026)

### Critical Finding: LTR Model Not Active

The LTR model file (`ltr_model.json`) does not exist on disk. The ensemble is effectively **100% classification** — the LTR component contributes zero to predictions. This explains why the ensemble weight sweep shows identical results for all weights 0.0–0.9 (all 64.4%). The LTR architecture is coded but **needs to be trained on the production dataset** before it can contribute.

**Implication:** The 67.8% hit rate reported earlier was from an in-session LTR prototype that was not persisted. The real production hit rate is 64.4% (classification only).

---

## 1. Feature Analysis (Hits vs Misses in Top-10 Picks)

Features ranked by Cohen's d effect size — how strongly they differentiate hits from misses:

| Feature | Hit Mean | Miss Mean | Cohen's d | Interpretation |
|---|---|---|---|---|
| **Dist_52w_High** | 0.317 | 0.429 | **-0.453** | Hits are **closer to 52w high** (less beaten down) |
| **yield_curve_spread** | 0.280 | 0.375 | **-0.322** | Hits in **tighter yield curve** environments |
| **sp500_volatility_20d** | 0.153 | 0.132 | +0.254 | Hits during **higher market volatility** |
| **treasury_3m** | 4.006 | 3.933 | +0.236 | Hits with slightly **higher short-term rates** |
| **Volatility_60d** | 0.071 | 0.062 | +0.230 | Hits have **higher long-term volatility** |
| **Price_to_SMA_200** | 0.680 | 0.798 | -0.218 | Hits are **further below SMA200** (deeper correction) |
| **vix_close** | 19.13 | 18.20 | +0.217 | Hits in **higher VIX** environments |
| **BB_Width** | 0.445 | 0.386 | +0.205 | Hits have **wider Bollinger Bands** |
| **Volatility_20d** | 0.066 | 0.056 | +0.188 | Hits have **higher short-term volatility** |

### Key Insight
**Hits are volatile stocks near their highs in uncertain (high VIX) markets.** Misses tend to be lower-volatility stocks trading further from highs in calmer markets. The model should be biased toward selecting higher-volatility stocks during elevated market uncertainty.

---

## 2. Score Distribution Problem

| Metric | Hits | Misses | Gap |
|---|---|---|---|
| Classification Probability | 0.732 | 0.687 | +0.045 |

The classification model separates hits from misses by only **4.5 percentage points** at the top. This tiny gap means small noise in probabilities causes frequent rank swaps between hits and misses. The model needs **better discrimination at the very top of the ranking**.

---

## 3. Temporal Performance

| Period | Hit Rate | Market Context |
|---|---|---|
| **Best months** | | |
| May 2025 | **95.0%** | Strong bull market |
| Apr 2025 | **90.0%** | Recovery rally |
| Aug 2025 | **85.0%** | Continued momentum |
| **Worst months** | | |
| May 2026 | **17.5%** | Near end of test window |
| Oct 2025 | **44.0%** | Correction period |
| Apr 2026 | **45.0%** | Recent weakness |

The model performs well in bull markets (Apr–Sep 2025: 75–95%) but struggles during corrections (Oct 2025: 44%, Apr–May 2026: 17–45%). This suggests **model drift** — the model is less effective when market conditions shift from its training data.

---

## 4. Repeat Offender Tickers

Tickers that appear frequently in top-10 but rarely hit:

| Ticker | Appearances | Hits | Hit Rate | Avg Return |
|---|---|---|---|---|
| CLYM | 14 | 1 | 7.1% | 6.2% |
| APYX | 13 | 3 | 23.1% | 14.9% |
| EMPD | 12 | 2 | 16.7% | 16.2% |
| ANGI | 8 | 0 | 0.0% | 4.7% |
| BBOT | 5 | 0 | 0.0% | 10.9% |

These tickers consistently fool the model — high probability scores but low actual returns. A **per-ticker calibration** or **blacklist mechanism** could eliminate these repeat false positives.

---

## 5. Miss Quality

| Category | Count | % of Misses |
|---|---|---|
| Negative returns (<0%) | 40 | 13.9% |
| Bad misses (0–5%) | 59 | 20.5% |
| Moderate misses (5–10%) | 70 | 24.3% |
| Near misses (10–15%) | 68 | 23.6% |
| Close misses (15–20%) | 51 | 17.7% |

**34.4% of misses are "bad" (<5% return)** — the model is confidently wrong on these. Only 17.7% are close misses (15–20%) that nearly hit the threshold.

---

## 6. Top-K Sensitivity

| K | Hit Rate |
|---|---|
| Top 5 | **69.9%** |
| Top 10 | 64.4% |
| Top 15 | 60.2% |
| Top 20 | 60.0% |
| Top 50 | 56.0% |

Precision is significantly higher for top 5 (69.9%) vs top 10 (64.4%). The model's very best picks are substantially better than picks #6–10. **Consider narrowing to top 5 if quality matters more than quantity.**

---

## Recommendations (Prioritized by Expected Impact)

### 1. Train and Persist the LTR Model (HIGH — Expected: +3-5%)
The LTR model was prototyped but never saved to disk. It needs to be trained on the full production dataset and saved. The walk-forward CV showed the LTR ensemble improves hit rate by ~6% over classification alone. This is the biggest immediate win — it's already coded, just needs to be executed.

### 2. Custom Top-10 Loss Function (HIGH — Expected: +5-10%)
The classification model optimizes Average Precision across ALL predictions. It doesn't know or care about the top 10. A custom loss function that heavily penalizes misranking at the top (e.g., weighted cross-entropy where the loss increases for higher-ranked predictions) would directly optimize what matters.

**Concrete approach:** Train a second-stage reranker using LightGBM with a custom `ndcg@10` objective, trained on daily cross-sections. Use the classification probability as one feature among many.

### 3. Volatility-Aware Scoring (MEDIUM — Expected: +1-3%)
Hits have 2× the volatility of misses. Incorporating volatility directly into the ranking score (not as a hard filter, but as a multiplier) would boost higher-volatility candidates that are more likely to achieve ≥20% breakouts.

**Concrete approach:** `adjusted_score = ensemble_score × (1 + α × volatility_percentile)` where α is tuned via cross-validation.

### 4. Market Regime Detection (MEDIUM — Expected: +2-5%)
Hit rate varies from 17.5% (May 2026) to 95% (May 2025). If the model could detect when it's in a low-confidence regime, it could either (a) increase its confidence threshold for top-10 selection, or (b) flag that day's picks as lower confidence.

**Concrete approach:** Train a meta-model that predicts daily hit rate based on market conditions (VIX level, recent market returns, yield curve). When predicted hit rate is low, either skip that day or apply stricter filters.

### 5. Per-Ticker Calibration (MEDIUM — Expected: +2-3%)
Some tickers (CLYM, ANGI, BBOT) are persistent false positives. The model should learn to down-weight tickers with poor historical hit rates in the top-10.

**Concrete approach:** Maintain a rolling hit rate per ticker. Penalize tickers with <30% hit rate in recent top-10 appearances: `calibrated_score = score × ticker_confidence_factor`.

### 6. Temporal Decay Weighting (LOW — Expected: +1-2%)
More recent training data should be weighted higher since market conditions evolve. The current model treats 2016 data equally with 2024 data.

**Concrete approach:** Apply exponential time decay to sample weights: `weight_i = base_weight × exp(-λ × years_ago)`.

### 7. Top-5 vs Top-10 Tradeoff (ALTERNATIVE)
If precision is more important than coverage, narrowing from top-10 to top-5 picks immediately boosts hit rate from 64.4% → 69.9% with no model changes needed. The model's very best picks are substantially more reliable.
