# Model & Feature Improvement Analysis

## Current Performance Summary

| Metric | Value |
|---|---|
| Test AUC | 0.6779 |
| Walk-Forward CV Hit Rate | 80% (40/50 across 5 folds) |
| Walk-Forward Avg Peak Return | 89.6% |
| Best Model | XGBoost (884 trees, depth 7) |
| Features | 49 used (67 available minus 5 dropped) |
| Train/Test AUC Gap | **0.116** (0.794 train → 0.678 test) |

---

## Issue 1: Significant Overfitting (HIGH IMPACT)

**Finding:** Train AUC = 0.794 vs Test AUC = 0.678 — a **0.116 gap**. This is the single biggest issue.

**Evidence:**
- XGBoost selected 884 trees at depth 7 — complex enough to memorize training patterns
- Calibration is poor at high probabilities: model predicts 93.8% but actual positive rate is only 74.8% (18.9% gap)
- Model predicts 84.7% but actual is 58.0% (26.7% gap)
- The model is overconfident on its top predictions

**Recommendations:**
1. **Stronger regularization** — Reduce `max_depth` from 7 → 4-5, increase `min_child_weight` from 37 → 100+, cap `n_estimators` at 300-500 with proper early stopping
2. **Probability calibration** — Apply Platt scaling (isotonic regression or sigmoid calibration) as a post-processing step. The model's raw probabilities are systematically overconfident at the top end
3. **Add early stopping on the held-out temporal test set** — Currently FLAML uses 5-fold CV on the training set for model selection, but the 5-fold CV is random (not temporal), which leaks temporal information. Use `eval_method="holdout"` with a temporal split instead, or implement manual early stopping on the chronological test set

---

## Issue 2: Poor Calibration (HIGH IMPACT)

**Finding:** The model's predicted probabilities don't match reality, especially at high confidence:

| Predicted | Actual | Gap |
|---|---|---|
| 6.8% | 12.9% | 6.2% (under) |
| 54.8% | 46.4% | 8.4% (over) |
| 74.8% | 53.6% | **21.2% (over)** |
| 84.7% | 58.0% | **26.7% (over)** |
| 93.8% | 74.8% | **19.0% (over)** |

**Why this matters:** When the model says a stock has 90%+ probability of 20% returns, the actual rate is only ~75%. The top-10 picks are selected based on these overconfident probabilities.

**Recommendations:**
1. **Isotonic regression calibration** on a held-out calibration set (e.g., split test set into calibration + evaluation)
2. **Use calibrated probabilities for stock ranking** — the relative ordering matters more than absolute values for top-N selection
3. **Consider using `predict_proba` with `CalibratedClassifierCV`** from sklearn as a wrapper

---

## Issue 3: Extreme Outlier Features (MEDIUM IMPACT)

**Finding:** Several features have extreme outlier values that could distort tree splits:

| Feature | Range | Issue |
|---|---|---|
| `Return_60d` | -0.88 to **71.38** | 71x return in 60d — likely data error or penny stock |
| `Return_20d` | -0.81 to **16.97** | 1697% in 20 days |
| `Dist_52w_Low` | 1.0 to **1475.79** | Price 1475x above 52-week low |
| `hist_profit_margin` | -23,151 to 49 | Negative margins of -23,000%+ |
| `hist_roe` | -3,700 to 361 | Extreme ROE ratios |
| `hist_debt_to_equity` | -386 to 634 | Extreme leverage ratios |

**Recommendations:**
1. **Winsorize extreme values** — Clip features at 1st and 99th percentiles. Tree models are not scale-invariant to split location — extreme outliers can create uninformative splits
2. **Remove or flag penny stock data** — Returns of 7000%+ in 60 days are almost certainly penny stocks or data errors that teach the model noise
3. **Add minimum price filter** — Exclude stocks trading below $1-2 from training (these are subject to manipulation and have unreliable data)

---

## Issue 4: Features with Near-Zero Signal (MEDIUM IMPACT)

**Finding:** 10+ features have Spearman correlation < 0.01 with the target:

| Feature | Correlation | Currently Used? |
|---|---|---|
| `MACD_Hist` | -0.0003 | Yes (importance: 0) |
| `Volume_Ratio` | -0.0004 | Yes (importance: 0) |
| `Volume_Spike_Magnitude` | +0.0018 | Yes (importance: 0) |
| `Volume_Surge_3d` | -0.0022 | Yes (importance: 0) |
| `BB_Squeeze_Duration` | -0.0025 | Yes (importance: 0) |
| `insider_buy_ratio_90d` | +0.0082 | Yes (importance: 0) |
| `Volatility_Contraction` | +0.0101 | Yes (importance: 0) |
| `treasury_3m` | +0.0009 | Yes |

These features have **zero split importance AND near-zero correlation** with the target. They add noise and search space complexity without contributing signal.

**Recommendations:**
1. **Remove zero-importance features** — Features with both zero split importance and zero permutation importance are provably useless for this model. Removing them reduces noise and speeds up training
2. **Validate via walk-forward CV** — Remove the bottom ~10 features and verify hit rate doesn't decrease
3. **Keep Volume_Surge_3d** only as the Stage 2 rule filter — it has no predictive signal for the model but is still used as a confirmation rule

---

## Issue 5: Temporal Feature Drift (MEDIUM IMPACT)

**Finding:** Key features have significant distribution shifts across years:

- `treasury_3m`: Ranges from 0.03 (2021) to 5.05 (2023) — massive regime change
- `Volatility_20d`: 0.024 (2017) to 0.056 (COVID 2020) — 2.3x swing
- `sp500_return_60d`: -0.040 (2022) to +0.070 (2024) — sign flip

**Why this matters:** The model trains on all historical data equally. Patterns from 2016 (zero-rate environment) may not apply in 2026 (high-rate environment). This likely explains why `treasury_10y` had negative permutation importance — it memorized the rate regime.

**Recommendations:**
1. **Time-based sample weighting** — Weight recent data more heavily (exponential decay). Stocks from 2024-2026 are more relevant than 2016-2018 patterns
2. **Relative/Z-score features instead of absolute** — Replace `treasury_3m` with `treasury_3m_percentile_2y` (where does the current rate sit relative to the past 2 years). This makes the feature stationary
3. **Rolling normalization** — For macro features, subtract the trailing 1-year mean and divide by trailing 1-year std
4. **Regime-aware features** — Add binary regime indicators (e.g., `high_rate_environment = treasury_3m > 2%`) that are more stable than raw levels

---

## Issue 6: NaN Handling Improvements (LOW-MEDIUM IMPACT)

**Finding:** `hist_debt_to_equity` has 58.8% NaN, and 10 features exceed 10% NaN.

**Current handling:** Semantic fill for profit-related features (good), median fill for everything else.

**Recommendations:**
1. **Add NaN indicator columns** for high-NaN features — Create binary `hist_debt_to_equity_missing` features. The missingness pattern itself may be informative (e.g., companies that don't report debt-to-equity may be systematically different)
2. **Use tree-native NaN handling** — XGBoost/LightGBM can natively route NaN values to the optimal child node during tree building. Instead of filling NaN with median, pass NaN through directly. This is often better than any imputation strategy
3. **Consider removing `hist_debt_to_equity`** — At 58.8% NaN, this feature is more missing than present. The model may be mostly learning from the NaN pattern rather than the actual values

---

## Issue 7: Target Definition and Labeling (LOW IMPACT but worth exploring)

**Current:** Binary classification — `Forward_Max_Return_3M >= 20%` (34.2% positive rate).

**Observations:**
- 34.2% positive rate is relatively balanced, which is good
- But the threshold is arbitrary — 19.9% return = negative class, 20.1% = positive class

**Recommendations:**
1. **Try regression instead of classification** — Predict the actual `Forward_Max_Return_3M` value, then rank stocks by predicted return. This preserves more information than a binary cutoff
2. **Multi-class buckets** — Low (<10%), Medium (10-30%), High (30%+) instead of binary. The model could learn different patterns for "moderate gainer" vs "breakout stock"
3. **Weighted target** — Use the continuous return value as sample weight (higher returns = more important to get right)

---

## Issue 8: Feature Engineering Opportunities (MEDIUM IMPACT)

**New features that could add signal based on the analysis:**

1. **Relative strength vs market** — `stock_return_20d - sp500_return_20d` (outperformance). Currently both exist separately but the difference isn't explicitly modeled. Cohen's d for `Dist_52w_High` = -0.488, suggesting relative price position matters

2. **Volatility regime interaction** — `Volatility_20d × vix_close` interaction. Volatility is the #1 predictor (r=0.315) and VIX adds context about whether high volatility is stock-specific or market-wide

3. **Earnings + momentum interaction** — `earnings_surprise_pct × Return_20d` (stocks beating earnings AND showing momentum). Currently `Fundamental_Surprise` uses `revenue_growth × surprise`, but momentum is a stronger signal than revenue growth

4. **Price position features** — `Dist_52w_High × Volatility_Contraction` (near highs + tightening volatility = breakout setup). This is the classic William O'Neil CAN SLIM pattern

5. **Rolling Z-scores** — For features like `Volume_Ratio`, compute the Z-score relative to the stock's own 60-day history. A Volume_Ratio of 3.0 means different things for a volatile biotech vs a stable utility

6. **Cross-sectional rank features** — Rank each stock's momentum/volatility against the universe on each date. `momentum_rank_pct` (percentile within the universe on that date) is stationary by construction

---

## Prioritized Improvement Roadmap

| Priority | Improvement | Expected Impact | Effort |
|---|---|---|---|
| **1** | Fix overfitting (regularization + early stopping) | +2-5% AUC | Low |
| **2** | Probability calibration (isotonic regression) | Better ranking, fewer false positives | Low |
| **3** | Winsorize extreme values + minimum price filter | Cleaner training signal | Low |
| **4** | Remove zero-importance features (~10 features) | Faster training, less noise | Low |
| **5** | Add relative/Z-score features (market-relative, rolling) | +1-3% AUC | Medium |
| **6** | Time-weighted training (recent data weighted more) | Better regime adaptation | Medium |
| **7** | Interaction features (volatility × momentum, earnings × price) | +1-2% AUC | Medium |
| **8** | Native NaN handling (pass through, not median fill) | +0.5-1% AUC | Low |
| **9** | Cross-sectional rank features | Stationarity | Medium |
| **10** | Regression target or multi-class | Better signal utilization | High |
