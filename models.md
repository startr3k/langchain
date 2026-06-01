# 4-Stage Walk-Forward ML Pipeline

## Pipeline Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CLASSIFICATION_THRESHOLD` | 0.20 | Target: stock achieves ≥20% peak return within 3 months |
| `CLS_PROB_THRESHOLD` | 0.50 | Stage 1 gate: classifier probability must be ≥50% |
| `MFD_PRED_THRESHOLD` | 0.25 | Stage 2 gate: predicted return magnitude must be ≥25% |
| `MIN_ELITE_POOL` | 75 | Minimum stocks in elite pool to generate daily picks |
| `FOLD_WEIGHTS` | [1, 2, 4, 8, 16] | Exponential fold weights (normalized to [0.032, 0.065, 0.129, 0.258, 0.516]) |

## Walk-Forward Cross-Validation

| Parameter | Value |
|-----------|-------|
| Number of folds | 5 |
| Minimum training window | 3 years |
| Train/test gap | 63 days (3 months, prevents forward-return leakage) |
| Test span | ~7 years (total data span minus 3-year min training) |
| Fold size | `int((test_years × 365.25) / n_folds)` ≈ 510 days |
| Data | 616 NASDAQ tickers, ~10 years (2016–2026), 49 features |

Each fold trains on all data before `test_start - 63 days` (expanding window) and evaluates on the next ~510-day period. Folds are non-overlapping in test periods.

---

## Stage 1: FLAML Binary Classifier

**Purpose:** Predict P(stock achieves ≥20% peak return in 3 months).

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Framework | FLAML AutoML |
| Task | Binary classification |
| Metric | Average Precision (AP) |
| Estimators | `["xgboost", "lgbm"]` |
| Eval method | 5-fold internal CV (`eval_method="cv"`, `n_splits=5`) |
| Early stopping | `True` |
| Time budget | 120 seconds per fold (walk-forward default: 60s) |
| Sample weights | Inverse class frequency: `1/pos_rate` for positives, `1/(1-pos_rate)` for negatives |
| Seed | 42 |

**FLAML auto-selected hyperparameters per fold** (from last training run):

| Parameter | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-----------|--------|--------|--------|--------|--------|
| Best estimator | lgbm | lgbm | lgbm | lgbm | lgbm |
| max_leaves | 825 | 29 | 198 | 103 | 28 |
| learning_rate | 0.383 | 0.420 | 1.000 | 0.257 | 0.372 |
| n_estimators | 72 | 167 | 14 | 8 | 16 |
| min_child_weight | 0.16 | 0.05 | 0.48 | 1.00 | 0.57 |
| subsample | 0.995 | 0.757 | 0.846 | 0.946 | 0.784 |
| reg_lambda | 2.30 | 4.32 | 12.05 | 22.45 | 2.85 |

> Note: FLAML searches its default hyperparameter ranges per fold. The `custom_hp` regularization constraints in the `train()` method are **not** passed to `train_walk_forward()`. These values change on each retrain.

**Gate:** Stocks with `P(≥20%) >= 0.50` pass to Stage 2.

---

## Stage 2: XGBoost Huber Regressor

**Purpose:** Predict the magnitude of forward return (continuous). Filters out stocks where the classifier is confident but the expected return magnitude is low.

**Fixed hyperparameters** (deterministic, same for all folds):

| Parameter | Value |
|-----------|-------|
| Objective | `reg:pseudohubererror` |
| Huber slope | 0.5 |
| Max depth | 6 |
| Learning rate | 0.05 |
| Min child weight | 50 |
| Subsample | 0.7 |
| Colsample bytree | 0.5 |
| Reg alpha (L1) | 5.0 |
| Reg lambda (L2) | 10.0 |
| Num boost rounds | 500 |
| Early stopping rounds | 50 |
| Seed | 42 |

**Gate:** Stocks with `predicted MFD >= 0.25` (≥25% predicted return) pass to Stage 3.

**Elite pool:** Stocks passing both Stage 1 AND Stage 2 gates form the elite pool for that day.

---

## Stage 3: Cross-Sectional Quantile Transform

**Purpose:** Convert raw features to per-date percentile ranks, removing cross-sectional scale effects before ranking.

**No learned parameters.** This is a purely mechanical rank transformation applied per trading date across the elite pool.

---

## Stage 4: LambdaMART Ranking (XGBoost)

**Purpose:** Rank elite-pool stocks to select the top-10 daily picks. Optimizes for NDCG@10 (Normalized Discounted Cumulative Gain).

**Fixed hyperparameters** (deterministic, same for all folds):

| Parameter | Value |
|-----------|-------|
| Objective | `rank:ndcg` |
| Eval metric | `ndcg@10` |
| Max depth | 6 |
| Learning rate | 0.05 |
| Min child weight | 50 |
| Subsample | 0.7 |
| Colsample bytree | 0.5 |
| Reg alpha (L1) | 5.0 |
| Reg lambda (L2) | 10.0 |
| Num boost rounds | 500 |
| Early stopping rounds | 50 |
| NDCG exp gain | False |
| Seed | 42 |

> Note: Stages 2 and 4 share identical regularization parameters by design.

---

## Ensemble Inference

### Exponential Fold Weighting

All inference paths (`predict_batch`, `predict`, `predict_ticker`) use exponential fold weighting:

```
Raw weights:       [1, 2, 4, 8, 16]  (5 folds)
Normalized:        [0.0323, 0.0645, 0.1290, 0.2581, 0.5161]
```

Fold 5 (most recent, most training data) gets ~52% weight. Earlier folds provide regime-change insurance.

If fewer folds are available, the last N weights are used and renormalized.

### Scoring Formula

For batch inference (daily picks):

```
Z_cls = (classifier_prob - mean(elite_proba)) / std(elite_proba)
Z_ltr = (ltr_score - mean(elite_ltr)) / std(elite_ltr)
pool_weight = min(elite_pool_size / 75, 2.0)

score = max(Z_cls, 0) × max(Z_ltr, 0) × pool_weight
```

- Z-scores are computed within each day's elite pool (cross-sectional)
- `max(Z, 0)` zeroes out below-average stocks
- `pool_weight` boosts scores on high-conviction days (large elite pools)
- Top 10 stocks by score are selected as daily picks

For single-ticker inference (Stock Analysis):
- Returns the exponential-weighted average of classifier probabilities across folds
- Compared against `optimal_threshold` for BUY/HOLD signal

---

## Preprocessing (Per-Fold)

| Step | Method |
|------|--------|
| NaN fill | Per-fold training set median values |
| Feature clipping | Per-fold 1st–99th percentile Winsorization |
| Sample weights | Inverse class frequency balancing |

---

## Performance (Last Training Run)

**Hit Rate at Pool ≥ 150:** 80.8% (751/930 picks, 93 qualifying days)

| Fold | Test Period | Days (pool≥150) | Hit Rate |
|------|-------------|-----------------|----------|
| 1 | ~2019–2020 | 47 | 79.6% |
| 2 | ~2020–2022 | 0 | N/A |
| 3 | ~2022–2023 | 17 | 68.8% |
| 4 | ~2023–2024 | 0 | N/A |
| 5 | ~2024–2026 | 29 | 89.7% |

**Exponential weighting on Fold 5 test period** (pool ≥ 150, 29 days):
- Hit Rate: 89.3% (259/290)
- Rank-1 Accuracy: 100% (29/29)

**Feature importance (Stage 1 classifier, top 10):**

| Feature | Importance |
|---------|-----------|
| Volatility_60d | 18.9% |
| treasury_3m | 9.9% |
| sp500_return_60d | 8.1% |
| yield_curve_spread | 7.9% |
| Dist_52w_High | 5.3% |
| hist_total_assets | 4.6% |
| vix_close | 3.9% |
| Volatility_20d | 3.5% |
| Return_60d | 3.2% |
| BB_Width | 2.8% |

39 out of 49 features contribute (>0% importance) in Stage 1.
45 out of 49 features contribute in Stage 4 (LTR).
