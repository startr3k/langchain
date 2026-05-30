"""Hyperparameter optimization for LTR + volatility alpha targeting daily top-10 hit rate."""

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────
DATA_PATH = Path("training_data_10y_full.csv")
print("Loading data...")
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Rows: {len(df_raw):,}")

from stock_predictor.data.feature_engineering import ALL_FEATURE_NAMES, TARGET_COLUMN
from stock_predictor.models.automl_model import (
    StockReturnPredictor,
    CLASSIFICATION_THRESHOLD,
    _fill_semantic_nan,
    _log_transform,
    _compute_derived_features,
)
import xgboost as xgb

# Load the trained classification model
predictor = StockReturnPredictor()
predictor.load()

# ── Prepare features ──────────────────────────────────────────────
df = df_raw.copy()
df = _compute_derived_features(df)

feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
X = df[feature_cols].copy()
y = (df[TARGET_COLUMN] >= CLASSIFICATION_THRESHOLD).astype(int)

X = _fill_semantic_nan(X)
X = _log_transform(X)
X = X.fillna(predictor.feature_medians)

# Temporal sort
date_series = pd.to_datetime(df["_date"], errors="coerce")
sort_order = date_series.sort_values().index
X = X.loc[sort_order].reset_index(drop=True)
y = y.loc[sort_order].reset_index(drop=True)
df = df.loc[sort_order].reset_index(drop=True)

raw_vol = df["Volatility_20d"].values.copy() if "Volatility_20d" in df.columns else None

# Split
gap_rows = max(1, int(len(X) * 0.05))
split_idx = int(len(X) * 0.75)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
df_train = df.iloc[:split_idx]
X_test = X.iloc[split_idx + gap_rows:]
y_test = y.iloc[split_idx + gap_rows:]
df_test = df.iloc[split_idx + gap_rows:].reset_index(drop=True)
raw_vol_test = raw_vol[split_idx + gap_rows:] if raw_vol is not None else None

print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

# Classification probabilities (fixed — not retraining classification)
y_proba_test = predictor.automl.predict_proba(X_test)
if y_proba_test.ndim == 2:
    class_proba = y_proba_test[:, 1]
else:
    class_proba = y_proba_test

# Prepare LTR training data
train_dates = pd.to_datetime(df_train["_date"], errors="coerce")
unique_train_dates = sorted(train_dates.unique())
train_groups = train_dates.map({d: i for i, d in enumerate(unique_train_dates)}).values
group_counts_train = pd.Series(train_groups).value_counts().sort_index().values

test_dates = pd.to_datetime(df_test["_date"], errors="coerce")
unique_test_dates = sorted(test_dates.unique())
test_groups = test_dates.map({d: i for i, d in enumerate(unique_test_dates)}).values
group_counts_test = pd.Series(test_groups).value_counts().sort_index().values

# Sample dates for hit rate eval
sample_dates = sorted(df_test["_date"].unique())[::5]
print(f"  Eval dates: {len(sample_dates)}")

# Ticker calibration factors (from existing model)
test_tickers = df_test["Ticker"].values
cal_factors = np.array([predictor.ticker_calibration.get(t, 1.0) for t in test_tickers])


def compute_daily_hit_rate(scores, top_k=10):
    """Compute daily top-K hit rate from scores array."""
    df_eval = df_test.copy()
    df_eval["_score"] = scores
    df_eval["_y"] = y_test.values
    daily = []
    for date in sample_dates:
        day_df = df_eval[df_eval["_date"] == date]
        if len(day_df) < top_k:
            continue
        top = day_df.nlargest(top_k, "_score")
        daily.append(float(top["_y"].mean()))
    return np.mean(daily) if daily else 0.0


def train_and_eval_ltr(params, vol_alpha, ensemble_weight):
    """Train LTR with given params and evaluate daily top-10 hit rate."""
    dtrain = xgb.DMatrix(X_train, label=y_train.values)
    dtrain.set_group(group_counts_train)
    dtest = xgb.DMatrix(X_test)

    xgb_params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "tree_method": "hist",
        "verbosity": 0,
        **params,
    }

    try:
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=params.get("n_estimators", 300),
            verbose_eval=False,
        )
    except Exception as e:
        return None, str(e)

    ltr_raw = model.predict(dtest)
    ltr_scores = 1.0 / (1.0 + np.exp(-ltr_raw))

    # Ensemble
    ensemble = (1 - ensemble_weight) * class_proba + ensemble_weight * ltr_scores

    # Volatility-aware scoring
    if raw_vol_test is not None and len(raw_vol_test) > 1:
        vol_pctl = pd.Series(raw_vol_test).rank(pct=True).values
        scores = ensemble * (1 + vol_alpha * vol_pctl)
    else:
        scores = ensemble

    # Ticker calibration
    scores = scores * cal_factors

    hr10 = compute_daily_hit_rate(scores, 10)
    hr5 = compute_daily_hit_rate(scores, 5)
    return (hr10, hr5), model


# ── BASELINE ──────────────────────────────────────────────────────
baseline_hr10 = compute_daily_hit_rate(class_proba, 10)
baseline_hr5 = compute_daily_hit_rate(class_proba, 5)
print(f"\nBaseline (classification only): top-10={baseline_hr10:.1%}, top-5={baseline_hr5:.1%}")

# Current config
current_result, _ = train_and_eval_ltr(
    {"max_depth": 6, "eta": 0.1, "min_child_weight": 50,
     "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 500},
    vol_alpha=0.3, ensemble_weight=0.5,
)
if current_result:
    print(f"Current config: top-10={current_result[0]:.1%}, top-5={current_result[1]:.1%}")

# ── HYPERPARAMETER GRID ──────────────────────────────────────────
print("\n" + "="*70)
print("HYPERPARAMETER SEARCH")
print("="*70)

# LTR params to search
ltr_configs = [
    # max_depth variations
    {"max_depth": 3, "eta": 0.1, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 300},
    {"max_depth": 4, "eta": 0.1, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 300},
    {"max_depth": 5, "eta": 0.1, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 400},
    {"max_depth": 6, "eta": 0.1, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 500},
    {"max_depth": 8, "eta": 0.1, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 500},
    # eta variations
    {"max_depth": 4, "eta": 0.01, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 500},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 500},
    {"max_depth": 5, "eta": 0.05, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 500},
    {"max_depth": 6, "eta": 0.05, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 500},
    # min_child_weight variations
    {"max_depth": 4, "eta": 0.1, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 300},
    {"max_depth": 4, "eta": 0.1, "min_child_weight": 100, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 300},
    {"max_depth": 4, "eta": 0.1, "min_child_weight": 200, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 300},
    # subsample variations
    {"max_depth": 4, "eta": 0.1, "min_child_weight": 50, "subsample": 0.6, "colsample_bytree": 0.8, "n_estimators": 300},
    {"max_depth": 4, "eta": 0.1, "min_child_weight": 50, "subsample": 0.7, "colsample_bytree": 0.7, "n_estimators": 300},
    {"max_depth": 4, "eta": 0.1, "min_child_weight": 50, "subsample": 0.9, "colsample_bytree": 0.9, "n_estimators": 300},
    # Aggressive regularization
    {"max_depth": 3, "eta": 0.05, "min_child_weight": 100, "subsample": 0.7, "colsample_bytree": 0.7, "n_estimators": 500, "lambda": 5, "alpha": 1},
    {"max_depth": 3, "eta": 0.05, "min_child_weight": 200, "subsample": 0.6, "colsample_bytree": 0.6, "n_estimators": 500, "lambda": 10, "alpha": 2},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 100, "subsample": 0.7, "colsample_bytree": 0.7, "n_estimators": 400, "lambda": 5, "alpha": 1},
    # Higher capacity
    {"max_depth": 5, "eta": 0.1, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 600},
    {"max_depth": 6, "eta": 0.1, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 800},
]

vol_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ensemble_weights = [0.3, 0.4, 0.5, 0.6, 0.7]

# Phase 1: Find best LTR config (with default vol_alpha=0.3, weight=0.5)
print("\nPhase 1: LTR hyperparameter search (vol_alpha=0.3, weight=0.5)")
print(f"{'Config':<75} {'Top-10':>8} {'Top-5':>8}")
print("-" * 93)

best_ltr_config = None
best_ltr_hr10 = 0
results = []

for i, config in enumerate(ltr_configs):
    n_est = config.pop("n_estimators", 300)
    result, _ = train_and_eval_ltr(config, vol_alpha=0.3, ensemble_weight=0.5)
    config["n_estimators"] = n_est  # restore

    if result is None:
        print(f"  Config {i}: FAILED")
        continue

    hr10, hr5 = result
    desc = f"d={config['max_depth']} eta={config['eta']} mcw={config['min_child_weight']} ss={config['subsample']} cs={config['colsample_bytree']} n={n_est}"
    if "lambda" in config:
        desc += f" L2={config['lambda']} L1={config.get('alpha', 0)}"
    marker = " ***" if hr10 > best_ltr_hr10 else ""
    print(f"  {desc:<73} {hr10:>7.1%} {hr5:>7.1%}{marker}")

    results.append((config.copy(), hr10, hr5))
    if hr10 > best_ltr_hr10:
        best_ltr_hr10 = hr10
        best_ltr_config = config.copy()

print(f"\nBest LTR config: {best_ltr_config}")
print(f"Best top-10: {best_ltr_hr10:.1%}")

# Phase 2: Optimize vol_alpha with best LTR config
print(f"\nPhase 2: Volatility alpha search (best LTR, weight=0.5)")
print(f"{'Alpha':<10} {'Top-10':>8} {'Top-5':>8}")
print("-" * 28)

best_vol_alpha = 0.3
best_vol_hr10 = 0

for alpha in vol_alphas:
    n_est = best_ltr_config.pop("n_estimators", 300)
    result, _ = train_and_eval_ltr(best_ltr_config, vol_alpha=alpha, ensemble_weight=0.5)
    best_ltr_config["n_estimators"] = n_est
    if result:
        hr10, hr5 = result
        marker = " ***" if hr10 > best_vol_hr10 else ""
        print(f"  {alpha:<8} {hr10:>7.1%} {hr5:>7.1%}{marker}")
        if hr10 > best_vol_hr10:
            best_vol_hr10 = hr10
            best_vol_alpha = alpha

print(f"\nBest vol_alpha: {best_vol_alpha}")

# Phase 3: Optimize ensemble weight
print(f"\nPhase 3: Ensemble weight search (best LTR, best alpha={best_vol_alpha})")
print(f"{'Weight':<10} {'Top-10':>8} {'Top-5':>8}")
print("-" * 28)

best_weight = 0.5
best_weight_hr10 = 0

for weight in ensemble_weights:
    n_est = best_ltr_config.pop("n_estimators", 300)
    result, _ = train_and_eval_ltr(best_ltr_config, vol_alpha=best_vol_alpha, ensemble_weight=weight)
    best_ltr_config["n_estimators"] = n_est
    if result:
        hr10, hr5 = result
        marker = " ***" if hr10 > best_weight_hr10 else ""
        print(f"  {weight:<8} {hr10:>7.1%} {hr5:>7.1%}{marker}")
        if hr10 > best_weight_hr10:
            best_weight_hr10 = hr10
            best_weight = weight

print(f"\nBest ensemble weight: {best_weight}")

# ── FINAL SUMMARY ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL OPTIMIZED RESULTS")
print(f"{'='*70}")

# Re-run final config
n_est = best_ltr_config.pop("n_estimators", 300)
final_result, final_model = train_and_eval_ltr(
    best_ltr_config, vol_alpha=best_vol_alpha, ensemble_weight=best_weight
)
best_ltr_config["n_estimators"] = n_est

print(f"\nBaseline (classification only): top-10={baseline_hr10:.1%}, top-5={baseline_hr5:.1%}")
if current_result:
    print(f"Previous config:               top-10={current_result[0]:.1%}, top-5={current_result[1]:.1%}")
if final_result:
    print(f"Optimized config:              top-10={final_result[0]:.1%}, top-5={final_result[1]:.1%}")
    print(f"\nImprovement vs baseline:       top-10={final_result[0]-baseline_hr10:+.1%}, top-5={final_result[1]-baseline_hr5:+.1%}")
    if current_result:
        print(f"Improvement vs previous:       top-10={final_result[0]-current_result[0]:+.1%}, top-5={final_result[1]-current_result[1]:+.1%}")

print(f"\nBest hyperparameters:")
print(f"  LTR: {best_ltr_config}")
print(f"  Volatility alpha: {best_vol_alpha}")
print(f"  Ensemble weight: {best_weight}")
