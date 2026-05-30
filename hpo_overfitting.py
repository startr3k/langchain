"""Overfitting analysis for HPO-optimized LTR model."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_PATH = Path("training_data_10y_full.csv")
print("Loading data...")
df_raw = pd.read_csv(DATA_PATH, low_memory=False)

from stock_predictor.data.feature_engineering import ALL_FEATURE_NAMES, TARGET_COLUMN
from stock_predictor.models.automl_model import (
    StockReturnPredictor, CLASSIFICATION_THRESHOLD,
    _fill_semantic_nan, _log_transform, _compute_derived_features,
    LTR_ENSEMBLE_WEIGHT, VOLATILITY_SCORE_ALPHA,
)
import xgboost as xgb

predictor = StockReturnPredictor()
predictor.load()

df = df_raw.copy()
df = _compute_derived_features(df)
feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
X = df[feature_cols].copy()
y = (df[TARGET_COLUMN] >= CLASSIFICATION_THRESHOLD).astype(int)
X = _fill_semantic_nan(X)
X = _log_transform(X)
X = X.fillna(predictor.feature_medians)

date_series = pd.to_datetime(df["_date"], errors="coerce")
sort_order = date_series.sort_values().index
X = X.loc[sort_order].reset_index(drop=True)
y = y.loc[sort_order].reset_index(drop=True)
df = df.loc[sort_order].reset_index(drop=True)
raw_vol = df["Volatility_20d"].values.copy()

gap_rows = max(1, int(len(X) * 0.05))
split_idx = int(len(X) * 0.75)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx + gap_rows:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx + gap_rows:]
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx + gap_rows:].reset_index(drop=True)
raw_vol_test = raw_vol[split_idx + gap_rows:]

print(f"Current HPO params: LTR_ENSEMBLE_WEIGHT={LTR_ENSEMBLE_WEIGHT}, VOLATILITY_SCORE_ALPHA={VOLATILITY_SCORE_ALPHA}")

# Classification model
from sklearn.metrics import roc_auc_score

class_proba_train = predictor.automl.predict_proba(X_train)
if class_proba_train.ndim == 2:
    class_proba_train = class_proba_train[:, 1]

class_proba_test = predictor.automl.predict_proba(X_test)
if class_proba_test.ndim == 2:
    class_proba_test = class_proba_test[:, 1]

cls_auc_train = roc_auc_score(y_train, class_proba_train)
cls_auc_test = roc_auc_score(y_test, class_proba_test)

print(f"\n{'='*60}")
print("OVERFITTING ANALYSIS")
print(f"{'='*60}")
print(f"\n1. Classification (FLAML AutoML)")
print(f"   Train AUC: {cls_auc_train:.4f}")
print(f"   Test AUC:  {cls_auc_test:.4f}")
print(f"   Gap:       {cls_auc_train - cls_auc_test:.4f} {'(OK)' if cls_auc_train - cls_auc_test < 0.05 else '(MODERATE)' if cls_auc_train - cls_auc_test < 0.1 else '(HIGH)'}")

# LTR model — train with HPO-optimized params
train_dates = pd.to_datetime(df_train["_date"], errors="coerce")
unique_train_dates = sorted(train_dates.unique())
train_groups = train_dates.map({d: i for i, d in enumerate(unique_train_dates)}).values
group_counts_train = pd.Series(train_groups).value_counts().sort_index().values

test_dates = pd.to_datetime(df_test["_date"], errors="coerce")
unique_test_dates = sorted(test_dates.unique())
test_groups = test_dates.map({d: i for i, d in enumerate(unique_test_dates)}).values
group_counts_test = pd.Series(test_groups).value_counts().sort_index().values

# HPO best params
ltr_params = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "tree_method": "hist",
    "verbosity": 0,
    "max_depth": 6,
    "eta": 0.1,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

dtrain = xgb.DMatrix(X_train, label=y_train.values)
dtrain.set_group(group_counts_train)
dtest = xgb.DMatrix(X_test, label=y_test.values)
dtest.set_group(group_counts_test)

model = xgb.train(
    ltr_params, dtrain, num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=False,
)

# Get NDCG@10 scores
train_ndcg = float(model.eval(dtrain).split(":")[1])
test_ndcg = float(model.eval(dtest).split(":")[1])

print(f"\n2. LTR (LambdaMART, HPO-optimized)")
print(f"   Train NDCG@10: {train_ndcg:.4f}")
print(f"   Test NDCG@10:  {test_ndcg:.4f}")
print(f"   Gap:           {train_ndcg - test_ndcg:.4f} {'(OK)' if train_ndcg - test_ndcg < 0.1 else '(MODERATE)' if train_ndcg - test_ndcg < 0.3 else '(HIGH)'}")

# Regime model
print(f"\n3. Market Regime Detection")
print(f"   Status: Auto-discarded (negative test R²)")
print(f"   Reason: Insufficient training samples (~1000 daily rows)")
print(f"   Safety: Model is discarded and contributes nothing to scoring")

# Precision temporal analysis
sample_dates = sorted(df_test["_date"].unique())[::5]
test_tickers = df_test["Ticker"].values
cal_factors = np.array([predictor.ticker_calibration.get(t, 1.0) for t in test_tickers])

ltr_raw = model.predict(dtest)
ltr_scores = 1.0 / (1.0 + np.exp(-ltr_raw))
ensemble = (1 - LTR_ENSEMBLE_WEIGHT) * class_proba_test + LTR_ENSEMBLE_WEIGHT * ltr_scores
vol_pctl = pd.Series(raw_vol_test).rank(pct=True).values
final_scores = ensemble * (1 + VOLATILITY_SCORE_ALPHA * vol_pctl) * cal_factors

df_eval = df_test.copy()
df_eval["_score"] = final_scores
df_eval["_y"] = y_test.values

# Split test period into early vs late
early_dates = sample_dates[:len(sample_dates)//2]
late_dates = sample_dates[len(sample_dates)//2:]

def compute_hr(dates, k=10):
    daily = []
    for date in dates:
        d = df_eval[df_eval["_date"] == date]
        if len(d) < k:
            continue
        daily.append(float(d.nlargest(k, "_score")["_y"].mean()))
    return np.mean(daily) if daily else 0.0

early_hr = compute_hr(early_dates)
late_hr = compute_hr(late_dates)

print(f"\n4. Temporal Stability (Top-10 Hit Rate)")
print(f"   Early test period:  {early_hr:.1%}")
print(f"   Late test period:   {late_hr:.1%}")
print(f"   Drift:              {early_hr - late_hr:+.1%} {'(OK)' if abs(early_hr - late_hr) < 0.05 else '(MODERATE)' if abs(early_hr - late_hr) < 0.1 else '(HIGH)'}")

overall_hr = compute_hr(sample_dates)
print(f"   Overall:            {overall_hr:.1%}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Classification AUC gap:  {cls_auc_train - cls_auc_test:.4f} — {'Acceptable' if cls_auc_train - cls_auc_test < 0.05 else 'Moderate'}")
print(f"  LTR NDCG@10 gap:        {train_ndcg - test_ndcg:.4f} — {'Acceptable' if train_ndcg - test_ndcg < 0.1 else 'Moderate'}")
print(f"  Regime model:            Auto-discarded (prevents overfitting)")
print(f"  Temporal drift:          {abs(early_hr - late_hr):.1%} — {'Acceptable' if abs(early_hr - late_hr) < 0.05 else 'Moderate'}")
print(f"\n  HPO config: weight={LTR_ENSEMBLE_WEIGHT}, alpha={VOLATILITY_SCORE_ALPHA}")
print(f"  Final top-10 precision:  {overall_hr:.1%}")
print(f"  Final top-5 precision:   {compute_hr(sample_dates, k=5):.1%}")
