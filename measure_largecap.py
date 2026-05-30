"""Measure daily top-10 precision filtered to >=1B market cap stocks."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Load market cap cache ─────────────────────────────────────────
with open("market_cap_cache.json") as f:
    mcap_cache = json.load(f)

above_1b = {t for t, v in mcap_cache.items() if v and v >= 1e9}
print(f"Tickers with market cap >= $1B: {len(above_1b)}")

# ── Load data ──────────────────────────────────────────────────────
DATA_PATH = Path("training_data_10y_full.csv")
print("Loading data...")
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Total rows: {len(df_raw):,}")

# Filter to >=1B tickers
df_raw_1b = df_raw[df_raw["Ticker"].isin(above_1b)].copy()
print(f"  Rows with >= $1B market cap: {len(df_raw_1b):,} ({len(df_raw_1b)/len(df_raw)*100:.1f}%)")
print(f"  Unique tickers >= $1B: {df_raw_1b['Ticker'].nunique()}")

# ── Setup model ────────────────────────────────────────────────────
from stock_predictor.data.feature_engineering import ALL_FEATURE_NAMES, TARGET_COLUMN
from stock_predictor.models.automl_model import (
    StockReturnPredictor,
    CLASSIFICATION_THRESHOLD,
    LTR_ENSEMBLE_WEIGHT,
    VOLATILITY_SCORE_ALPHA,
    _fill_semantic_nan,
    _log_transform,
    _compute_derived_features,
)
import xgboost as xgb

predictor = StockReturnPredictor()
predictor.load()

# ── Run analysis on FULL universe and 1B-filtered ─────────────────
for label, df_input in [("FULL UNIVERSE (>=100M)", df_raw), ("LARGE CAP (>=1B)", df_raw_1b)]:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    df = df_input.copy()
    df = _compute_derived_features(df)

    feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].copy()
    y = (df[TARGET_COLUMN] >= CLASSIFICATION_THRESHOLD).astype(int)

    X = _fill_semantic_nan(X)
    X = _log_transform(X)
    X = X.fillna(predictor.feature_medians)

    # Temporal sort & split
    date_series = pd.to_datetime(df["_date"], errors="coerce")
    sort_order = date_series.sort_values().index
    X = X.loc[sort_order].reset_index(drop=True)
    y = y.loc[sort_order].reset_index(drop=True)
    df = df.loc[sort_order].reset_index(drop=True)

    raw_vol = df["Volatility_20d"].values.copy() if "Volatility_20d" in df.columns else None

    gap_rows = max(1, int(len(X) * 0.05))
    split_idx = int(len(X) * 0.75)
    X_test = X.iloc[split_idx + gap_rows:]
    y_test = y.iloc[split_idx + gap_rows:]
    df_test = df.iloc[split_idx + gap_rows:].reset_index(drop=True)
    raw_vol_test = raw_vol[split_idx + gap_rows:] if raw_vol is not None else None

    print(f"  Test set: {len(X_test):,} rows, {df_test['Ticker'].nunique()} tickers")
    print(f"  Test dates: {df_test['_date'].min()} to {df_test['_date'].max()}")

    # Compute scores
    y_proba = predictor.automl.predict_proba(X_test)
    if y_proba.ndim == 2:
        class_proba = y_proba[:, 1]
    else:
        class_proba = y_proba

    if predictor.ltr_model is not None:
        dtest = xgb.DMatrix(X_test)
        ltr_raw = predictor.ltr_model.predict(dtest)
        ltr_scores = 1.0 / (1.0 + np.exp(-ltr_raw))
        has_ltr = True
    else:
        ltr_scores = np.zeros(len(X_test))
        has_ltr = False

    ensemble_scores = (1 - LTR_ENSEMBLE_WEIGHT) * class_proba + LTR_ENSEMBLE_WEIGHT * ltr_scores

    # Volatility-aware scores
    if raw_vol_test is not None and len(raw_vol_test) > 1:
        vol_pctl = pd.Series(raw_vol_test).rank(pct=True).values
        vol_scores = ensemble_scores * (1 + VOLATILITY_SCORE_ALPHA * vol_pctl)
    else:
        vol_scores = ensemble_scores.copy()

    # Ticker calibration
    test_tickers = df_test["Ticker"].values
    cal_factors = np.array([predictor.ticker_calibration.get(t, 1.0) for t in test_tickers])
    cal_scores = vol_scores * cal_factors

    df_test = df_test.copy()
    df_test["class_proba"] = class_proba
    df_test["ltr_score"] = ltr_scores
    df_test["ensemble_score"] = ensemble_scores
    df_test["vol_score"] = vol_scores
    df_test["cal_score"] = cal_scores
    df_test["y_true"] = y_test.values
    df_test["actual_return"] = df_test[TARGET_COLUMN].values

    # Sample weekly
    unique_dates = sorted(df_test["_date"].unique())
    sample_dates = unique_dates[::5]

    def hit_rate(score_col, top_k=10):
        daily = []
        for date in sample_dates:
            day_df = df_test[df_test["_date"] == date].copy()
            if len(day_df) < top_k:
                continue
            day_df = day_df.sort_values(score_col, ascending=False)
            top = day_df.head(top_k)
            daily.append(float(top["y_true"].mean()))
        return np.mean(daily) if daily else 0.0

    # Count dates with enough stocks
    valid_10 = sum(1 for d in sample_dates if len(df_test[df_test["_date"] == d]) >= 10)
    valid_5 = sum(1 for d in sample_dates if len(df_test[df_test["_date"] == d]) >= 5)
    print(f"  Dates with >=10 stocks: {valid_10}/{len(sample_dates)}")
    print(f"  Dates with >=5 stocks: {valid_5}/{len(sample_dates)}")

    b10 = hit_rate("class_proba", 10)
    b5 = hit_rate("class_proba", 5)
    e10 = hit_rate("ensemble_score", 10)
    e5 = hit_rate("ensemble_score", 5)
    v10 = hit_rate("vol_score", 10)
    v5 = hit_rate("vol_score", 5)
    c10 = hit_rate("cal_score", 10)
    c5 = hit_rate("cal_score", 5)

    print(f"\n  {'Configuration':<40} {'Top-10':>8} {'Top-5':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Baseline (classification only)':<40} {b10:>7.1%} {b5:>7.1%}")
    if has_ltr:
        print(f"  {'+ LTR Ensemble (50/50)':<40} {e10:>7.1%} {e5:>7.1%}")
    print(f"  {'+ Volatility-aware scoring':<40} {v10:>7.1%} {v5:>7.1%}")
    print(f"  {'+ All improvements':<40} {c10:>7.1%} {c5:>7.1%}")
    print(f"  {'-'*58}")
    print(f"  {'Total improvement (top-10)':<40} {c10-b10:>+7.1%}")
    print(f"  {'Total improvement (top-5)':<40} {c5-b5:>+7.1%}")

    # Positive rate in test set
    pos_rate = y_test.mean()
    print(f"\n  Positive rate (>=20% return): {pos_rate:.1%}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
