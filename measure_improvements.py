"""Measure the impact of all 6 recommendations on daily top-10 precision.

Tests each improvement individually and combined to quantify
the gain over the baseline.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────
DATA_PATH = Path("training_data_10y_full.csv")
print("Loading data...")
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Rows: {len(df_raw):,}, Cols: {df_raw.shape[1]}")

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

# ── Prepare features ──────────────────────────────────────────────
df = df_raw.copy()
df = _compute_derived_features(df)

feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
print(f"  Features: {len(feature_cols)}")

X = df[feature_cols].copy()
y = (df[TARGET_COLUMN] >= CLASSIFICATION_THRESHOLD).astype(int)

# Keep raw volatility before transforms
raw_vol = df["Volatility_20d"].values.copy() if "Volatility_20d" in df.columns else None

X = _fill_semantic_nan(X)
X = _log_transform(X)
X = X.fillna(predictor.feature_medians)

# ── Temporal sort & split ─────────────────────────────────────────
date_series = pd.to_datetime(df["_date"], errors="coerce")
sort_order = date_series.sort_values().index
X = X.loc[sort_order].reset_index(drop=True)
y = y.loc[sort_order].reset_index(drop=True)
df = df.loc[sort_order].reset_index(drop=True)
if raw_vol is not None:
    raw_vol = raw_vol[sort_order]

gap_rows = max(1, int(len(X) * 0.05))
split_idx = int(len(X) * 0.75)
X_test = X.iloc[split_idx + gap_rows:]
y_test = y.iloc[split_idx + gap_rows:]
df_test = df.iloc[split_idx + gap_rows:].reset_index(drop=True)
if raw_vol is not None:
    raw_vol_test = raw_vol[split_idx + gap_rows:]
else:
    raw_vol_test = None

print(f"  Test set: {len(X_test):,} rows")
print(f"  Test dates: {df_test['_date'].min()} to {df_test['_date'].max()}")

# ── Compute base scores ──────────────────────────────────────────
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

# Ensemble
ensemble_scores = (1 - LTR_ENSEMBLE_WEIGHT) * class_proba + LTR_ENSEMBLE_WEIGHT * ltr_scores

# ── Set up df_test ────────────────────────────────────────────────
df_test = df_test.copy()
df_test["class_proba"] = class_proba
df_test["ltr_score"] = ltr_scores
df_test["ensemble_score"] = ensemble_scores
df_test["y_true"] = y_test.values
df_test["actual_return"] = df_test[TARGET_COLUMN].values
if raw_vol_test is not None:
    df_test["raw_vol"] = raw_vol_test

# Sample weekly
unique_dates = sorted(df_test["_date"].unique())
sample_dates = unique_dates[::5]


def compute_daily_hit_rate(df_t, score_col, top_k=10):
    """Compute daily top-K hit rate."""
    daily_hits = []
    for date in sample_dates:
        day_df = df_t[df_t["_date"] == date].copy()
        if len(day_df) < top_k:
            continue
        day_df = day_df.sort_values(score_col, ascending=False)
        top = day_df.head(top_k)
        hits = int(top["y_true"].sum())
        daily_hits.append(hits / top_k)
    return np.mean(daily_hits) if daily_hits else 0.0


def compute_daily_hit_rate_with_scores(df_t, scores, top_k=10):
    """Compute daily top-K hit rate from external score array."""
    df_t = df_t.copy()
    df_t["_ext_score"] = scores
    return compute_daily_hit_rate(df_t, "_ext_score", top_k)


# ── 1. BASELINE ──────────────────────────────────────────────────
print("\n" + "="*70)
print("DAILY TOP-10 PRECISION ANALYSIS")
print("="*70)

baseline_10 = compute_daily_hit_rate(df_test, "class_proba", 10)
baseline_5 = compute_daily_hit_rate(df_test, "class_proba", 5)
print(f"\n1. BASELINE (classification only)")
print(f"   Top-10 hit rate: {baseline_10:.1%}")
print(f"   Top-5  hit rate: {baseline_5:.1%}")

# ── 2. LTR ENSEMBLE ─────────────────────────────────────────────
if has_ltr:
    ltr_ens_10 = compute_daily_hit_rate(df_test, "ensemble_score", 10)
    ltr_ens_5 = compute_daily_hit_rate(df_test, "ensemble_score", 5)
    print(f"\n2. + LTR ENSEMBLE (50/50)")
    print(f"   Top-10 hit rate: {ltr_ens_10:.1%} (delta: {ltr_ens_10-baseline_10:+.1%})")
    print(f"   Top-5  hit rate: {ltr_ens_5:.1%} (delta: {ltr_ens_5-baseline_5:+.1%})")
else:
    ltr_ens_10 = baseline_10
    ltr_ens_5 = baseline_5
    print(f"\n2. LTR MODEL NOT PRESENT — skipping (will contribute after retraining)")

# ── 3. VOLATILITY-AWARE SCORING ──────────────────────────────────
if raw_vol_test is not None:
    vol_pctl = pd.Series(raw_vol_test).rank(pct=True).values
    vol_scores = ensemble_scores * (1 + VOLATILITY_SCORE_ALPHA * vol_pctl)
    vol_10 = compute_daily_hit_rate_with_scores(df_test, vol_scores, 10)
    vol_5 = compute_daily_hit_rate_with_scores(df_test, vol_scores, 5)
    print(f"\n3. + VOLATILITY-AWARE SCORING (alpha={VOLATILITY_SCORE_ALPHA})")
    print(f"   Top-10 hit rate: {vol_10:.1%} (delta vs baseline: {vol_10-baseline_10:+.1%})")
    print(f"   Top-5  hit rate: {vol_5:.1%} (delta vs baseline: {vol_5-baseline_5:+.1%})")
else:
    vol_scores = ensemble_scores.copy()
    vol_10 = ltr_ens_10
    vol_5 = ltr_ens_5
    print("\n3. VOLATILITY DATA NOT AVAILABLE — skipping")

# ── 4. MARKET REGIME DETECTION ────────────────────────────────────
# Train regime model on training data
print(f"\n4. MARKET REGIME DETECTION")
regime_features = [
    "vix_close", "sp500_return_20d", "sp500_return_60d",
    "sp500_volatility_20d", "yield_curve_spread",
    "treasury_3m", "dollar_index_return_20d",
]
regime_features = [f for f in regime_features if f in df.columns]

if len(regime_features) >= 3:
    # Compute regime confidence for test dates
    train_df_for_regime = df.iloc[:split_idx].copy()
    train_X_for_regime = X.iloc[:split_idx].copy()
    train_y_for_regime = y.iloc[:split_idx].copy()

    try:
        train_proba_regime = predictor.automl.predict_proba(train_X_for_regime)
        if train_proba_regime.ndim == 2:
            train_proba_regime = train_proba_regime[:, 1]

        train_df_for_regime = train_df_for_regime.copy()
        train_df_for_regime["_proba"] = train_proba_regime
        train_df_for_regime["_y"] = train_y_for_regime.values

        daily_stats = []
        for date, grp in train_df_for_regime.groupby("_date"):
            if len(grp) < 10:
                continue
            top10_idx = grp["_proba"].nlargest(10).index
            hit_rate = float(grp.loc[top10_idx, "_y"].mean())
            macro_vals = grp[regime_features].iloc[0].to_dict()
            macro_vals["_hit_rate"] = hit_rate
            daily_stats.append(macro_vals)

        if len(daily_stats) >= 50:
            from sklearn.ensemble import GradientBoostingRegressor
            regime_df = pd.DataFrame(daily_stats)
            X_regime = regime_df[regime_features].fillna(0)
            y_regime = regime_df["_hit_rate"]

            r_split = int(len(X_regime) * 0.8)
            X_r_train = X_regime.iloc[:r_split]
            y_r_train = y_regime.iloc[:r_split]
            X_r_test = X_regime.iloc[r_split:]
            y_r_test = y_regime.iloc[r_split:]

            regime_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                min_samples_leaf=10, subsample=0.8, random_state=42,
            )
            regime_model.fit(X_r_train, y_r_train)

            r2_train = regime_model.score(X_r_train, y_r_train)
            r2_test = regime_model.score(X_r_test, y_r_test)
            print(f"   Regime model R²: train={r2_train:.4f}, test={r2_test:.4f}, gap={r2_train-r2_test:.4f}")

            # Predict regime confidence on test dates
            test_regime_conf = []
            for date in sample_dates:
                day_df = df_test[df_test["_date"] == date]
                if len(day_df) == 0:
                    test_regime_conf.append(0.5)
                    continue
                macro = day_df[regime_features].iloc[0:1].fillna(0)
                conf = float(regime_model.predict(macro)[0])
                test_regime_conf.append(max(0.0, min(1.0, conf)))

            # Show regime confidence distribution
            conf_arr = np.array(test_regime_conf)
            print(f"   Regime confidence range: {conf_arr.min():.2f} - {conf_arr.max():.2f}")
            print(f"   Mean regime confidence: {conf_arr.mean():.2f}")
            high_conf_days = sum(1 for c in conf_arr if c >= 0.6)
            low_conf_days = sum(1 for c in conf_arr if c < 0.4)
            print(f"   High confidence days (≥60%): {high_conf_days}/{len(conf_arr)}")
            print(f"   Low confidence days (<40%): {low_conf_days}/{len(conf_arr)}")

            # Hit rate on high vs low confidence days
            high_hits = []
            low_hits = []
            for date, conf in zip(sample_dates, test_regime_conf):
                day_df = df_test[df_test["_date"] == date].copy()
                if len(day_df) < 10:
                    continue
                day_df = day_df.sort_values("ensemble_score", ascending=False)
                top10 = day_df.head(10)
                hr = float(top10["y_true"].mean())
                if conf >= 0.6:
                    high_hits.append(hr)
                elif conf < 0.4:
                    low_hits.append(hr)
            if high_hits:
                print(f"   Hit rate on high-confidence days: {np.mean(high_hits):.1%}")
            if low_hits:
                print(f"   Hit rate on low-confidence days: {np.mean(low_hits):.1%}")
        else:
            print("   Too few training dates for regime model")
    except Exception as e:
        print(f"   Regime model error: {e}")
else:
    print("   Insufficient macro features")

# ── 5. PER-TICKER CALIBRATION ─────────────────────────────────────
print(f"\n5. PER-TICKER CALIBRATION")
if "Ticker" in df.columns:
    # Compute calibration from training data
    train_df_cal = df.iloc[:split_idx].copy()
    train_X_cal = X.iloc[:split_idx].copy()
    train_y_cal = y.iloc[:split_idx].copy()

    try:
        train_proba_cal = predictor.automl.predict_proba(train_X_cal)
        if train_proba_cal.ndim == 2:
            train_proba_cal = train_proba_cal[:, 1]

        train_df_cal = train_df_cal.copy()
        train_df_cal["_proba"] = train_proba_cal
        train_df_cal["_y"] = train_y_cal.values

        ticker_stats = {}
        for date, grp in train_df_cal.groupby("_date"):
            if len(grp) < 10:
                continue
            top10_idx = grp["_proba"].nlargest(10).index
            top10 = grp.loc[top10_idx]
            for _, row in top10.iterrows():
                ticker = row["Ticker"]
                if ticker not in ticker_stats:
                    ticker_stats[ticker] = {"appearances": 0, "hits": 0}
                ticker_stats[ticker]["appearances"] += 1
                ticker_stats[ticker]["hits"] += int(row["_y"])

        # Compute calibration factors
        ticker_calibration = {}
        penalized_count = 0
        for ticker, stats in ticker_stats.items():
            if stats["appearances"] >= 5:
                hit_rate = stats["hits"] / stats["appearances"]
                if hit_rate < 0.30:
                    factor = max(0.5, hit_rate / 0.50)
                    ticker_calibration[ticker] = round(factor, 3)
                    penalized_count += 1

        print(f"   Tickers tracked: {len(ticker_stats)}")
        print(f"   Tickers penalized (hit rate <30%): {penalized_count}")
        if ticker_calibration:
            worst = sorted(ticker_calibration.items(), key=lambda x: x[1])[:10]
            print(f"   Worst offenders:")
            for t, f in worst:
                s = ticker_stats[t]
                print(f"     {t}: factor={f:.3f}, hit_rate={s['hits']}/{s['appearances']} ({s['hits']/s['appearances']:.0%})")

        # Apply calibration to test scores
        test_tickers = df_test["Ticker"].values
        cal_factors = np.array([ticker_calibration.get(t, 1.0) for t in test_tickers])
        cal_scores = vol_scores * cal_factors
        cal_10 = compute_daily_hit_rate_with_scores(df_test, cal_scores, 10)
        cal_5 = compute_daily_hit_rate_with_scores(df_test, cal_scores, 5)
        print(f"\n   After calibration:")
        print(f"   Top-10 hit rate: {cal_10:.1%} (delta vs baseline: {cal_10-baseline_10:+.1%})")
        print(f"   Top-5  hit rate: {cal_5:.1%} (delta vs baseline: {cal_5-baseline_5:+.1%})")

    except Exception as e:
        print(f"   Calibration error: {e}")
        cal_scores = vol_scores
        cal_10 = vol_10
        cal_5 = vol_5
else:
    cal_scores = vol_scores
    cal_10 = vol_10
    cal_5 = vol_5
    print("   No Ticker column")

# ── 6. TOP-5 DEFAULT ─────────────────────────────────────────────
print(f"\n6. TOP-5 DEFAULT (narrowing picks)")
print(f"   Top-5  hit rate (baseline): {baseline_5:.1%}")
print(f"   Top-5  hit rate (all improvements): {cal_5:.1%}")
print(f"   Top-10 hit rate (all improvements): {cal_10:.1%}")

# ── SUMMARY ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY OF IMPROVEMENTS")
print("="*70)
print(f"{'Configuration':<45} {'Top-10':>8} {'Top-5':>8}")
print("-" * 63)
print(f"{'Baseline (classification only)':<45} {baseline_10:>7.1%} {baseline_5:>7.1%}")
if has_ltr:
    print(f"{'+ LTR Ensemble (50/50)':<45} {ltr_ens_10:>7.1%} {ltr_ens_5:>7.1%}")
else:
    print(f"{'+ LTR Ensemble (NOT SAVED — shows after retrain)':<45} {'—':>8} {'—':>8}")
print(f"{'+ Volatility-aware scoring':<45} {vol_10:>7.1%} {vol_5:>7.1%}")
print(f"{'+ All improvements combined':<45} {cal_10:>7.1%} {cal_5:>7.1%}")
print("-" * 63)
print(f"{'Total improvement (top-10)':<45} {cal_10-baseline_10:>+7.1%}")
print(f"{'Total improvement (top-5)':<45} {cal_5-baseline_5:>+7.1%}")

# ── OVERFITTING ANALYSIS ─────────────────────────────────────────
print("\n" + "="*70)
print("OVERFITTING ANALYSIS")
print("="*70)

# Classification model
y_proba_train = predictor.automl.predict_proba(X.iloc[:split_idx])
if y_proba_train.ndim == 2:
    y_proba_train = y_proba_train[:, 1]
y_train = y.iloc[:split_idx]

from sklearn.metrics import roc_auc_score, average_precision_score
auc_train = roc_auc_score(y_train, y_proba_train)
auc_test = roc_auc_score(y_test, class_proba)
ap_train = average_precision_score(y_train, y_proba_train)
ap_test = average_precision_score(y_test, class_proba)

print(f"\nClassification Model (FLAML):")
print(f"  AUC-ROC:  train={auc_train:.4f}, test={auc_test:.4f}, gap={auc_train-auc_test:.4f}")
print(f"  Avg Prec: train={ap_train:.4f}, test={ap_test:.4f}, gap={ap_train-ap_test:.4f}")
if auc_train - auc_test > 0.10:
    print(f"  WARNING: AUC gap > 0.10 indicates significant overfitting")
else:
    print(f"  OK: AUC gap is acceptable")

# LTR model
if has_ltr:
    print(f"\nLTR Model (LambdaMART):")
    # We don't have the evals_result from training, but we can check
    # if it's saved in the model
    print(f"  (NDCG@10 train/test gaps reported during training)")
else:
    print(f"\nLTR Model: NOT PRESENT (will be trained on next model training run)")

# Regime model
if 'regime_model' in dir() and regime_model is not None:
    print(f"\nRegime Model (GradientBoosting):")
    print(f"  R²: train={r2_train:.4f}, test={r2_test:.4f}, gap={r2_train-r2_test:.4f}")
    if r2_train - r2_test > 0.20:
        print(f"  WARNING: R² gap > 0.20 indicates overfitting")
    elif r2_test < 0:
        print(f"  WARNING: Negative R² on test — model is worse than baseline")
    else:
        print(f"  OK: R² gap is acceptable")

# Daily hit rate train vs test temporal analysis
print(f"\nTemporal Hit Rate Analysis:")
# Split test dates into halves
mid = len(sample_dates) // 2
early_dates = sample_dates[:mid]
late_dates = sample_dates[mid:]

early_hits = []
for date in early_dates:
    day_df = df_test[df_test["_date"] == date].copy()
    if len(day_df) < 10:
        continue
    day_df = day_df.sort_values("class_proba", ascending=False)
    top10 = day_df.head(10)
    early_hits.append(float(top10["y_true"].mean()))

late_hits = []
for date in late_dates:
    day_df = df_test[df_test["_date"] == date].copy()
    if len(day_df) < 10:
        continue
    day_df = day_df.sort_values("class_proba", ascending=False)
    top10 = day_df.head(10)
    late_hits.append(float(top10["y_true"].mean()))

if early_hits and late_hits:
    print(f"  Early test period hit rate: {np.mean(early_hits):.1%} ({len(early_hits)} days)")
    print(f"  Late test period hit rate:  {np.mean(late_hits):.1%} ({len(late_hits)} days)")
    drift = np.mean(late_hits) - np.mean(early_hits)
    if abs(drift) > 0.10:
        print(f"  WARNING: {abs(drift):.1%} drift between early and late test — potential model staleness")
    else:
        print(f"  OK: No significant temporal drift")

print("\n" + "="*70)
print("DONE")
print("="*70)
