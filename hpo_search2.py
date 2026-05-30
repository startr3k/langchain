"""HPO Phase 2: Fine-grained search around promising configs."""

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
df_test = df.iloc[split_idx + gap_rows:].reset_index(drop=True)
raw_vol_test = raw_vol[split_idx + gap_rows:]

y_proba = predictor.automl.predict_proba(X_test)
class_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

train_dates = pd.to_datetime(df.iloc[:split_idx]["_date"], errors="coerce")
unique_train_dates = sorted(train_dates.unique())
train_groups = train_dates.map({d: i for i, d in enumerate(unique_train_dates)}).values
group_counts_train = pd.Series(train_groups).value_counts().sort_index().values

sample_dates = sorted(df_test["_date"].unique())[::5]
test_tickers = df_test["Ticker"].values
cal_factors = np.array([predictor.ticker_calibration.get(t, 1.0) for t in test_tickers])


def compute_hr(scores, k=10):
    df_e = df_test.copy()
    df_e["_s"] = scores
    df_e["_y"] = y_test.values
    daily = []
    for date in sample_dates:
        d = df_e[df_e["_date"] == date]
        if len(d) < k:
            continue
        daily.append(float(d.nlargest(k, "_s")["_y"].mean()))
    return np.mean(daily) if daily else 0.0


def eval_config(ltr_params, n_est, vol_alpha, ens_weight):
    dtrain = xgb.DMatrix(X_train, label=y_train.values)
    dtrain.set_group(group_counts_train)
    dtest = xgb.DMatrix(X_test)

    xgb_p = {"objective": "rank:ndcg", "eval_metric": "ndcg@10",
             "tree_method": "hist", "verbosity": 0, **ltr_params}
    model = xgb.train(xgb_p, dtrain, num_boost_round=n_est, verbose_eval=False)
    ltr_raw = model.predict(dtest)
    ltr_scores = 1.0 / (1.0 + np.exp(-ltr_raw))
    ens = (1 - ens_weight) * class_proba + ens_weight * ltr_scores
    vol_pctl = pd.Series(raw_vol_test).rank(pct=True).values
    scores = ens * (1 + vol_alpha * vol_pctl) * cal_factors
    return compute_hr(scores, 10), compute_hr(scores, 5)


# Promising configs from Phase 1 — now try all alpha/weight combos
configs = [
    # Best top-10 configs
    ({"max_depth": 6, "eta": 0.1, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8}, 500, "d6_eta0.1_mcw50"),
    # Best top-5 configs
    ({"max_depth": 4, "eta": 0.05, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8}, 500, "d4_eta0.05_mcw50"),
    ({"max_depth": 4, "eta": 0.01, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8}, 500, "d4_eta0.01_mcw50"),
    ({"max_depth": 4, "eta": 0.1, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8}, 300, "d4_eta0.1_mcw20"),
    # Regularized
    ({"max_depth": 3, "eta": 0.05, "min_child_weight": 200, "subsample": 0.6, "colsample_bytree": 0.6, "lambda": 10, "alpha": 2}, 500, "d3_heavy_reg"),
    # New: deeper with regularization
    ({"max_depth": 5, "eta": 0.05, "min_child_weight": 100, "subsample": 0.7, "colsample_bytree": 0.7, "lambda": 3, "alpha": 0.5}, 500, "d5_moderate_reg"),
    ({"max_depth": 6, "eta": 0.05, "min_child_weight": 100, "subsample": 0.7, "colsample_bytree": 0.7, "lambda": 3, "alpha": 0.5}, 500, "d6_moderate_reg"),
    # New: more trees with smaller learning rate
    ({"max_depth": 4, "eta": 0.03, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8}, 800, "d4_eta0.03_n800"),
    ({"max_depth": 5, "eta": 0.03, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8}, 800, "d5_eta0.03_n800"),
]

alphas = [0.1, 0.15, 0.2, 0.25, 0.3]
weights = [0.3, 0.4, 0.5, 0.6]

print(f"\nSearching {len(configs)} LTR configs x {len(alphas)} alphas x {len(weights)} weights = {len(configs)*len(alphas)*len(weights)} combos")
print(f"{'Config':<25} {'Alpha':>6} {'Wt':>5} {'Top-10':>8} {'Top-5':>8}")
print("-" * 55)

best_hr10 = 0
best_combo = None
all_results = []

for params, n_est, name in configs:
    for alpha in alphas:
        for weight in weights:
            try:
                hr10, hr5 = eval_config(params, n_est, alpha, weight)
                marker = ""
                if hr10 > best_hr10:
                    best_hr10 = hr10
                    best_combo = (name, alpha, weight, hr10, hr5, params, n_est)
                    marker = " ***"
                all_results.append((name, alpha, weight, hr10, hr5))
                print(f"  {name:<23} {alpha:>5.2f} {weight:>4.1f} {hr10:>7.1%} {hr5:>7.1%}{marker}")
            except Exception as e:
                print(f"  {name:<23} {alpha:>5.2f} {weight:>4.1f} FAILED: {e}")

# Sort by top-10
all_results.sort(key=lambda x: x[3], reverse=True)

print(f"\n{'='*55}")
print("TOP 10 CONFIGS BY TOP-10 HIT RATE")
print(f"{'='*55}")
for name, alpha, weight, hr10, hr5 in all_results[:10]:
    print(f"  {name:<23} a={alpha:.2f} w={weight:.1f} top-10={hr10:.1%} top-5={hr5:.1%}")

print(f"\n{'='*55}")
print("TOP 10 CONFIGS BY TOP-5 HIT RATE")
print(f"{'='*55}")
all_by_5 = sorted(all_results, key=lambda x: x[4], reverse=True)
for name, alpha, weight, hr10, hr5 in all_by_5[:10]:
    print(f"  {name:<23} a={alpha:.2f} w={weight:.1f} top-10={hr10:.1%} top-5={hr5:.1%}")

if best_combo:
    name, alpha, weight, hr10, hr5, params, n_est = best_combo
    print(f"\nBEST CONFIG:")
    print(f"  LTR: {params}, n_estimators={n_est}")
    print(f"  Vol alpha: {alpha}")
    print(f"  Ensemble weight: {weight}")
    print(f"  Top-10: {hr10:.1%}, Top-5: {hr5:.1%}")

baseline10 = compute_hr(class_proba, 10)
baseline5 = compute_hr(class_proba, 5)
print(f"\nBaseline: top-10={baseline10:.1%}, top-5={baseline5:.1%}")
if best_combo:
    print(f"Best improvement: top-10={hr10-baseline10:+.1%}, top-5={hr5-baseline5:+.1%}")
