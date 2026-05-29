import pandas as pd
import numpy as np
from stock_predictor.models.automl_model import (
    StockReturnPredictor,
    _fill_semantic_nan,
    _log_transform,
    _compute_derived_features,
)
from stock_predictor.data.feature_engineering import TARGET_COLUMN

df = pd.read_csv("training_data_10y_full.csv")
predictor = StockReturnPredictor()
predictor.load()
print(f"Model features: {len(predictor.feature_names)}")

# Quality filter (same as training)
ticker_counts = df.groupby("Ticker").size()
tickers_2q = set(ticker_counts[ticker_counts >= 126].index)
has_rev = df.groupby("Ticker")["hist_total_revenue"].apply(
    lambda x: (x.notna() & (x > 0)).mean() > 0.5
)
tickers_with_rev = set(has_rev[has_rev].index)
quality_tickers = tickers_2q & tickers_with_rev
dq = df[df["Ticker"].isin(quality_tickers)].sort_values("_date").reset_index(drop=True)
print(f"After filter: {len(dq):,} rows, {dq['Ticker'].nunique()} tickers")

available_cols = [c for c in predictor.feature_names if c in dq.columns]
X = dq[available_cols].copy()
y = dq[TARGET_COLUMN]
X = _fill_semantic_nan(X)
X = _log_transform(X)
X = _compute_derived_features(X)
X = X[[c for c in predictor.feature_names if c in X.columns]]

split_idx = int(len(X) * 0.75)
gap_rows = max(1, int(len(X) * 0.05))
X_test = X.iloc[split_idx + gap_rows :]
y_test = y.iloc[split_idx + gap_rows :]
df_test = dq.iloc[split_idx + gap_rows :]

if predictor.feature_medians is not None:
    X_test = X_test.fillna(predictor.feature_medians)
else:
    X_test = X_test.fillna(0.0)

proba = predictor.automl.predict_proba(X_test)
proba_pos = proba[:, 1] if proba.ndim == 2 else proba

results = pd.DataFrame(
    {
        "Ticker": df_test["Ticker"].values,
        "Date": df_test["_date"].values,
        "Prob": proba_pos,
        "Target": y_test.values,
        "Peak": df_test["Forward_Max_Return_3M"].values,
    }
)

tb = results.sort_values("Prob", ascending=False).drop_duplicates("Ticker", keep="first")
t10 = tb.head(10)

print(f"\nThreshold: {predictor.optimal_threshold:.4f}")
print(f"\n{'Rank':<5} {'Ticker':<8} {'Prob':>8} {'Peak Return':>12} {'Hit?':>6}")
print("-" * 45)
for i, (_, row) in enumerate(t10.iterrows(), 1):
    hit = "YES" if row["Peak"] >= 0.20 else "NO"
    print(f"{i:<5} {row['Ticker']:<8} {row['Prob']:>7.1%} {row['Peak']:>11.1%} {hit:>6}")

h10 = (t10["Peak"] >= 0.20).sum()
h5 = (tb.head(5)["Peak"] >= 0.20).sum()
h3 = (tb.head(3)["Peak"] >= 0.20).sum()
print(f"\nTop-10 hit rate: {h10}/10 ({h10*10}%)")
print(f"Top-5 hit rate: {h5}/5 ({h5*20}%)")
print(f"Top-3 hit rate: {h3}/3")
print(f"Avg peak return (top 10): {t10['Peak'].mean():.1%}")
