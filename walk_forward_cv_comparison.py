"""Walk-forward CV comparison: current features vs cleaned features.

Runs walk-forward CV twice — once with all current features, once with
harmful/redundant features removed — and compares AUC, top-10 hit rate,
and average peak return across 5 market regimes.

Features removed (based on grouped permutation importance analysis):
- treasury_10y: negative permutation importance (AUC drop = -0.014)
- hist_current_ratio: negative permutation importance (AUC drop = -0.010)
- hist_revenue_growth_qoq: negative permutation importance (AUC drop = -0.004)
- gold_return_20d: negative permutation importance (AUC drop = -0.001)
- insider_total_transactions_90d: perfectly correlated with insider_net_buys_90d (r=1.0)
"""

import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from stock_predictor.data.feature_engineering import ALL_FEATURE_NAMES, TARGET_COLUMN
from stock_predictor.models.automl_model import (
    _fill_semantic_nan,
    _log_transform,
    _compute_derived_features,
    CLASSIFICATION_THRESHOLD,
)

FOLDS = [
    {"name": "Fold 1 (train→2019, test 2020-2021)", "train_end": "2019-12-31", "test_start": "2020-04-01", "test_end": "2021-12-31"},
    {"name": "Fold 2 (train→2020, test 2021-2022)", "train_end": "2020-12-31", "test_start": "2021-04-01", "test_end": "2022-12-31"},
    {"name": "Fold 3 (train→2021, test 2022-2023)", "train_end": "2021-12-31", "test_start": "2022-04-01", "test_end": "2023-12-31"},
    {"name": "Fold 4 (train→2022, test 2023-2024)", "train_end": "2022-12-31", "test_start": "2023-04-01", "test_end": "2024-12-31"},
    {"name": "Fold 5 (train→2023, test 2024-2025)", "train_end": "2023-12-31", "test_start": "2024-04-01", "test_end": "2025-12-31"},
]

HARMFUL_FEATURES = [
    "treasury_10y",
    "hist_current_ratio",
    "hist_revenue_growth_qoq",
    "gold_return_20d",
    "insider_total_transactions_90d",
]


def run_cv(df, feature_cols, label):
    """Run walk-forward CV with the given feature set and return results."""
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"# Features: {len(feature_cols)}")
    print(f"{'#'*70}\n")

    all_results = []

    for fold in FOLDS:
        print(f"  {fold['name']}")

        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])

        df_train = df[df["_date_dt"] <= train_end].copy()
        df_test = df[(df["_date_dt"] >= test_start) & (df["_date_dt"] <= test_end)].copy()

        if len(df_train) < 1000 or len(df_test) < 1000:
            print(f"    Skipping: train={len(df_train)}, test={len(df_test)}")
            continue

        X_train = df_train[feature_cols].copy()
        y_train = df_train[TARGET_COLUMN].copy()
        X_test = df_test[feature_cols].copy()
        y_test = df_test[TARGET_COLUMN].copy()

        valid_train = y_train.notna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        valid_test = y_test.notna()
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        df_test_valid = df_test[valid_test]

        y_train = (y_train >= CLASSIFICATION_THRESHOLD).astype(int)
        y_test_binary = (y_test >= CLASSIFICATION_THRESHOLD).astype(int)

        X_train = _fill_semantic_nan(X_train)
        X_train = _log_transform(X_train)
        medians = X_train.median()
        X_train = X_train.fillna(medians)

        X_test = _fill_semantic_nan(X_test)
        X_test = _log_transform(X_test)
        X_test = X_test.fillna(medians)

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        n_total = n_pos + n_neg
        w_neg = n_total / (2.0 * n_neg)
        w_pos = n_total / (2.0 * n_pos)
        sw_train = y_train.map({0: w_neg, 1: w_pos}).values

        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            time_budget=300,
            metric="ap",
            estimator_list=["xgboost", "lgbm"],
            eval_method="cv",
            n_splits=5,
            verbose=0,
            early_stop=True,
            sample_weight=sw_train,
        )

        proba = automl.predict_proba(X_test)
        proba_pos = proba[:, 1] if proba.ndim == 2 else proba

        auc = roc_auc_score(y_test_binary, proba_pos)
        ap = average_precision_score(y_test_binary, proba_pos)

        results = pd.DataFrame({
            "Ticker": df_test_valid["Ticker"].values,
            "Date": df_test_valid["_date"].values,
            "Prob": proba_pos,
            "Target": y_test_binary.values,
            "Peak": df_test_valid["Forward_Max_Return_3M"].values,
        })

        top_by_ticker = results.sort_values("Prob", ascending=False).drop_duplicates("Ticker", keep="first")
        top10 = top_by_ticker.head(10)

        hits = (top10["Peak"] >= 0.20).sum()
        hit_rate = hits / 10
        avg_return = top10["Peak"].mean()

        print(f"    AUC: {auc:.4f}, Hits: {hits}/10 ({hit_rate:.0%}), Avg peak: {avg_return:.1%}, Model: {automl.best_estimator}")

        # Print individual picks
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            hit = "YES" if row["Peak"] >= 0.20 else "NO"
            print(f"      {i}. {row['Ticker']:<8} {row['Prob']:>7.1%} peak={row['Peak']:>7.1%} {hit}")

        all_results.append({
            "fold": fold["name"],
            "auc": auc,
            "ap": ap,
            "hit_rate": hit_rate,
            "hits": hits,
            "avg_return": avg_return,
            "best_model": automl.best_estimator,
        })

    return all_results


def main():
    print("Loading dataset...")
    df = pd.read_csv("training_data_10y_full.csv")
    df["_date_dt"] = pd.to_datetime(df["_date"])

    # Quality filter
    ticker_counts = df.groupby("Ticker").size()
    tickers_2q = set(ticker_counts[ticker_counts >= 126].index)
    has_rev = df.groupby("Ticker")["hist_total_revenue"].apply(
        lambda x: (x.notna() & (x > 0)).mean() > 0.5
    )
    tickers_with_rev = set(has_rev[has_rev].index)
    quality_tickers = tickers_2q & tickers_with_rev
    df = df[df["Ticker"].isin(quality_tickers)].sort_values("_date").reset_index(drop=True)
    print(f"After quality filter: {len(df):,} rows, {df['Ticker'].nunique()} tickers\n")

    # Current feature set
    current_features = [c for c in ALL_FEATURE_NAMES if c in df.columns]

    # Cleaned feature set
    cleaned_features = [c for c in current_features if c not in HARMFUL_FEATURES]

    print(f"Current features: {len(current_features)}")
    print(f"Cleaned features: {len(cleaned_features)} (removed {len(current_features) - len(cleaned_features)})")
    print(f"Removed: {[f for f in HARMFUL_FEATURES if f in current_features]}")

    # Run both
    results_current = run_cv(df, current_features, "CURRENT MODEL (all features)")
    results_cleaned = run_cv(df, cleaned_features, "CLEANED MODEL (harmful features removed)")

    # Comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON: CURRENT vs CLEANED")
    print(f"{'='*80}\n")

    print(f"{'Fold':<45} {'Current AUC':>12} {'Cleaned AUC':>12} {'Curr Hits':>10} {'Clean Hits':>10} {'Curr Ret':>10} {'Clean Ret':>10}")
    print("-" * 112)

    for rc, rcl in zip(results_current, results_cleaned):
        auc_delta = rcl["auc"] - rc["auc"]
        hit_delta = rcl["hits"] - rc["hits"]
        ret_delta = rcl["avg_return"] - rc["avg_return"]
        print(
            f"{rc['fold']:<45} "
            f"{rc['auc']:>12.4f} {rcl['auc']:>12.4f} "
            f"{rc['hits']:>10}/10 {rcl['hits']:>10}/10 "
            f"{rc['avg_return']:>9.1%} {rcl['avg_return']:>9.1%}"
        )

    # Averages
    avg_auc_curr = np.mean([r["auc"] for r in results_current])
    avg_auc_clean = np.mean([r["auc"] for r in results_cleaned])
    avg_hits_curr = sum(r["hits"] for r in results_current)
    avg_hits_clean = sum(r["hits"] for r in results_cleaned)
    avg_ret_curr = np.mean([r["avg_return"] for r in results_current])
    avg_ret_clean = np.mean([r["avg_return"] for r in results_cleaned])

    total_picks = len(results_current) * 10

    print("-" * 112)
    print(
        f"{'AVERAGE':<45} "
        f"{avg_auc_curr:>12.4f} {avg_auc_clean:>12.4f} "
        f"{avg_hits_curr:>8}/{total_picks:>2} {avg_hits_clean:>8}/{total_picks:>2} "
        f"{avg_ret_curr:>9.1%} {avg_ret_clean:>9.1%}"
    )

    print(f"\n--- DELTAS (Cleaned - Current) ---")
    print(f"AUC:       {avg_auc_clean - avg_auc_curr:+.4f}")
    print(f"Hit rate:  {avg_hits_clean - avg_hits_curr:+d}/{total_picks} ({(avg_hits_clean - avg_hits_curr) / total_picks:+.1%})")
    print(f"Avg return: {avg_ret_clean - avg_ret_curr:+.1%}")


if __name__ == "__main__":
    main()
