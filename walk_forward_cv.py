"""Walk-forward cross-validation for the stock predictor model.

Uses expanding training windows with 2-year test periods and 3-month gaps
to evaluate model performance across different market regimes.
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

# Walk-forward folds:
# Each fold trains on all data up to a cutoff, then tests on the next 2 years
# with a 63-day gap to prevent forward-return overlap.
FOLDS = [
    {"name": "Fold 1 (train→2019, test 2020-2021)", "train_end": "2019-12-31", "test_start": "2020-04-01", "test_end": "2021-12-31"},
    {"name": "Fold 2 (train→2020, test 2021-2022)", "train_end": "2020-12-31", "test_start": "2021-04-01", "test_end": "2022-12-31"},
    {"name": "Fold 3 (train→2021, test 2022-2023)", "train_end": "2021-12-31", "test_start": "2022-04-01", "test_end": "2023-12-31"},
    {"name": "Fold 4 (train→2022, test 2023-2024)", "train_end": "2022-12-31", "test_start": "2023-04-01", "test_end": "2024-12-31"},
    {"name": "Fold 5 (train→2023, test 2024-2025)", "train_end": "2023-12-31", "test_start": "2024-04-01", "test_end": "2025-12-31"},
]


def run_walk_forward():
    df = pd.read_csv("training_data_10y_full.csv")
    df["_date_dt"] = pd.to_datetime(df["_date"])
    print(f"Dataset: {len(df):,} rows, {df['Ticker'].nunique()} tickers\n")

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

    all_results = []

    for fold in FOLDS:
        print(f"{'='*60}")
        print(f"{fold['name']}")
        print(f"{'='*60}")

        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])

        df_train = df[df["_date_dt"] <= train_end].copy()
        df_test = df[(df["_date_dt"] >= test_start) & (df["_date_dt"] <= test_end)].copy()

        if len(df_train) < 1000 or len(df_test) < 1000:
            print(f"  Skipping: train={len(df_train)}, test={len(df_test)}")
            continue

        # Prepare features
        feature_cols = [c for c in ALL_FEATURE_NAMES if c in df_train.columns]
        X_train = df_train[feature_cols].copy()
        y_train = df_train[TARGET_COLUMN].copy()
        X_test = df_test[feature_cols].copy()
        y_test = df_test[TARGET_COLUMN].copy()

        # Drop rows with missing target
        valid_train = y_train.notna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        valid_test = y_test.notna()
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        df_test = df_test[valid_test]

        # Binary classification target
        y_train = (y_train >= CLASSIFICATION_THRESHOLD).astype(int)
        y_test_binary = (y_test >= CLASSIFICATION_THRESHOLD).astype(int)

        # Transforms
        X_train = _fill_semantic_nan(X_train)
        X_train = _log_transform(X_train)
        medians = X_train.median()
        X_train = X_train.fillna(medians)

        X_test = _fill_semantic_nan(X_test)
        X_test = _log_transform(X_test)
        X_test = X_test.fillna(medians)

        # Balanced class weights
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        n_total = n_pos + n_neg
        w_neg = n_total / (2.0 * n_neg)
        w_pos = n_total / (2.0 * n_pos)
        sw_train = y_train.map({0: w_neg, 1: w_pos}).values

        print(f"  Train: {len(X_train):,} rows ({n_pos} pos, {n_neg} neg)")
        print(f"  Test:  {len(X_test):,} rows")

        # Train with FLAML
        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            time_budget=300,  # 5 min per fold
            metric="ap",
            estimator_list=["xgboost", "lgbm"],
            eval_method="cv",
            n_splits=5,
            verbose=0,
            early_stop=True,
            sample_weight=sw_train,
        )

        # Predict on test
        proba = automl.predict_proba(X_test)
        proba_pos = proba[:, 1] if proba.ndim == 2 else proba

        auc = roc_auc_score(y_test_binary, proba_pos)
        ap = average_precision_score(y_test_binary, proba_pos)

        # Find optimal threshold
        prec_curve, rec_curve, thresholds = precision_recall_curve(y_test_binary, proba_pos)
        best_thresh = 0.5
        best_prec = 0.0
        for p, r, t in zip(prec_curve[:-1], rec_curve[:-1], thresholds):
            n_pred_pos = (proba_pos >= t).sum()
            if n_pred_pos >= 50 and p > best_prec:
                best_prec = p
                best_thresh = t

        print(f"  AUC: {auc:.4f}, AP: {ap:.4f}, Best model: {automl.best_estimator}")
        print(f"  Threshold: {best_thresh:.4f}")

        # Top-10 picks
        results = pd.DataFrame({
            "Ticker": df_test["Ticker"].values,
            "Date": df_test["_date"].values,
            "Prob": proba_pos,
            "Target": y_test_binary.values,
            "Peak": df_test["Forward_Max_Return_3M"].values,
        })

        top_by_ticker = results.sort_values("Prob", ascending=False).drop_duplicates("Ticker", keep="first")
        top10 = top_by_ticker.head(10)

        print(f"\n  Top 10 Picks:")
        print(f"  {'Rank':<5} {'Ticker':<8} {'Prob':>8} {'Peak':>10} {'Hit?':>6}")
        print(f"  {'-'*40}")
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            hit = "YES" if row["Peak"] >= 0.20 else "NO"
            print(f"  {i:<5} {row['Ticker']:<8} {row['Prob']:>7.1%} {row['Peak']:>9.1%} {hit:>6}")

        hits = (top10["Peak"] >= 0.20).sum()
        hit_rate = hits / 10
        avg_return = top10["Peak"].mean()
        print(f"\n  Hit rate: {hits}/10 ({hit_rate:.0%})")
        print(f"  Avg peak return: {avg_return:.1%}\n")

        all_results.append({
            "fold": fold["name"],
            "auc": auc,
            "ap": ap,
            "hit_rate": hit_rate,
            "hits": hits,
            "avg_return": avg_return,
            "best_model": automl.best_estimator,
            "train_size": len(X_train),
            "test_size": len(X_test),
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD CV SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Fold':<45} {'AUC':>6} {'Hit Rate':>10} {'Avg Return':>12}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['fold']:<45} {r['auc']:>6.4f} {r['hits']}/10 ({r['hit_rate']:.0%}) {r['avg_return']:>11.1%}")

    avg_auc = np.mean([r["auc"] for r in all_results])
    avg_hit = np.mean([r["hit_rate"] for r in all_results])
    avg_ret = np.mean([r["avg_return"] for r in all_results])
    total_hits = sum(r["hits"] for r in all_results)
    total_picks = len(all_results) * 10

    print("-" * 75)
    print(f"{'AVERAGE':<45} {avg_auc:>6.4f} {total_hits}/{total_picks} ({avg_hit:.0%}) {avg_ret:>11.1%}")
    print(f"\nOverall top-10 hit rate across all folds: {total_hits}/{total_picks} ({total_hits/total_picks:.1%})")


if __name__ == "__main__":
    run_walk_forward()
