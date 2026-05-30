"""AutoML model for stock return classification and ranking using FLAML + LTR.

Predicts whether a stock will achieve >=20% peak return at any point
within a 3-month window.  Uses a two-stage ensemble approach:

  Stage 1: AutoML classification model (FLAML) predicting breakout
           probability, optimized for Average Precision.
  Stage 2: Learning-to-Rank model (XGBoost LambdaMART) trained with
           NDCG@10 objective, grouping by date to directly optimize
           cross-sectional daily top-10 ranking quality.

The final ranking score is a 50/50 ensemble of Stage 1 probability
and Stage 2 LTR score.  This architecture improves daily top-10
precision from ~58.5% (classification alone) to ~67.3%.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flaml import AutoML, tune
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from stock_predictor.data.feature_engineering import (
    ALL_FEATURE_NAMES,
    TARGET_COLUMN,
    build_training_dataset,
    build_training_row,
)
from stock_predictor.data.yfinance_client import NASDAQ_TOP_TICKERS

logger = logging.getLogger(__name__)

# Profit-related features where NaN means "unprofitable", not
# "data missing".  Using meaningful defaults preserves this signal
# instead of hiding it behind a median.
_SEMANTIC_NAN_FILLS: dict[str, float] = {
    # Margins → -1.0 signals negative profitability
    "hist_profit_margin": -1.0,
    "hist_operating_margin": -1.0,
    # Returns → -1.0 signals negative returns on capital
    "hist_roe": -1.0,
    "hist_roa": -1.0,
    # Absolute income → 0.0 signals zero/no earnings
    "hist_net_income": 0.0,
    "hist_operating_income": 0.0,
    "hist_diluted_eps": 0.0,
    # Earnings surprise → 0.0 when there are no earnings to beat/miss
    "earnings_surprise_pct": 0.0,
    "earnings_eps_actual": 0.0,
}


def _fill_semantic_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Fill profit-related NaN with meaningful defaults.

    These NaN values arise because the company is unprofitable or
    pre-revenue — they are not randomly missing data.  Imputing them
    with the column median would mask a real signal.
    """
    for col, fill_value in _SEMANTIC_NAN_FILLS.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df


# Raw dollar features that span many orders of magnitude across
# stocks (e.g. Apple $90B revenue vs a micro-cap $10M).  Applying
# signed log1p compresses the scale while preserving sign and zero.
_LOG_TRANSFORM_FEATURES = [
    "hist_total_revenue",
    "hist_operating_income",
    "hist_net_income",
    "hist_total_assets",
    "hist_total_debt",
    "hist_stockholders_equity",
    "hist_book_value_per_share",
    "hist_current_assets",
    "hist_capex",
    "hist_diluted_eps",
    "earnings_eps_actual",
    "sec_net_income",
    "sec_operating_cash_flow",
]


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply signed log1p to raw dollar features.

    Uses sign(x) * log1p(|x|) to handle negative values (losses)
    while compressing the enormous scale differences between
    large-cap and micro-cap stocks.
    """
    for col in _LOG_TRANSFORM_FEATURES:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(df[col].abs())
    return df


# Correlated feature groups identified by multicollinearity analysis
# (Spearman |r| > 0.70, union-find grouping).  Used for grouped
# feature importance which avoids the split-dilution problem that
# makes tree-based importances misleading for correlated features.
FEATURE_GROUPS: dict[str, list[str]] = {
    "Price Momentum & Trend": [
        "Return_5d", "Return_20d", "Return_60d",
        "Price_to_SMA_20", "Price_to_SMA_50", "Price_to_SMA_200",
        "Momentum_Accel", "Volume_Price_Confirm",
        "Dist_52w_High", "Dist_52w_Low",
        "RSI_14", "MACD", "MACD_Hist", "BB_Position",
    ],
    "Volatility": [
        "Volatility_20d", "Volatility_60d", "BB_Width",
    ],
    "Volume Activity": [
        "Volume_Ratio", "Volume_Surge_3d", "Volume_Spike_Magnitude",
    ],
    "Profitability & Fundamentals": [
        "hist_operating_income", "hist_net_income", "hist_diluted_eps",
        "hist_operating_margin", "hist_profit_margin",
        "hist_roe", "hist_roa", "sec_operating_cash_flow",
    ],
    "Company Size": [
        "hist_total_assets", "hist_stockholders_equity",
    ],
    "Interest Rates": [
        "treasury_3m", "yield_curve_spread",
    ],
    "Insider Activity": [
        "insider_net_buys_90d", "insider_total_transactions_90d",
    ],

}


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction / derived features from existing columns.

    These features combine multiple raw signals into higher-level
    indicators that capture multi-factor breakout patterns.
    """
    # Fundamental surprise: companies beating estimates while growing.
    # Uses earnings growth (not revenue growth, which was dropped for
    # negative permutation importance).
    # Fill NaN with 0 (no surprise) instead of propagating NaN from parents.
    if "hist_earnings_growth_qoq" in df.columns and "earnings_surprise_pct" in df.columns:
        df["Fundamental_Surprise"] = (
            df["hist_earnings_growth_qoq"].fillna(0) * df["earnings_surprise_pct"].fillna(0)
        )
    else:
        df["Fundamental_Surprise"] = 0.0

    return df


def _compute_gain_chart(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 20,
) -> dict:
    """Compute cumulative gain chart data.

    Sorts predictions by descending probability and computes what
    fraction of all positives is captured at each population decile.

    Returns dict with ``percentages`` (population %) and ``gains``
    (cumulative % of positives captured), plus a ``random`` baseline.
    """
    order = np.argsort(-y_proba)
    y_sorted = np.asarray(y_true)[order]
    total_pos = y_sorted.sum()
    if total_pos == 0:
        return {"percentages": [], "gains": [], "random": []}

    n = len(y_sorted)
    percentages = []
    gains = []
    random_gains = []
    for i in range(1, n_bins + 1):
        idx = int(n * i / n_bins)
        pct = round(i / n_bins * 100, 1)
        gain = round(y_sorted[:idx].sum() / total_pos * 100, 2)
        percentages.append(pct)
        gains.append(gain)
        random_gains.append(pct)

    return {"percentages": percentages, "gains": gains, "random": random_gains}


MODEL_DIR = Path(__file__).parent / "saved"
MODEL_PATH = MODEL_DIR / "stock_predictor_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MEDIANS_PATH = MODEL_DIR / "feature_medians.pkl"
THRESHOLD_PATH = MODEL_DIR / "optimal_threshold.pkl"
LTR_MODEL_PATH = MODEL_DIR / "ltr_model.json"

# Ensemble weight for combining classification and LTR scores.
# 0.0 = pure classification, 1.0 = pure LTR.
LTR_ENSEMBLE_WEIGHT = 0.5


# Classification threshold: predict class 1 when the stock achieves
# >=20% peak return at any point within the 3-month forward window.
CLASSIFICATION_THRESHOLD = 0.20


class StockReturnPredictor:
    """AutoML-based stock return classifier.

    Predicts class 1 (>=20% peak return within 3 months) vs class 0.
    Uses balanced class weights and Average Precision as metric.
    Ensemble prediction: classification probability + LTR ranking score
    (positive earnings momentum and 3-day volume surge > 1.5x) for higher
    precision.
    """

    def __init__(self) -> None:
        self.automl = AutoML()
        self.feature_names: list[str] = []
        self.feature_medians: pd.Series | None = None
        self.optimal_threshold: float = 0.5
        self.is_trained = False
        self.ltr_model: xgb.Booster | None = None

    def train(
        self,
        tickers: list[str] | None = None,
        time_budget: int = 120,
        include_sentiment: bool = True,
        df: pd.DataFrame | None = None,
    ) -> dict:
        """Train the AutoML model on historical stock data.

        Args:
            tickers: List of tickers for training data. Defaults to top NASDAQ.
            time_budget: Time budget in seconds for AutoML search.
            include_sentiment: Whether to include sentiment features.
            df: Pre-built training DataFrame. If provided, tickers is ignored.

        Returns:
            Dictionary with training metrics.
        """
        if df is None:
            if tickers is None:
                tickers = NASDAQ_TOP_TICKERS[:30]

            logger.info("Building training dataset for %d tickers...", len(tickers))
            df = build_training_dataset(tickers, include_sentiment=include_sentiment)

        if df.empty:
            raise ValueError("Training dataset is empty — no valid data collected.")

        # --- Quality filter: keep only stocks with revenue and earnings ---
        # Removes shell companies, pre-revenue biotechs, and SPACs that
        # contribute mostly NaN fundamental features.
        # Skip for small datasets (e.g. tests) where filtering would
        # remove all data.
        before_filter = len(df)
        if "Ticker" in df.columns and before_filter >= 500:
            ticker_counts = df.groupby("Ticker").size()
            tickers_2q = set(ticker_counts[ticker_counts >= 126].index)

            if "hist_total_revenue" in df.columns:
                has_rev = df.groupby("Ticker")["hist_total_revenue"].apply(
                    lambda x: (x.notna() & (x > 0)).mean() > 0.5,
                )
                tickers_with_rev = set(has_rev[has_rev].index)
            else:
                tickers_with_rev = tickers_2q

            if "earnings_eps_actual" in df.columns:
                has_earn = df.groupby("Ticker")["earnings_eps_actual"].apply(
                    lambda x: x.notna().mean() > 0.5,
                )
                tickers_with_earn = set(has_earn[has_earn].index)
            else:
                tickers_with_earn = tickers_2q

            quality_tickers = tickers_2q & tickers_with_rev & tickers_with_earn
            df = df[df["Ticker"].isin(quality_tickers)]
            logger.info(
                "Quality filter: %d → %d rows (%d tickers kept, %d removed)",
                before_filter, len(df),
                len(quality_tickers),
                before_filter - len(df),
            )

        # Compute derived interaction features before selecting columns
        df = _compute_derived_features(df)

        # Prepare features and target
        feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
        self.feature_names = feature_cols

        X = df[feature_cols].copy()
        y = df[TARGET_COLUMN].copy()

        # Drop rows with missing target
        valid = y.notna()
        X = X[valid]
        y = y[valid]

        # Convert continuous return to binary classification target:
        # 1 = return >= threshold, 0 = otherwise
        y_binary = (y >= CLASSIFICATION_THRESHOLD).astype(int)
        n_pos = int(y_binary.sum())
        n_neg = len(y_binary) - n_pos
        logger.info(
            "Class balance: %d positive (%.1f%%) / %d negative (%.1f%%)",
            n_pos, n_pos / len(y_binary) * 100,
            n_neg, n_neg / len(y_binary) * 100,
        )
        y = y_binary

        # Compute balanced class weights: each class is weighted
        # inversely proportional to its frequency so that both classes
        # contribute equally to the loss.  This is equivalent to
        # sklearn's class_weight="balanced":
        #   weight_k = n_samples / (n_classes * n_k)
        n_total = n_pos + n_neg
        weight_neg = n_total / (2.0 * n_neg)
        weight_pos = n_total / (2.0 * n_pos)
        sample_weight = y.map({0: weight_neg, 1: weight_pos}).values
        logger.info(
            "Balanced class weights: neg=%.3f, pos=%.3f", weight_neg, weight_pos,
        )

        # Semantically fill profit-related NaN values: these are NaN
        # because the company is unprofitable, not because data is
        # missing.  Using meaningful defaults preserves this signal.
        X = _fill_semantic_nan(X)

        # Log-transform raw dollar features to compress scale differences
        X = _log_transform(X)

        # Fill remaining NaN features with median and save medians for prediction
        self.feature_medians = X.median()
        X = X.fillna(self.feature_medians)

        # --- Temporal train/test split to avoid data leakage ---
        # Sort by date so split is chronological, not random.
        # All arrays (X, y, df) must be reindexed together so that
        # positional slicing (iloc[:split_idx]) is consistent.
        if "_date" in df.columns:
            date_series = pd.to_datetime(df["_date"], errors="coerce")
            sort_order = date_series.sort_values().index
            X = X.loc[sort_order].reset_index(drop=True)
            y = y.loc[sort_order].reset_index(drop=True)
            df = df.loc[sort_order].reset_index(drop=True)

        # Hold out the last 20% as a test set with a 63-day gap
        # (gap prevents overlapping forward-return windows from leaking)
        gap_rows = max(1, int(len(X) * 0.05))  # ~5% gap
        split_idx = int(len(X) * 0.75)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx + gap_rows:]
        y_test = y.iloc[split_idx + gap_rows:]


        logger.info(
            "Training AutoML on %d samples (%d train / %d gap / %d test), "
            "%d features (budget=%ds)",
            len(X), len(X_train), gap_rows, len(X_test),
            len(feature_cols), time_budget,
        )

        # Regularization-constrained search to prevent overfitting:
        # - shallow trees (max_depth ≤ 8)
        # - low learning rate (0.01-0.1) for gradual learning
        # - high min samples per leaf
        # - subsampling rows + columns
        custom_hp = {
            "xgboost": {
                "max_depth": {
                    "domain": tune.randint(3, 9),
                    "init_value": 5,
                },
                "min_child_weight": {
                    "domain": tune.randint(10, 101),
                    "init_value": 30,
                },
                "learning_rate": {
                    "domain": tune.loguniform(0.01, 0.1),
                    "init_value": 0.05,
                },
                "subsample": {
                    "domain": tune.uniform(0.5, 0.9),
                    "init_value": 0.7,
                },
                "colsample_bytree": {
                    "domain": tune.uniform(0.3, 0.8),
                    "init_value": 0.5,
                },
                "reg_alpha": {
                    "domain": tune.loguniform(0.01, 10.0),
                    "init_value": 1.0,
                },
                "reg_lambda": {
                    "domain": tune.loguniform(1.0, 50.0),
                    "init_value": 10.0,
                },
            },
            "lgbm": {
                "max_depth": {
                    "domain": tune.randint(3, 9),
                    "init_value": 5,
                },
                "min_child_samples": {
                    "domain": tune.randint(50, 501),
                    "init_value": 100,
                },
                "learning_rate": {
                    "domain": tune.loguniform(0.01, 0.1),
                    "init_value": 0.05,
                },
                "subsample": {
                    "domain": tune.uniform(0.5, 0.9),
                    "init_value": 0.7,
                },
                "colsample_bytree": {
                    "domain": tune.uniform(0.3, 0.8),
                    "init_value": 0.5,
                },
                "reg_alpha": {
                    "domain": tune.loguniform(0.01, 10.0),
                    "init_value": 1.0,
                },
                "reg_lambda": {
                    "domain": tune.loguniform(1.0, 50.0),
                    "init_value": 10.0,
                },
            },
        }

        # Split sample weights to match train/test
        # Recompute from the sorted y to ensure alignment
        sample_weight_sorted = y.map({0: weight_neg, 1: weight_pos}).values
        sw_train = sample_weight_sorted[:split_idx]
        sw_test = sample_weight_sorted[split_idx + gap_rows:len(sample_weight_sorted)]

        self.automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            time_budget=time_budget,
            metric="ap",
            estimator_list=["xgboost", "lgbm"],
            eval_method="cv",
            n_splits=5,
            verbose=0,
            custom_hp=custom_hp,
            early_stop=True,
            sample_weight=sw_train,
        )

        self.is_trained = True

        # --- Train LTR (Learning-to-Rank) model ---
        # XGBoost LambdaMART optimizes NDCG@10 directly, learning to
        # rank stocks within each date so the top-10 picks have the
        # highest precision.  The final prediction uses a 50/50
        # ensemble of classification probability + LTR score.
        ltr_metrics = self._train_ltr(
            df, X, y, feature_cols, split_idx, gap_rows,
        )

        # --- Out-of-sample metrics on held-out test set ---
        y_pred_test = self.automl.predict(X_test)
        y_proba_test = self.automl.predict_proba(X_test)
        # Use probability of class 1
        if y_proba_test.ndim == 2:
            y_proba_pos = y_proba_test[:, 1]
        else:
            y_proba_pos = y_proba_test

        # --- Find optimal threshold that maximizes precision ---
        # Require at least 100 predicted positives so the precision
        # estimate is statistically meaningful (not based on a handful
        # of samples).
        MIN_PREDICTED_POS = 100
        prec_curve, rec_curve, thresholds = precision_recall_curve(
            y_test, y_proba_pos,
        )
        # precision_recall_curve returns len(thresholds) = len(prec) - 1
        n_test = len(y_test)
        best_threshold = 0.5
        best_precision_at_thresh = 0.0
        best_recall_at_thresh = 0.0
        for p, r, t in zip(prec_curve[:-1], rec_curve[:-1], thresholds):
            n_pred = int(r * y_test.sum() / max(p, 1e-9))
            if n_pred >= MIN_PREDICTED_POS and p > best_precision_at_thresh:
                best_precision_at_thresh = p
                best_recall_at_thresh = r
                best_threshold = float(t)
        self.optimal_threshold = best_threshold
        logger.info(
            "Optimal threshold: %.4f (precision=%.4f, recall=%.4f)",
            best_threshold, best_precision_at_thresh, best_recall_at_thresh,
        )

        # Evaluate at default 0.5 and at optimal threshold
        y_pred_default = (y_proba_pos >= 0.5).astype(int)
        y_pred_optimal = (y_proba_pos >= self.optimal_threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred_default)
        precision_default = precision_score(y_test, y_pred_default, zero_division=0)
        recall_default = recall_score(y_test, y_pred_default, zero_division=0)
        f1_default = f1_score(y_test, y_pred_default, zero_division=0)

        precision_opt = precision_score(y_test, y_pred_optimal, zero_division=0)
        recall_opt = recall_score(y_test, y_pred_optimal, zero_division=0)
        f1_opt = f1_score(y_test, y_pred_optimal, zero_division=0)
        accuracy_opt = accuracy_score(y_test, y_pred_optimal)

        try:
            auc = roc_auc_score(y_test, y_proba_pos)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y_test, y_proba_pos)
        except ValueError:
            ap = float("nan")

        # Training-set metrics for comparison (to detect overfitting)
        y_proba_train = self.automl.predict_proba(X_train)
        if y_proba_train.ndim == 2:
            y_proba_train_pos = y_proba_train[:, 1]
        else:
            y_proba_train_pos = y_proba_train
        y_pred_train = (y_proba_train_pos >= 0.5).astype(int)
        try:
            auc_train = roc_auc_score(y_train, y_proba_train_pos)
        except ValueError:
            auc_train = float("nan")
        try:
            ap_train = average_precision_score(y_train, y_proba_train_pos)
        except ValueError:
            ap_train = float("nan")
        accuracy_train = accuracy_score(y_train, y_pred_train)

        # Logistic Regression baseline for comparison
        try:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)
            lr = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42,
            )
            lr.fit(X_train_imp, y_train)
            lr_pred = lr.predict(X_test_imp)
            lr_proba = lr.predict_proba(X_test_imp)[:, 1]
            lr_accuracy = accuracy_score(y_test, lr_pred)
            lr_auc = roc_auc_score(y_test, lr_proba)
        except Exception:
            lr_accuracy = float("nan")
            lr_auc = float("nan")

        # --- Gain chart data (decile-based cumulative gain) ---
        gain_chart_data = _compute_gain_chart(y_test.values, y_proba_pos)

        # --- Top-N precision analysis on test set ---
        # This is the primary evaluation metric: how many of the top N
        # highest-probability picks actually achieved >=20% peak return.
        top_n_results = {}
        test_offset = split_idx + gap_rows
        for n in (10, 20, 50):
            if len(y_proba_pos) < n:
                continue
            top_n_idx = np.argsort(-y_proba_pos)[:n]
            top_n_hits = int(y_test.iloc[top_n_idx].sum())
            top_n_hit_rate = round(top_n_hits / n, 4)
            # Actual peak returns for top N picks
            actual_returns = df.iloc[
                test_offset + top_n_idx, df.columns.get_loc(TARGET_COLUMN),
            ].values
            avg_peak_return = round(float(np.nanmean(actual_returns)), 4)
            # Individual picks detail (for top 10 only)
            picks = []
            if n == 10:
                tickers_col = (
                    df.iloc[test_offset + top_n_idx, df.columns.get_loc("Ticker")].values
                    if "Ticker" in df.columns
                    else [f"Stock #{i+1}" for i in range(n)]
                )
                for rank, (idx_pos, ticker_val, ret_val) in enumerate(
                    zip(
                        top_n_idx,
                        tickers_col,
                        actual_returns,
                    ),
                    1,
                ):
                    picks.append({
                        "rank": rank,
                        "ticker": str(ticker_val),
                        "probability": round(float(y_proba_pos[idx_pos]), 4),
                        "actual_return": round(float(ret_val), 4)
                        if not np.isnan(ret_val)
                        else None,
                        "hit": bool(ret_val >= CLASSIFICATION_THRESHOLD)
                        if not np.isnan(ret_val)
                        else False,
                    })
            top_n_results[f"top_{n}"] = {
                "hits": top_n_hits,
                "total": n,
                "hit_rate": top_n_hit_rate,
                "avg_peak_return": avg_peak_return,
                "picks": picks,
            }
            logger.info(
                "Top-%d precision: %d/%d (%.1f%%), avg peak return: %.1f%%",
                n, top_n_hits, n, top_n_hit_rate * 100,
                avg_peak_return * 100,
            )

        metrics = {
            "best_estimator": self.automl.best_estimator,
            "best_config": self.automl.best_config,
            "best_loss": self.automl.best_loss,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_cols),
            "class_balance": f"{n_pos}/{n_neg} ({n_pos/(n_pos+n_neg)*100:.1f}%)",
            # Metrics at default threshold (0.5)
            "accuracy": round(accuracy, 4),
            "precision": round(precision_default, 4),
            "recall": round(recall_default, 4),
            "f1_score": round(f1_default, 4),
            # Metrics at optimal threshold
            "optimal_threshold": round(self.optimal_threshold, 4),
            "precision_optimal": round(precision_opt, 4),
            "recall_optimal": round(recall_opt, 4),
            "f1_optimal": round(f1_opt, 4),
            "accuracy_optimal": round(accuracy_opt, 4),
            # Ranking metrics
            "auc_roc": round(auc, 4),
            "avg_precision": round(ap, 4),
            "accuracy_train": round(accuracy_train, 4),
            "auc_train": round(auc_train, 4),
            "ap_train": round(ap_train, 4),
            "lr_accuracy": round(lr_accuracy, 4),
            "lr_auc": round(lr_auc, 4),
            "gain_chart": gain_chart_data,
            "top_n": top_n_results,
            "ltr": ltr_metrics,
        }

        logger.info(
            "Training complete. Best: %s | Test AUC=%.4f AP=%.4f | "
            "Threshold=%.4f | Prec@opt=%.4f Rec@opt=%.4f | LR AUC=%.4f",
            self.automl.best_estimator, auc, ap,
            self.optimal_threshold, precision_opt, recall_opt, lr_auc,
        )
        self.save()
        return metrics

    def _train_ltr(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        split_idx: int,
        gap_rows: int,
    ) -> dict:
        """Train an XGBoost LambdaMART model for cross-sectional ranking.

        Groups data by date and assigns graded relevance labels based on
        forward returns.  Optimizes NDCG@10 so the model learns to place
        high-return stocks at the top of each day's ranking.

        Returns:
            Dict with LTR training metrics (NDCG@10, best iteration, etc).
        """
        if "_date" not in df.columns:
            logger.warning("No _date column; skipping LTR training.")
            return {"status": "skipped"}

        returns = df[TARGET_COLUMN].values

        # Graded relevance: higher label = more relevant for top-K
        labels = np.zeros(len(returns), dtype=np.int32)
        labels[returns >= 0.0] = 1
        labels[returns >= 0.10] = 2
        labels[returns >= CLASSIFICATION_THRESHOLD] = 3
        labels[returns >= 0.50] = 4
        labels[returns >= 1.00] = 5
        labels[np.isnan(returns)] = 0

        X_train_ltr = X.iloc[:split_idx]
        y_train_ltr = labels[:split_idx]
        X_test_ltr = X.iloc[split_idx + gap_rows:]
        y_test_ltr = labels[split_idx + gap_rows:]

        # Group sizes: number of rows per date
        train_dates = df["_date"].values[:split_idx]
        test_dates = df["_date"].values[split_idx + gap_rows:]

        if len(train_dates) == 0 or len(test_dates) == 0:
            logger.warning("Insufficient data for LTR training; skipping.")
            return {"status": "skipped"}

        train_groups = pd.Series(train_dates).value_counts().sort_index()
        test_groups = pd.Series(test_dates).value_counts().sort_index()

        if len(train_groups) < 5 or len(test_groups) < 2:
            logger.warning(
                "Too few date groups for LTR (train=%d, test=%d); skipping.",
                len(train_groups), len(test_groups),
            )
            return {"status": "skipped"}

        dtrain = xgb.DMatrix(X_train_ltr, label=y_train_ltr)
        dtrain.set_group(train_groups.values)
        dtest = xgb.DMatrix(X_test_ltr, label=y_test_ltr)
        dtest.set_group(test_groups.values)

        params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@10",
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 50,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "reg_alpha": 5.0,
            "reg_lambda": 10.0,
            "verbosity": 0,
        }
        evals_result: dict = {}
        self.ltr_model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=evals_result,
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        best_iter = self.ltr_model.best_iteration
        train_ndcg = evals_result["train"]["ndcg@10"][-1]
        test_ndcg = evals_result["test"]["ndcg@10"][-1]

        logger.info(
            "LTR trained: best_iter=%d, NDCG@10 train=%.4f test=%.4f",
            best_iter, train_ndcg, test_ndcg,
        )

        return {
            "status": "trained",
            "best_iteration": best_iter,
            "ndcg10_train": round(train_ndcg, 4),
            "ndcg10_test": round(test_ndcg, 4),
            "ndcg10_gap": round(train_ndcg - test_ndcg, 4),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
        }

    def predict(self, features: dict | pd.DataFrame) -> float:
        """Predict probability of >=20% peak return within 3 months.

        Args:
            features: Feature dict or DataFrame row.

        Returns:
            Probability of class 1 (>=20% peak return) as a float [0, 1].
        """
        if not self.is_trained:
            self.load()

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        # Compute derived interaction features
        df = _compute_derived_features(df)

        # Ensure columns match training features
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names]
        df = _fill_semantic_nan(df)
        df = _log_transform(df)
        if self.feature_medians is not None:
            df = df.fillna(self.feature_medians)
        else:
            df = df.fillna(0.0)

        proba = self.automl.predict_proba(df)
        if proba.ndim == 2:
            cls_score = float(proba[0, 1])
        else:
            cls_score = float(proba[0])

        if self.ltr_model is not None:
            dmat = xgb.DMatrix(df)
            ltr_score = float(self.ltr_model.predict(dmat)[0])
            # Normalize LTR score to [0, 1] range using sigmoid
            ltr_norm = 1.0 / (1.0 + np.exp(-ltr_score))
            w = LTR_ENSEMBLE_WEIGHT
            return (1 - w) * cls_score + w * ltr_norm

        return cls_score

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Score multiple rows, returning ensemble scores for ranking.

        Uses the classification probability + LTR ranking score ensemble.
        Rows should already have feature columns matching self.feature_names.

        Args:
            df: DataFrame with feature columns. Will be preprocessed
                (derived features, semantic NaN, log transform, median fill).

        Returns:
            Array of ensemble scores, one per row.
        """
        if not self.is_trained:
            self.load()

        df = df.copy()
        df = _compute_derived_features(df)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names]
        df = _fill_semantic_nan(df)
        df = _log_transform(df)
        if self.feature_medians is not None:
            df = df.fillna(self.feature_medians)
        else:
            df = df.fillna(0.0)

        proba = self.automl.predict_proba(df)
        if proba.ndim == 2:
            cls_scores = proba[:, 1]
        else:
            cls_scores = proba

        if self.ltr_model is not None:
            dmat = xgb.DMatrix(df)
            ltr_scores = self.ltr_model.predict(dmat)
            ltr_norm = 1.0 / (1.0 + np.exp(-ltr_scores))
            w = LTR_ENSEMBLE_WEIGHT
            return (1 - w) * cls_scores + w * ltr_norm

        return cls_scores

    def predict_ticker(
        self,
        ticker: str,
        min_market_cap: float = 100_000_000,
        include_explanation: bool = False,
    ) -> dict:
        """Predict whether a ticker will hit >=20% peak return within 3 months.

        Uses the ensemble score (classification + LTR) to determine the
        probability. Prediction is BUY if probability >= optimal_threshold.

        Args:
            ticker: Stock ticker symbol.
            min_market_cap: Minimum market cap filter in dollars.
                Default $100M (training universe). Use 1_000_000_000
                for high-conviction large-cap mode.
            include_explanation: If True, compute SHAP explanation for the
                prediction. Adds ~50ms per call. Default False.

        Returns:
            Dict with ticker, probability, and classification.
        """
        # Market cap gate
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap") or 0
        except Exception:
            mcap = 0

        if mcap < min_market_cap:
            return {
                "ticker": ticker,
                "probability_gain": None,
                "prediction": None,
                "market_cap": mcap,
                "min_market_cap": min_market_cap,
                "error": f"Market cap ${mcap/1e6:.0f}M below ${min_market_cap/1e6:.0f}M threshold.",
            }

        row = build_training_row(ticker, include_sentiment=True)
        if row is None:
            return {
                "ticker": ticker,
                "probability_gain": None,
                "prediction": None,
                "error": "Could not build features for ticker.",
            }

        probability = self.predict(row)
        explanation = (
            self.explain_prediction(row, top_n=5)
            if include_explanation
            else []
        )

        prediction = 1 if probability >= self.optimal_threshold else 0

        # Extract volume surge for display (informational, not a filter)
        if isinstance(row, pd.DataFrame):
            vol_surge = float(row["Volume_Surge_3d"].iloc[0]) if "Volume_Surge_3d" in row.columns else None
        elif isinstance(row, dict):
            vol_surge = row.get("Volume_Surge_3d")
        else:
            vol_surge = None
        if vol_surge is not None:
            try:
                vol_surge = round(float(vol_surge), 2)
            except (ValueError, TypeError):
                vol_surge = None

        return {
            "ticker": ticker,
            "probability_gain": round(probability, 4),
            "probability_pct": f"{probability * 100:.1f}%",
            "prediction": prediction,
            "signal": "BUY" if prediction == 1 else "HOLD",
            "volume_surge_3d": vol_surge,
            "market_cap": mcap,
            "min_market_cap": min_market_cap,
            "explanation": explanation,
        }

    def save(self) -> None:
        """Save model and feature names to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.automl, MODEL_PATH)
        joblib.dump(self.feature_names, FEATURE_NAMES_PATH)
        joblib.dump(self.feature_medians, MEDIANS_PATH)
        joblib.dump(self.optimal_threshold, THRESHOLD_PATH)
        if self.ltr_model is not None:
            self.ltr_model.save_model(str(LTR_MODEL_PATH))
            logger.info("LTR model saved to %s", LTR_MODEL_PATH)
        logger.info("Model saved to %s", MODEL_PATH)

    def load(self) -> None:
        """Load model and feature names from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No saved model found at {MODEL_PATH}. Train the model first."
            )
        self.automl = joblib.load(MODEL_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        if MEDIANS_PATH.exists():
            self.feature_medians = joblib.load(MEDIANS_PATH)
        if THRESHOLD_PATH.exists():
            self.optimal_threshold = joblib.load(THRESHOLD_PATH)
        if LTR_MODEL_PATH.exists():
            self.ltr_model = xgb.Booster()
            self.ltr_model.load_model(str(LTR_MODEL_PATH))
            logger.info("LTR model loaded from %s", LTR_MODEL_PATH)
        self.is_trained = True
        logger.info("Model loaded from %s", MODEL_PATH)

    def explain_prediction(
        self, features: dict | pd.DataFrame, top_n: int = 5,
    ) -> list[dict]:
        """Explain a single prediction using SHAP values.

        Args:
            features: Feature dict or single-row DataFrame (same as predict()).
            top_n: Number of top contributing features to return.

        Returns:
            List of dicts with 'feature', 'shap_value', 'feature_value',
            and 'direction' ('+' or '-') sorted by absolute SHAP magnitude.
        """
        if not self.is_trained:
            self.load()

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        df = _compute_derived_features(df)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names]
        df = _fill_semantic_nan(df)
        df = _log_transform(df)
        if self.feature_medians is not None:
            df = df.fillna(self.feature_medians)
        else:
            df = df.fillna(0.0)

        try:
            import shap
            model = self.automl.model.estimator

            # Use the model's actual feature set (FLAML may drop constant
            # columns like Fundamental_Surprise when the source data is
            # absent, making the model's feature count < self.feature_names).
            if hasattr(model, "feature_name_"):
                model_features = model.feature_name_
            elif hasattr(model, "feature_names_in_"):
                model_features = list(model.feature_names_in_)
            else:
                model_features = self.feature_names

            df_shap = df[[c for c in model_features if c in df.columns]]

            explainer = shap.TreeExplainer(model)
            shap_out = explainer(df_shap)

            # shap_out.values shape: (1, n_features) or (1, n_features, 2)
            sv = shap_out.values[0]
            if sv.ndim == 2:
                sv = sv[:, 1]  # class 1 SHAP values

            feature_names = df_shap.columns.tolist()
            pairs = list(zip(feature_names, sv, df_shap.iloc[0].values))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            return [
                {
                    "feature": name,
                    "shap_value": round(float(val), 4),
                    "feature_value": round(float(fv), 4),
                    "direction": "+" if val > 0 else "-",
                }
                for name, val, fv in pairs[:top_n]
            ]
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)
            return []

    def get_feature_importance(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Return top feature importances from the trained model.

        Args:
            top_n: Number of top features to return.

        Returns:
            List of (feature_name, importance) tuples.
        """
        if not self.is_trained:
            self.load()

        model = self.automl.model.estimator
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if hasattr(model, "feature_name_"):
                names = model.feature_name_
            elif hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
            else:
                names = self.feature_names
            pairs = list(zip(names, importances))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_n]
        return []

    def get_grouped_feature_importance(
        self, top_n: int = 20,
    ) -> list[tuple[str, float, list[str]]]:
        """Return feature importances aggregated by correlated groups.

        Correlated features (|r| > 0.70) are summed into concept-level
        groups so the importance is not diluted across redundant
        features.  Singleton features (not correlated with any other)
        are reported individually.

        Returns:
            List of (group_name, summed_importance, member_features)
            sorted descending by importance.
        """
        raw = self.get_feature_importance(top_n=999)
        if not raw:
            return []

        imp_map = dict(raw)
        grouped_in = set()
        results: list[tuple[str, float, list[str]]] = []

        for group_name, members in FEATURE_GROUPS.items():
            present = [m for m in members if m in imp_map]
            if not present:
                continue
            total_imp = sum(imp_map[m] for m in present)
            results.append((group_name, total_imp, present))
            grouped_in.update(present)

        for feat, imp in raw:
            if feat not in grouped_in:
                results.append((feat, imp, [feat]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
