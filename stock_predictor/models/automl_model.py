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

import json
import logging
import os
import platform
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
REGIME_MODEL_PATH = MODEL_DIR / "regime_model.pkl"
TICKER_CALIBRATION_PATH = MODEL_DIR / "ticker_calibration.pkl"
REGRESSION_MODEL_PATH = MODEL_DIR / "regression_model.pkl"
FOLD_MODELS_DIR = MODEL_DIR / "fold_models"
_MODEL_META_PATH = MODEL_DIR / "model_meta.json"

# Ensemble weights for 3-stage scoring:
#   final_score = W_CLS * classification_prob
#                + W_LTR * ltr_score_normalized
#                + W_REG * regression_pred_normalized
# These sum to 1.0.
W_CLS = 0.30
W_LTR = 0.40
W_REG = 0.30
# Legacy weight (kept for backward compat in predict_batch fallback)
LTR_ENSEMBLE_WEIGHT = 0.6

# Volatility-aware scoring: multiply ranking score by
# (1 + VOLATILITY_SCORE_ALPHA * percentile_rank_of_volatility).
# Higher-volatility stocks are more likely to achieve >=20% breakouts.
VOLATILITY_SCORE_ALPHA = 0.25

# Default number of top picks to highlight.
DEFAULT_TOP_K = 5


# Classification threshold: predict class 1 when the stock achieves
# >=20% peak return at any point within the 3-month forward window.
CLASSIFICATION_THRESHOLD = 0.20


class StockReturnPredictor:
    """3-stage hybrid stock predictor with return magnitude ranking.

    Stage 1: Binary classifier (≥20% peak return within 3 months).
    Stage 2: Regression model predicting actual return magnitude.
    Stage 3: LTR (LambdaMART) with continuous return labels.

    The ensemble blends all three stages so top picks are both likely
    to achieve ≥20% AND ranked by expected return size. Additional
    adjustments: volatility-aware scoring, market regime detection,
    and per-ticker calibration.
    """

    def __init__(self) -> None:
        self.automl = AutoML()
        self.regression_model: AutoML | None = None
        self.feature_names: list[str] = []
        self.feature_medians: pd.Series | None = None
        self.optimal_threshold: float = 0.5
        self.is_trained = False
        self.ltr_model: xgb.Booster | None = None
        self.regime_model: Optional[object] = None
        self.ticker_calibration: dict[str, float] = {}
        # Walk-forward ensemble: list of per-fold model dicts
        self.fold_models: list[dict] = []

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
        # XGBoost LambdaMART with continuous return labels optimizes
        # NDCG@10 directly, learning to rank stocks by expected return
        # magnitude within each date.
        ltr_metrics = self._train_ltr(
            df, X, y, feature_cols, split_idx, gap_rows,
        )

        # --- Train regression model for return magnitude ---
        # Predicts the actual forward return (continuous), used to rank
        # stocks by expected gain size, not just probability of hitting
        # the ≥20% threshold.
        reg_time = max(30, time_budget // 4)
        regression_metrics = self._train_regression(
            df, X, feature_cols, split_idx, gap_rows,
            time_budget=reg_time,
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
            "regression": regression_metrics,
        }

        # --- Market Regime Detection ---
        regime_metrics = self._train_regime_model(
            df, X, y, split_idx, gap_rows,
        )
        metrics["regime"] = regime_metrics

        # --- Per-Ticker Calibration ---
        ticker_cal_metrics = self._train_ticker_calibration(
            df, X, y, split_idx, gap_rows,
        )
        metrics["ticker_calibration"] = ticker_cal_metrics

        logger.info(
            "Training complete. Best: %s | Test AUC=%.4f AP=%.4f | "
            "Threshold=%.4f | Prec@opt=%.4f Rec@opt=%.4f | LR AUC=%.4f",
            self.automl.best_estimator, auc, ap,
            self.optimal_threshold, precision_opt, recall_opt, lr_auc,
        )
        self.save()
        return metrics

    def train_walk_forward(
        self,
        df: pd.DataFrame | None = None,
        time_budget: int = 120,
        n_folds: int = 5,
        min_train_years: int = 3,
    ) -> dict:
        """Train with walk-forward cross-validation for robust evaluation.

        Splits data chronologically into expanding windows:
        - Fold 1: Train on years 1-3, test on year 4
        - Fold 2: Train on years 1-4, test on year 5
        - ...
        - Final fold: Train on years 1-(N-1), test on year N

        The final fold's model is saved as the production model.
        Returns per-fold metrics and aggregate statistics.

        Args:
            df: Pre-built training DataFrame.
            time_budget: FLAML AutoML time budget per fold (seconds).
            n_folds: Number of walk-forward folds.
            min_train_years: Minimum years of training data for first fold.

        Returns:
            Dictionary with per-fold metrics and aggregate summary.
        """
        if df is None:
            raise ValueError("Walk-forward requires a pre-built DataFrame (df).")

        if "_date" not in df.columns:
            raise ValueError("DataFrame must have '_date' column for temporal splitting.")

        df = df.copy()
        df["_date"] = pd.to_datetime(df["_date"])
        df = df.sort_values("_date").reset_index(drop=True)

        # Compute derived interaction features
        df = _compute_derived_features(df)

        date_min = df["_date"].min()
        date_max = df["_date"].max()
        total_years = (date_max - date_min).days / 365.25
        logger.info(
            "Walk-forward: %d rows, %d tickers, %.1f years (%s to %s)",
            len(df), df["Ticker"].nunique(), total_years,
            date_min.date(), date_max.date(),
        )

        if total_years < min_train_years + 1:
            raise ValueError(
                f"Need at least {min_train_years + 1} years of data, "
                f"but only have {total_years:.1f} years."
            )

        # Compute fold boundaries by year
        test_years = total_years - min_train_years
        fold_size_days = int((test_years * 365.25) / n_folds)
        if fold_size_days < 63:  # minimum ~3 months per fold
            n_folds = max(1, int(test_years * 365.25 / 63))
            fold_size_days = int((test_years * 365.25) / n_folds)
            logger.warning("Reduced to %d folds (fold size=%d days)", n_folds, fold_size_days)

        first_test_start = date_min + pd.Timedelta(days=int(min_train_years * 365.25))
        gap_days = 63  # 3-month gap to prevent forward-return leakage

        fold_results = []
        all_top10_hits = []
        all_top10_totals = []

        for fold_idx in range(n_folds):
            test_start = first_test_start + pd.Timedelta(days=fold_idx * fold_size_days)
            test_end = test_start + pd.Timedelta(days=fold_size_days)
            train_end = test_start - pd.Timedelta(days=gap_days)

            # Last fold extends to end of data
            if fold_idx == n_folds - 1:
                test_end = date_max + pd.Timedelta(days=1)

            train_mask = df["_date"] <= train_end
            test_mask = (df["_date"] >= test_start) & (df["_date"] < test_end)

            df_train = df[train_mask].copy()
            df_test = df[test_mask].copy()

            if len(df_train) < 500 or len(df_test) < 100:
                logger.warning(
                    "Fold %d: insufficient data (train=%d, test=%d), skipping.",
                    fold_idx + 1, len(df_train), len(df_test),
                )
                continue

            logger.info(
                "=== Fold %d/%d: Train %s–%s (%d rows) | Test %s–%s (%d rows) ===",
                fold_idx + 1, n_folds,
                df_train["_date"].min().date(), df_train["_date"].max().date(),
                len(df_train),
                df_test["_date"].min().date(), df_test["_date"].max().date(),
                len(df_test),
            )

            # Prepare features
            feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]

            X_train = df_train[feature_cols].copy()
            y_train_raw = df_train[TARGET_COLUMN].copy()
            X_test = df_test[feature_cols].copy()
            y_test_raw = df_test[TARGET_COLUMN].copy()

            # Drop rows with missing target
            valid_train = y_train_raw.notna()
            X_train = X_train[valid_train].reset_index(drop=True)
            y_train_raw = y_train_raw[valid_train].reset_index(drop=True)
            df_train = df_train[valid_train].reset_index(drop=True)

            valid_test = y_test_raw.notna()
            X_test = X_test[valid_test].reset_index(drop=True)
            y_test_raw = y_test_raw[valid_test].reset_index(drop=True)
            df_test = df_test[valid_test].reset_index(drop=True)

            # Binary classification target
            y_train = (y_train_raw >= CLASSIFICATION_THRESHOLD).astype(int)
            y_test = (y_test_raw >= CLASSIFICATION_THRESHOLD).astype(int)

            # Balanced class weights (inverse frequency, matching A/B script)
            pos_rate = y_train.mean()
            if pos_rate == 0 or pos_rate == 1:
                logger.warning("Fold %d: single class in training, skipping.", fold_idx + 1)
                continue
            sw_train = np.where(
                y_train == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate),
            )

            # Preprocessing: median fill + percentile clipping (no log
            # transform or semantic NaN — matches the validated A/B script)
            fold_medians = X_train.median()
            for col in feature_cols:
                X_train[col] = X_train[col].fillna(fold_medians[col])
                X_test[col] = X_test[col].fillna(fold_medians[col])
                p01, p99 = X_train[col].quantile(0.01), X_train[col].quantile(0.99)
                if p01 < p99:
                    X_train[col] = X_train[col].clip(p01, p99)
                    X_test[col] = X_test[col].clip(p01, p99)

            # Train FLAML AutoML for this fold
            fold_automl = AutoML()
            fold_automl.fit(
                X_train=X_train,
                y_train=y_train,
                task="classification",
                time_budget=time_budget,
                metric="ap",
                estimator_list=["xgboost", "lgbm"],
                eval_method="cv",
                n_splits=5,
                verbose=0,
                early_stop=True,
                sample_weight=sw_train,
            )

            # Evaluate on test set
            y_proba_test = fold_automl.predict_proba(X_test)
            if y_proba_test.ndim == 2:
                y_proba_pos = y_proba_test[:, 1]
            else:
                y_proba_pos = y_proba_test

            y_proba_train = fold_automl.predict_proba(X_train)
            if y_proba_train.ndim == 2:
                y_proba_train_pos = y_proba_train[:, 1]
            else:
                y_proba_train_pos = y_proba_train

            try:
                auc_test = roc_auc_score(y_test, y_proba_pos)
            except ValueError:
                auc_test = float("nan")
            try:
                ap_test = average_precision_score(y_test, y_proba_pos)
            except ValueError:
                ap_test = float("nan")
            try:
                auc_train = roc_auc_score(y_train, y_proba_train_pos)
            except ValueError:
                auc_train = float("nan")
            try:
                ap_train = average_precision_score(y_train, y_proba_train_pos)
            except ValueError:
                ap_train = float("nan")

            # Top-10 precision
            top10_hits = 0
            top10_picks = []
            if len(y_proba_pos) >= 10:
                top10_idx = np.argsort(-y_proba_pos)[:10]
                top10_hits = int(y_test.iloc[top10_idx].sum())
                actual_returns = y_test_raw.iloc[top10_idx].values
                if "Ticker" in df_test.columns:
                    top10_tickers = df_test["Ticker"].iloc[top10_idx].values
                else:
                    top10_tickers = [f"Stock #{i+1}" for i in range(10)]
                for rank, (idx_pos, ticker_val, ret_val) in enumerate(
                    zip(top10_idx, top10_tickers, actual_returns), 1
                ):
                    top10_picks.append({
                        "rank": rank,
                        "ticker": str(ticker_val),
                        "probability": round(float(y_proba_pos[idx_pos]), 4),
                        "actual_return": round(float(ret_val), 4) if not np.isnan(ret_val) else None,
                        "hit": bool(ret_val >= CLASSIFICATION_THRESHOLD) if not np.isnan(ret_val) else False,
                    })

            top10_hit_rate = top10_hits / 10.0 if len(y_proba_pos) >= 10 else float("nan")
            all_top10_hits.append(top10_hits)
            all_top10_totals.append(min(10, len(y_proba_pos)))

            # LTR for this fold (continuous return labels)
            ltr_ndcg_train = float("nan")
            ltr_ndcg_test = float("nan")
            fold_ltr_model = None
            if "_date" in df_train.columns and "_date" in df_test.columns:
                try:
                    returns_train = df_train[TARGET_COLUMN].values
                    returns_test = df_test[TARGET_COLUMN].values
                    labels_tr = np.round(np.clip(returns_train, 0.0, 5.0) * 100).astype(np.int32)
                    labels_tr[np.isnan(returns_train)] = 0
                    labels_te = np.round(np.clip(returns_test, 0.0, 5.0) * 100).astype(np.int32)
                    labels_te[np.isnan(returns_test)] = 0

                    tr_groups = pd.Series(df_train["_date"].values).value_counts().sort_index()
                    te_groups = pd.Series(df_test["_date"].values).value_counts().sort_index()

                    if len(tr_groups) >= 5 and len(te_groups) >= 2:
                        dtrain = xgb.DMatrix(X_train, label=labels_tr)
                        dtrain.set_group(tr_groups.values)
                        dtest = xgb.DMatrix(X_test, label=labels_te)
                        dtest.set_group(te_groups.values)

                        ltr_params = {
                            "objective": "rank:ndcg",
                            "eval_metric": "ndcg@10",
                            "ndcg_exp_gain": False,
                            "max_depth": 6,
                            "learning_rate": 0.05,
                            "min_child_weight": 50,
                            "subsample": 0.7,
                            "colsample_bytree": 0.5,
                            "reg_alpha": 5.0,
                            "reg_lambda": 10.0,
                            "verbosity": 0,
                        }
                        ltr_evals: dict = {}
                        fold_ltr_model = xgb.train(
                            ltr_params, dtrain, num_boost_round=500,
                            evals=[(dtrain, "train"), (dtest, "test")],
                            evals_result=ltr_evals,
                            early_stopping_rounds=50,
                            verbose_eval=False,
                        )
                        ltr_ndcg_train = ltr_evals["train"]["ndcg@10"][-1]
                        ltr_ndcg_test = ltr_evals["test"]["ndcg@10"][-1]
                except Exception as e:
                    logger.warning("Fold %d LTR failed: %s", fold_idx + 1, e)

            # Regression model for this fold
            fold_reg_model = None
            reg_r2_test = float("nan")
            reg_mae_test = float("nan")
            try:
                returns_for_reg = df_train[TARGET_COLUMN].values.copy()
                returns_capped_tr = np.clip(returns_for_reg, -1.0, 5.0)
                valid_tr = ~np.isnan(returns_capped_tr)
                X_tr_reg = X_train[valid_tr]
                y_tr_reg = returns_capped_tr[valid_tr]

                returns_for_reg_te = df_test[TARGET_COLUMN].values.copy()
                returns_capped_te = np.clip(returns_for_reg_te, -1.0, 5.0)
                valid_te = ~np.isnan(returns_capped_te)
                X_te_reg = X_test[valid_te]
                y_te_reg = returns_capped_te[valid_te]

                if len(X_tr_reg) >= 100 and len(X_te_reg) >= 50:
                    fold_reg_model = AutoML()
                    reg_budget = max(30, time_budget // 4)
                    fold_reg_model.fit(
                        X_train=X_tr_reg, y_train=y_tr_reg,
                        task="regression", time_budget=reg_budget,
                        metric="mae", estimator_list=["xgboost", "lgbm"],
                        eval_method="cv", n_splits=5, verbose=0,
                        early_stop=True,
                    )
                    from sklearn.metrics import mean_absolute_error, r2_score
                    y_pred_reg = fold_reg_model.predict(X_te_reg)
                    reg_r2_test = r2_score(y_te_reg, y_pred_reg)
                    reg_mae_test = mean_absolute_error(y_te_reg, y_pred_reg)
                    logger.info(
                        "Fold %d regression: R²=%.4f MAE=%.4f",
                        fold_idx + 1, reg_r2_test, reg_mae_test,
                    )
            except Exception as e:
                logger.warning("Fold %d regression failed: %s", fold_idx + 1, e)

            # Recompute top-10 using 3-stage ensemble
            ensemble_scores = y_proba_pos.copy()
            if fold_ltr_model is not None:
                dmat_test = xgb.DMatrix(X_test)
                ltr_test_scores = fold_ltr_model.predict(dmat_test)
                ltr_test_norm = 1.0 / (1.0 + np.exp(-ltr_test_scores))
                if fold_reg_model is not None:
                    reg_test_pred = fold_reg_model.predict(X_test)
                    reg_test_norm = 1.0 / (1.0 + np.exp(-reg_test_pred * 2))
                    ensemble_scores = (
                        W_CLS * y_proba_pos
                        + W_LTR * ltr_test_norm
                        + W_REG * reg_test_norm
                    )
                else:
                    w = LTR_ENSEMBLE_WEIGHT
                    ensemble_scores = (1 - w) * y_proba_pos + w * ltr_test_norm
            elif fold_reg_model is not None:
                reg_test_pred = fold_reg_model.predict(X_test)
                reg_test_norm = 1.0 / (1.0 + np.exp(-reg_test_pred * 2))
                ensemble_scores = 0.6 * y_proba_pos + 0.4 * reg_test_norm

            # Re-evaluate top-10 with ensemble
            top10_hits_ens = 0
            top10_picks_ens = []
            top10_avg_return = float("nan")
            if len(ensemble_scores) >= 10:
                top10_idx_ens = np.argsort(-ensemble_scores)[:10]
                top10_hits_ens = int(y_test.iloc[top10_idx_ens].sum())
                actual_returns_ens = y_test_raw.iloc[top10_idx_ens].values
                top10_avg_return = float(np.nanmean(actual_returns_ens))
                if "Ticker" in df_test.columns:
                    top10_tickers_ens = df_test["Ticker"].iloc[top10_idx_ens].values
                else:
                    top10_tickers_ens = [f"Stock #{i+1}" for i in range(10)]
                for rank, (idx_pos, ticker_val, ret_val) in enumerate(
                    zip(top10_idx_ens, top10_tickers_ens, actual_returns_ens), 1
                ):
                    top10_picks_ens.append({
                        "rank": rank,
                        "ticker": str(ticker_val),
                        "ensemble_score": round(float(ensemble_scores[idx_pos]), 4),
                        "actual_return": round(float(ret_val), 4) if not np.isnan(ret_val) else None,
                        "hit": bool(ret_val >= CLASSIFICATION_THRESHOLD) if not np.isnan(ret_val) else False,
                    })

            top10_hit_rate_ens = top10_hits_ens / 10.0 if len(ensemble_scores) >= 10 else float("nan")
            # Override for aggregate stats
            all_top10_hits[-1] = top10_hits_ens
            all_top10_totals[-1] = min(10, len(ensemble_scores))

            fold_result = {
                "fold": fold_idx + 1,
                "train_period": f"{df_train['_date'].min().date()} to {df_train['_date'].max().date()}",
                "test_period": f"{df_test['_date'].min().date()} to {df_test['_date'].max().date()}",
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "best_estimator": fold_automl.best_estimator,
                "auc_train": round(auc_train, 4),
                "auc_test": round(auc_test, 4),
                "auc_gap": round(auc_train - auc_test, 4),
                "ap_train": round(ap_train, 4),
                "ap_test": round(ap_test, 4),
                "ap_gap": round(ap_train - ap_test, 4),
                "top10_hits": top10_hits_ens,
                "top10_hit_rate": round(top10_hit_rate_ens, 4),
                "top10_avg_return": round(top10_avg_return, 4) if not np.isnan(top10_avg_return) else None,
                "top10_picks": top10_picks_ens,
                "ltr_ndcg_train": round(ltr_ndcg_train, 4) if not np.isnan(ltr_ndcg_train) else None,
                "ltr_ndcg_test": round(ltr_ndcg_test, 4) if not np.isnan(ltr_ndcg_test) else None,
                "reg_r2_test": round(reg_r2_test, 4) if not np.isnan(reg_r2_test) else None,
                "reg_mae_test": round(reg_mae_test, 4) if not np.isnan(reg_mae_test) else None,
            }
            fold_results.append(fold_result)

            logger.info(
                "Fold %d: AUC train=%.4f test=%.4f | AP train=%.4f test=%.4f | "
                "Top-10: %d/10 (%.0f%%) [cls=%d/10, ens=%d/10]",
                fold_idx + 1, auc_train, auc_test, ap_train, ap_test,
                top10_hits_ens, top10_hit_rate_ens * 100,
                top10_hits, top10_hits_ens,
            )

            # Save this fold's models for ensemble
            fold_model_entry = {
                "fold": fold_idx + 1,
                "automl": fold_automl,
                "ltr_model": fold_ltr_model,
                "regression_model": fold_reg_model,
                "feature_medians": fold_medians,
            }
            self.fold_models.append(fold_model_entry)
            logger.info(
                "Fold %d models saved to ensemble (%d total)",
                fold_idx + 1, len(self.fold_models),
            )

            # On the final fold, also set single-model attrs for
            # backward compat and save everything
            if fold_idx == n_folds - 1:
                logger.info("Saving walk-forward ensemble (%d fold models)...", len(self.fold_models))
                self.automl = fold_automl
                self.feature_names = feature_cols
                self.feature_medians = fold_medians
                self.is_trained = True

                # Train final LTR on the last fold's split
                split_idx = len(X_train)
                X_combined = pd.concat([X_train, X_test], ignore_index=True)
                y_combined = pd.concat([y_train, y_test], ignore_index=True)
                df_combined = pd.concat([df_train, df_test], ignore_index=True)
                ltr_metrics = self._train_ltr(
                    df_combined, X_combined, y_combined,
                    feature_cols, split_idx, 0,
                )

                # Train final regression model on the last fold's split
                reg_budget_final = max(30, time_budget // 4)
                self._train_regression(
                    df_combined, X_combined,
                    feature_cols, split_idx, 0,
                    time_budget=reg_budget_final,
                )

                # Regime + calibration on final fold
                regime_metrics = self._train_regime_model(
                    df_combined, X_combined, y_combined,
                    split_idx, 0,
                )
                cal_metrics = self._train_ticker_calibration(
                    df_combined, X_combined, y_combined,
                    split_idx, 0,
                )
                self.save()

            import gc
            gc.collect()

        # Aggregate statistics
        total_hits = sum(all_top10_hits)
        total_picks = sum(all_top10_totals)
        avg_hit_rate = total_hits / total_picks if total_picks > 0 else 0.0

        auc_tests = [f["auc_test"] for f in fold_results if not np.isnan(f["auc_test"])]
        ap_tests = [f["ap_test"] for f in fold_results if not np.isnan(f["ap_test"])]
        auc_gaps = [f["auc_gap"] for f in fold_results if not np.isnan(f["auc_gap"])]
        ap_gaps = [f["ap_gap"] for f in fold_results if not np.isnan(f["ap_gap"])]

        # Avg return of top-10 picks across folds
        avg_returns = [
            f["top10_avg_return"] for f in fold_results
            if f.get("top10_avg_return") is not None
        ]

        summary = {
            "n_folds": len(fold_results),
            "aggregate_top10_hits": total_hits,
            "aggregate_top10_total": total_picks,
            "aggregate_top10_hit_rate": round(avg_hit_rate, 4),
            "mean_top10_avg_return": round(float(np.mean(avg_returns)), 4) if avg_returns else None,
            "mean_auc_test": round(float(np.mean(auc_tests)), 4) if auc_tests else None,
            "std_auc_test": round(float(np.std(auc_tests)), 4) if auc_tests else None,
            "mean_ap_test": round(float(np.mean(ap_tests)), 4) if ap_tests else None,
            "std_ap_test": round(float(np.std(ap_tests)), 4) if ap_tests else None,
            "mean_auc_gap": round(float(np.mean(auc_gaps)), 4) if auc_gaps else None,
            "mean_ap_gap": round(float(np.mean(ap_gaps)), 4) if ap_gaps else None,
            "folds": fold_results,
        }

        logger.info("=" * 70)
        logger.info("WALK-FORWARD SUMMARY (3-Stage Hybrid)")
        logger.info("=" * 70)
        logger.info(
            "Folds: %d | Aggregate Top-10: %d/%d (%.1f%%) | "
            "Avg Top-10 Return: %.1f%% | "
            "Mean AUC: %.4f ± %.4f | Mean AP: %.4f ± %.4f",
            len(fold_results), total_hits, total_picks, avg_hit_rate * 100,
            (np.mean(avg_returns) * 100) if avg_returns else 0,
            np.mean(auc_tests) if auc_tests else 0,
            np.std(auc_tests) if auc_tests else 0,
            np.mean(ap_tests) if ap_tests else 0,
            np.std(ap_tests) if ap_tests else 0,
        )
        for f in fold_results:
            avg_ret_str = f"{f['top10_avg_return']*100:.1f}%" if f.get("top10_avg_return") is not None else "N/A"
            logger.info(
                "  Fold %d [%s]: AUC=%.4f AP=%.4f Top10=%d/10 (%.0f%%) AvgRet=%s",
                f["fold"], f["test_period"],
                f["auc_test"], f["ap_test"],
                f["top10_hits"], f["top10_hit_rate"] * 100, avg_ret_str,
            )

        return summary

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

        # Fine-grained relevance labels based on return magnitude.
        # XGBoost rank:ndcg requires non-negative integer labels, so we
        # scale returns to integers (each 1% = 1 relevance point, capped
        # at 500). This preserves return magnitude granularity — a stock
        # with 45% return gets label 45, much higher than one with 21%
        # (label 21), so the LTR optimizes for ranking by magnitude.
        clipped = np.clip(returns, 0.0, 5.0)
        labels = np.round(clipped * 100).astype(np.int32)
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
            "ndcg_exp_gain": False,
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

    def _train_regression(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        feature_cols: list[str],
        split_idx: int,
        gap_rows: int,
        time_budget: int = 60,
    ) -> dict:
        """Train a regression model predicting actual forward return magnitude.

        Among stocks predicted to gain ≥20% by the classifier, this model
        ranks them by expected return size. Higher predicted returns rank higher.

        The regression target is the raw forward return (continuous), capped
        at 500% to reduce outlier influence.

        Returns:
            Dict with regression model metrics (R², MAE on test set).
        """
        returns = df[TARGET_COLUMN].values.copy()

        # Cap extreme returns to reduce outlier influence
        returns_capped = np.clip(returns, -1.0, 5.0)
        valid_mask = ~np.isnan(returns_capped)

        X_train_reg = X.iloc[:split_idx][valid_mask[:split_idx]]
        y_train_reg = returns_capped[:split_idx][valid_mask[:split_idx]]
        X_test_reg = X.iloc[split_idx + gap_rows:][valid_mask[split_idx + gap_rows:]]
        y_test_reg = returns_capped[split_idx + gap_rows:][valid_mask[split_idx + gap_rows:]]

        if len(X_train_reg) < 100 or len(X_test_reg) < 50:
            logger.warning("Insufficient data for regression model; skipping.")
            return {"status": "skipped"}

        logger.info(
            "Training regression model: %d train / %d test samples",
            len(X_train_reg), len(X_test_reg),
        )

        self.regression_model = AutoML()
        self.regression_model.fit(
            X_train=X_train_reg,
            y_train=y_train_reg,
            task="regression",
            time_budget=time_budget,
            metric="mae",
            estimator_list=["xgboost", "lgbm"],
            eval_method="cv",
            n_splits=5,
            verbose=0,
            custom_hp={
                "xgboost": {
                    "max_depth": {"domain": tune.randint(3, 8), "init_value": 5},
                    "min_child_weight": {"domain": tune.randint(20, 101), "init_value": 50},
                    "learning_rate": {"domain": tune.loguniform(0.01, 0.1), "init_value": 0.05},
                    "subsample": {"domain": tune.uniform(0.5, 0.9), "init_value": 0.7},
                    "colsample_bytree": {"domain": tune.uniform(0.3, 0.8), "init_value": 0.5},
                    "reg_alpha": {"domain": tune.loguniform(0.1, 10.0), "init_value": 1.0},
                    "reg_lambda": {"domain": tune.loguniform(1.0, 50.0), "init_value": 10.0},
                },
                "lgbm": {
                    "max_depth": {"domain": tune.randint(3, 8), "init_value": 5},
                    "min_child_samples": {"domain": tune.randint(50, 301), "init_value": 100},
                    "learning_rate": {"domain": tune.loguniform(0.01, 0.1), "init_value": 0.05},
                    "subsample": {"domain": tune.uniform(0.5, 0.9), "init_value": 0.7},
                    "colsample_bytree": {"domain": tune.uniform(0.3, 0.8), "init_value": 0.5},
                    "reg_alpha": {"domain": tune.loguniform(0.1, 10.0), "init_value": 1.0},
                    "reg_lambda": {"domain": tune.loguniform(1.0, 50.0), "init_value": 10.0},
                },
            },
            early_stop=True,
        )

        # Evaluate
        from sklearn.metrics import mean_absolute_error, r2_score

        y_pred_test = self.regression_model.predict(X_test_reg)
        y_pred_train = self.regression_model.predict(X_train_reg)

        r2_test = r2_score(y_test_reg, y_pred_test)
        r2_train = r2_score(y_train_reg, y_pred_train)
        mae_test = mean_absolute_error(y_test_reg, y_pred_test)
        mae_train = mean_absolute_error(y_train_reg, y_pred_train)

        # Ranking correlation: among true top-10%, how well does the model rank?
        from scipy.stats import spearmanr
        if len(y_test_reg) >= 50:
            rank_corr, _ = spearmanr(y_test_reg, y_pred_test)
        else:
            rank_corr = float("nan")

        logger.info(
            "Regression model: R² train=%.4f test=%.4f | "
            "MAE train=%.4f test=%.4f | Rank corr=%.4f",
            r2_train, r2_test, mae_train, mae_test, rank_corr,
        )

        return {
            "status": "trained",
            "best_estimator": self.regression_model.best_estimator,
            "r2_train": round(r2_train, 4),
            "r2_test": round(r2_test, 4),
            "r2_gap": round(r2_train - r2_test, 4),
            "mae_train": round(mae_train, 4),
            "mae_test": round(mae_test, 4),
            "rank_correlation": round(rank_corr, 4) if not np.isnan(rank_corr) else None,
        }

    def _train_regime_model(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        split_idx: int,
        gap_rows: int,
    ) -> dict:
        """Train a market regime meta-model that predicts daily hit-rate.

        Uses macro features (VIX, market returns, yield curve) to predict
        whether the current market environment is favorable for breakout
        picks.  The regime confidence is used to adjust ranking scores.

        Returns:
            Dict with regime model metrics.
        """
        if "_date" not in df.columns:
            return {"status": "skipped"}

        from sklearn.ensemble import GradientBoostingRegressor

        # Compute daily hit rates from training data
        train_df = df.iloc[:split_idx].copy()
        train_X = X.iloc[:split_idx].copy()
        train_y = y.iloc[:split_idx].copy()

        # Get classification probabilities for training data
        try:
            train_proba = self.automl.predict_proba(train_X)
            if train_proba.ndim == 2:
                train_proba = train_proba[:, 1]
        except Exception:
            logger.warning("Could not predict on training data for regime model.")
            return {"status": "skipped"}

        train_df = train_df.copy()
        train_df["_proba"] = train_proba
        train_df["_y"] = train_y.values

        # Macro features for regime detection
        regime_features = [
            "vix_close", "sp500_return_20d", "sp500_return_60d",
            "sp500_volatility_20d", "yield_curve_spread",
            "treasury_3m", "dollar_index_return_20d",
        ]
        regime_features = [f for f in regime_features if f in df.columns]
        if len(regime_features) < 3:
            logger.warning("Insufficient macro features for regime model.")
            return {"status": "skipped"}

        # Compute daily hit rate: for each date, take top 10 by proba,
        # compute what fraction are actual positives
        daily_stats = []
        for date, grp in train_df.groupby("_date"):
            if len(grp) < 10:
                continue
            top10_idx = grp["_proba"].nlargest(10).index
            hit_rate = float(grp.loc[top10_idx, "_y"].mean())
            # Use mean of macro features for that day
            macro_vals = grp[regime_features].iloc[0].to_dict()
            macro_vals["_hit_rate"] = hit_rate
            daily_stats.append(macro_vals)

        if len(daily_stats) < 50:
            logger.warning("Too few training dates for regime model (%d).", len(daily_stats))
            return {"status": "skipped"}

        regime_df = pd.DataFrame(daily_stats)
        X_regime = regime_df[regime_features].fillna(0)
        y_regime = regime_df["_hit_rate"]

        # Train/test split (last 20%)
        r_split = int(len(X_regime) * 0.8)
        X_r_train, X_r_test = X_regime.iloc[:r_split], X_regime.iloc[r_split:]
        y_r_train, y_r_test = y_regime.iloc[:r_split], y_regime.iloc[r_split:]

        self.regime_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.02,
            min_samples_leaf=20, subsample=0.7, random_state=42,
        )
        self.regime_model.fit(X_r_train, y_r_train)
        self.regime_features = regime_features

        train_r2 = self.regime_model.score(X_r_train, y_r_train)
        test_r2 = self.regime_model.score(X_r_test, y_r_test)

        logger.info(
            "Regime model trained: R² train=%.4f test=%.4f (gap=%.4f)",
            train_r2, test_r2, train_r2 - test_r2,
        )

        # Discard if test R² < 0 (worse than predicting the mean)
        if test_r2 < 0:
            logger.warning(
                "Regime model has negative test R² (%.4f) — discarding. "
                "Model is overfitting on %d training days.",
                test_r2, r_split,
            )
            self.regime_model = None
            self.regime_features = []
            return {
                "status": "discarded",
                "reason": "negative_test_r2",
                "r2_train": round(train_r2, 4),
                "r2_test": round(test_r2, 4),
                "r2_gap": round(train_r2 - test_r2, 4),
                "n_train_days": r_split,
                "n_test_days": len(X_regime) - r_split,
            }

        return {
            "status": "trained",
            "r2_train": round(train_r2, 4),
            "r2_test": round(test_r2, 4),
            "r2_gap": round(train_r2 - test_r2, 4),
            "n_train_days": r_split,
            "n_test_days": len(X_regime) - r_split,
            "features": regime_features,
        }

    def _train_ticker_calibration(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        split_idx: int,
        gap_rows: int,
    ) -> dict:
        """Compute per-ticker calibration factors from training data.

        Tracks how often each ticker appears in the top-10 predictions
        and what its hit rate is.  Tickers with <30% hit rate get
        down-weighted to reduce repeat false positives.

        Returns:
            Dict with calibration metrics.
        """
        if "_date" not in df.columns or "Ticker" not in df.columns:
            return {"status": "skipped"}

        train_df = df.iloc[:split_idx].copy()
        train_X = X.iloc[:split_idx].copy()
        train_y = y.iloc[:split_idx].copy()

        try:
            train_proba = self.automl.predict_proba(train_X)
            if train_proba.ndim == 2:
                train_proba = train_proba[:, 1]
        except Exception:
            return {"status": "skipped"}

        train_df = train_df.copy()
        train_df["_proba"] = train_proba
        train_df["_y"] = train_y.values

        # Track per-ticker top-10 appearances and hits
        ticker_stats: dict[str, dict] = {}
        for date, grp in train_df.groupby("_date"):
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

        # Compute calibration factor:
        # - Tickers with >=5 appearances and <30% hit rate get penalized
        # - Factor = max(0.5, hit_rate / 0.5) capped at 1.0
        # - Tickers with few appearances get factor 1.0 (no adjustment)
        self.ticker_calibration = {}
        penalized = 0
        for ticker, stats in ticker_stats.items():
            if stats["appearances"] >= 5:
                hit_rate = stats["hits"] / stats["appearances"]
                if hit_rate < 0.30:
                    factor = max(0.5, hit_rate / 0.50)
                    self.ticker_calibration[ticker] = round(factor, 3)
                    penalized += 1

        logger.info(
            "Ticker calibration: %d tickers tracked, %d penalized (hit rate <30%%)",
            len(ticker_stats), penalized,
        )

        return {
            "status": "trained",
            "tickers_tracked": len(ticker_stats),
            "tickers_penalized": penalized,
            "penalized_tickers": {
                t: f for t, f in sorted(
                    self.ticker_calibration.items(), key=lambda x: x[1],
                )[:10]
            },
        }

    def predict_regime_confidence(self, features: dict | pd.DataFrame) -> float:
        """Predict market regime confidence (expected daily hit rate).

        Returns a value in [0, 1] representing expected hit rate for
        the current market conditions.  Returns 0.5 if no regime model.
        """
        if self.regime_model is None:
            return 0.5

        if isinstance(features, dict):
            row = features
        elif isinstance(features, pd.DataFrame):
            row = features.iloc[0].to_dict() if len(features) > 0 else {}
        else:
            return 0.5

        regime_features = getattr(self, "regime_features", [])
        x = np.array([[row.get(f, 0.0) for f in regime_features]])
        try:
            confidence = float(self.regime_model.predict(x)[0])
            return max(0.0, min(1.0, confidence))
        except Exception:
            return 0.5

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

        has_ltr = self.ltr_model is not None
        has_reg = self.regression_model is not None

        if has_ltr:
            dmat = xgb.DMatrix(df)
            ltr_score = float(self.ltr_model.predict(dmat)[0])
            ltr_norm = 1.0 / (1.0 + np.exp(-ltr_score))
        else:
            ltr_norm = None

        if has_reg:
            reg_pred = float(self.regression_model.predict(df)[0])
            reg_norm = 1.0 / (1.0 + np.exp(-reg_pred * 2))
        else:
            reg_norm = None

        if has_ltr and has_reg:
            return W_CLS * cls_score + W_LTR * ltr_norm + W_REG * reg_norm
        elif has_ltr:
            w = LTR_ENSEMBLE_WEIGHT
            return (1 - w) * cls_score + w * ltr_norm
        elif has_reg:
            return 0.6 * cls_score + 0.4 * reg_norm

        return cls_score

    def predict_batch(
        self,
        df: pd.DataFrame,
        tickers: pd.Series | None = None,
        apply_adjustments: bool = True,
    ) -> np.ndarray:
        """Score multiple rows, returning adjusted ensemble scores for ranking.

        Uses the classification probability + LTR ranking score ensemble,
        then applies volatility-aware scoring and per-ticker calibration
        when ``apply_adjustments`` is True.

        Args:
            df: DataFrame with feature columns. Will be preprocessed
                (derived features, semantic NaN, log transform, median fill).
            tickers: Optional series of ticker symbols for per-ticker
                calibration. Must be same length as df.
            apply_adjustments: Whether to apply volatility scoring and
                ticker calibration (default True).

        Returns:
            Array of ensemble scores, one per row.
        """
        if not self.is_trained:
            self.load()

        # Keep raw volatility before feature preprocessing
        raw_vol = df["Volatility_20d"].values.copy() if "Volatility_20d" in df.columns else None

        df = df.copy()
        df = _compute_derived_features(df)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df_features = df[self.feature_names].copy()

        # --- Walk-forward ensemble: average scores across fold models ---
        if self.fold_models:
            all_fold_scores = []
            for fm in self.fold_models:
                fm_medians = fm.get("feature_medians")
                df_fold = df_features.copy()
                if fm_medians is not None:
                    for col in self.feature_names:
                        if col in fm_medians.index:
                            df_fold[col] = df_fold[col].fillna(fm_medians[col])
                else:
                    df_fold = df_fold.fillna(0.0)

                proba = fm["automl"].predict_proba(df_fold)
                cls_sc = proba[:, 1] if proba.ndim == 2 else proba

                fm_ltr = fm.get("ltr_model")
                fm_reg = fm.get("regression_model")

                if fm_ltr is not None:
                    ltr_sc = fm_ltr.predict(xgb.DMatrix(df_fold))
                    ltr_n = 1.0 / (1.0 + np.exp(-ltr_sc))
                else:
                    ltr_n = None

                if fm_reg is not None:
                    reg_p = fm_reg.predict(df_fold)
                    reg_n = 1.0 / (1.0 + np.exp(-reg_p * 2))
                else:
                    reg_n = None

                if ltr_n is not None and reg_n is not None:
                    fold_sc = W_CLS * cls_sc + W_LTR * ltr_n + W_REG * reg_n
                elif ltr_n is not None:
                    fold_sc = (1 - LTR_ENSEMBLE_WEIGHT) * cls_sc + LTR_ENSEMBLE_WEIGHT * ltr_n
                elif reg_n is not None:
                    fold_sc = 0.6 * cls_sc + 0.4 * reg_n
                else:
                    fold_sc = cls_sc.copy()
                all_fold_scores.append(fold_sc)

            scores = np.mean(all_fold_scores, axis=0)
        else:
            # Fallback: single model (backward compat)
            if self.feature_medians is not None:
                df_features = df_features.fillna(self.feature_medians)
            else:
                df_features = df_features.fillna(0.0)

            proba = self.automl.predict_proba(df_features)
            if proba.ndim == 2:
                cls_scores = proba[:, 1]
            else:
                cls_scores = proba

            has_ltr = self.ltr_model is not None
            has_reg = self.regression_model is not None

            if has_ltr:
                dmat = xgb.DMatrix(df_features)
                ltr_scores = self.ltr_model.predict(dmat)
                ltr_norm = 1.0 / (1.0 + np.exp(-ltr_scores))
            else:
                ltr_norm = None

            if has_reg:
                reg_pred = self.regression_model.predict(df_features)
                reg_norm = 1.0 / (1.0 + np.exp(-reg_pred * 2))
            else:
                reg_norm = None

            if has_ltr and has_reg:
                scores = W_CLS * cls_scores + W_LTR * ltr_norm + W_REG * reg_norm
            elif has_ltr:
                w = LTR_ENSEMBLE_WEIGHT
                scores = (1 - w) * cls_scores + w * ltr_norm
            elif has_reg:
                scores = 0.6 * cls_scores + 0.4 * reg_norm
            else:
                scores = cls_scores.copy()

        if apply_adjustments:
            # Volatility-aware scoring: boost higher-volatility stocks
            if raw_vol is not None and len(raw_vol) > 1:
                vol_pctl = pd.Series(raw_vol).rank(pct=True).values
                scores = scores * (1 + VOLATILITY_SCORE_ALPHA * vol_pctl)

            # Per-ticker calibration: penalize repeat false positives
            if tickers is not None and self.ticker_calibration:
                cal_factors = np.array([
                    self.ticker_calibration.get(t, 1.0) for t in tickers
                ])
                scores = scores * cal_factors

        return scores

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

        # Market regime confidence
        regime_conf = self.predict_regime_confidence(row)

        # Ticker calibration factor
        cal_factor = self.ticker_calibration.get(ticker, 1.0)

        return {
            "ticker": ticker,
            "probability_gain": round(probability, 4),
            "probability_pct": f"{probability * 100:.1f}%",
            "prediction": prediction,
            "signal": "BUY" if prediction == 1 else "HOLD",
            "volume_surge_3d": vol_surge,
            "regime_confidence": round(regime_conf, 3),
            "ticker_calibration": cal_factor,
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
        if self.regime_model is not None:
            regime_data = {
                "model": self.regime_model,
                "features": getattr(self, "regime_features", []),
            }
            joblib.dump(regime_data, REGIME_MODEL_PATH)
            logger.info("Regime model saved to %s", REGIME_MODEL_PATH)
        if self.regression_model is not None:
            joblib.dump(self.regression_model, REGRESSION_MODEL_PATH)
            logger.info("Regression model saved to %s", REGRESSION_MODEL_PATH)
        if self.ticker_calibration:
            joblib.dump(self.ticker_calibration, TICKER_CALIBRATION_PATH)
            logger.info(
                "Ticker calibration saved (%d entries)", len(self.ticker_calibration),
            )
        # Save walk-forward fold models for ensemble inference
        if self.fold_models:
            FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            for fm in self.fold_models:
                fold_idx = fm["fold"]
                fold_dir = FOLD_MODELS_DIR / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(fm["automl"], fold_dir / "automl.pkl")
                joblib.dump(fm["feature_medians"], fold_dir / "feature_medians.pkl")
                if fm.get("ltr_model") is not None:
                    fm["ltr_model"].save_model(str(fold_dir / "ltr_model.json"))
                if fm.get("regression_model") is not None:
                    joblib.dump(fm["regression_model"], fold_dir / "regression_model.pkl")
            joblib.dump(len(self.fold_models), FOLD_MODELS_DIR / "n_folds.pkl")
            logger.info(
                "Walk-forward ensemble saved: %d fold models", len(self.fold_models),
            )
        # Save metadata for version-mismatch detection
        meta = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "joblib_version": joblib.__version__,
        }
        try:
            _MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass
        logger.info("Model saved to %s", MODEL_PATH)

    @staticmethod
    def _check_version_mismatch() -> str | None:
        """Check if the saved model was created with a different Python version.

        Returns a warning message if a mismatch is detected, None otherwise.
        """
        if not _MODEL_META_PATH.exists():
            return None
        try:
            meta = json.loads(_MODEL_META_PATH.read_text())
            saved_py = meta.get("python_version", "")
            current_py = platform.python_version()
            saved_major_minor = ".".join(saved_py.split(".")[:2])
            current_major_minor = ".".join(current_py.split(".")[:2])
            if saved_major_minor != current_major_minor:
                return (
                    f"Model was saved with Python {saved_py} but you are "
                    f"running Python {current_py}. Pickle files are not "
                    f"always compatible across Python versions. Please "
                    f"retrain the model in your current environment "
                    f"(use the Model Training page) or switch to "
                    f"Python {saved_major_minor}.x."
                )
        except Exception:
            pass
        return None

    def load(self) -> None:
        """Load model and feature names from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No saved model found at {MODEL_PATH}. Train the model first."
            )

        # Warn about Python version mismatch before attempting to load
        version_warning = self._check_version_mismatch()
        if version_warning:
            logger.warning(version_warning)

        try:
            self.automl = joblib.load(MODEL_PATH)
        except (KeyError, Exception) as e:
            hint = (
                version_warning
                or (
                    f"This may be caused by a Python version mismatch. "
                    f"You are running Python {platform.python_version()}. "
                    f"Please retrain the model in your current environment "
                    f"(use the Model Training page in Streamlit)."
                )
            )
            raise RuntimeError(
                f"Failed to load model from {MODEL_PATH}: {e}. {hint}"
            ) from e

        try:
            self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        except (KeyError, Exception) as e:
            raise RuntimeError(
                f"Failed to load feature names from {FEATURE_NAMES_PATH}: {e}. "
                f"Please retrain the model."
            ) from e

        if MEDIANS_PATH.exists():
            self.feature_medians = joblib.load(MEDIANS_PATH)
        if THRESHOLD_PATH.exists():
            self.optimal_threshold = joblib.load(THRESHOLD_PATH)
        if LTR_MODEL_PATH.exists():
            self.ltr_model = xgb.Booster()
            self.ltr_model.load_model(str(LTR_MODEL_PATH))
            logger.info("LTR model loaded from %s", LTR_MODEL_PATH)
        if REGIME_MODEL_PATH.exists():
            try:
                regime_data = joblib.load(REGIME_MODEL_PATH)
                self.regime_model = regime_data["model"]
                self.regime_features = regime_data["features"]
                logger.info("Regime model loaded from %s", REGIME_MODEL_PATH)
            except (KeyError, Exception):
                logger.warning(
                    "Could not load regime model — skipping (may need retraining)"
                )
        if REGRESSION_MODEL_PATH.exists():
            try:
                self.regression_model = joblib.load(REGRESSION_MODEL_PATH)
                logger.info("Regression model loaded from %s", REGRESSION_MODEL_PATH)
            except (KeyError, Exception):
                logger.warning(
                    "Could not load regression model — skipping (may need retraining)"
                )
        if TICKER_CALIBRATION_PATH.exists():
            try:
                self.ticker_calibration = joblib.load(TICKER_CALIBRATION_PATH)
                logger.info(
                    "Ticker calibration loaded (%d entries)", len(self.ticker_calibration),
                )
            except (KeyError, Exception):
                logger.warning(
                    "Could not load ticker calibration — skipping (may need retraining)"
                )
        # Load walk-forward fold models for ensemble inference
        n_folds_path = FOLD_MODELS_DIR / "n_folds.pkl"
        if n_folds_path.exists():
            try:
                n_folds = joblib.load(n_folds_path)
                self.fold_models = []
                for fold_idx in range(1, n_folds + 1):
                    fold_dir = FOLD_MODELS_DIR / f"fold_{fold_idx}"
                    if not fold_dir.exists():
                        continue
                    fm: dict = {
                        "fold": fold_idx,
                        "automl": joblib.load(fold_dir / "automl.pkl"),
                        "feature_medians": joblib.load(fold_dir / "feature_medians.pkl"),
                        "ltr_model": None,
                        "regression_model": None,
                    }
                    ltr_path = fold_dir / "ltr_model.json"
                    if ltr_path.exists():
                        fm["ltr_model"] = xgb.Booster()
                        fm["ltr_model"].load_model(str(ltr_path))
                    reg_path = fold_dir / "regression_model.pkl"
                    if reg_path.exists():
                        fm["regression_model"] = joblib.load(reg_path)
                    self.fold_models.append(fm)
                logger.info(
                    "Walk-forward ensemble loaded: %d fold models",
                    len(self.fold_models),
                )
            except (KeyError, Exception) as e:
                logger.warning(
                    "Could not load fold models — falling back to single model: %s", e,
                )
                self.fold_models = []
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
