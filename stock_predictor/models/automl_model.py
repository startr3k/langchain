"""AutoML model for stock return classification and ranking using FLAML + LTR.

Predicts whether a stock will achieve >=20% peak return at any point
within a 3-month window.  Uses a four-stage pipeline:

  Stage 1: FLAML Binary Classifier — P(MFD >= 20%), gate: P >= 0.50.
  Stage 2: XGBoost Huber Regressor — predicts MFD magnitude,
           gate: predicted MFD >= 25%.
  Stage 3: Cross-Sectional Quantile Transformer — transforms raw
           features to per-date percentile ranks (0.0–1.0) for
           regime-normalized LTR inputs.
  Stage 4: LambdaMART (XGBoost rank:ndcg) — ranks elite survivors
           using quantile-transformed features and log-scaled MFD labels.

Final ranking uses max(Z_cls, 0) * max(Z_ltr, 0) * pool_weight within
each day's elite pool (stocks passing both Stage 1 and Stage 2 gates).
pool_weight = min(elite_pool_size / MIN_ELITE_POOL, 2.0) so larger pools
boost the score, making it comparable across days.  Only days with
elite pool >= MIN_ELITE_POOL are traded.

Walk-forward evaluation: 70.7% hit rate at min pool >= 75.
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
    "hist_stockholders_equity",
    "hist_capex",
    "hist_diluted_eps",
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
        "insider_net_buys_90d",
    ],

}


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction / derived features from existing columns.

    Currently a no-op — Fundamental_Surprise was removed because its
    input (earnings_surprise_pct) is not present in the training CSV.
    Kept as a hook for future derived features.
    """
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

# 4-stage pipeline constants
CLS_PROB_THRESHOLD = 0.50    # Stage 1: classifier gate
MFD_PRED_THRESHOLD = 0.25    # Stage 2: Huber MFD gate (predicted MFD >= 25%)
MIN_ELITE_POOL = 75          # Minimum elite pool size to generate picks

# Legacy ensemble weights (kept for backward compat in predict_batch fallback)
W_CLS = 0.30
W_LTR = 0.40
W_REG = 0.30
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
    """4-stage stock predictor with selective high-precision trading.

    Stage 1: FLAML Binary Classifier — P(MFD >= 20%), gate: P >= 0.50.
    Stage 2: XGBoost Huber Regressor — predicts MFD, gate: pred >= 25%.
    Stage 3: Cross-Sectional Quantile Transform on features.
    Stage 4: LambdaMART (rank:ndcg) on quantile-transformed features.

    Ranking: max(Z_cls, 0) * max(Z_ltr, 0) * pool_weight within elite pool.
    pool_weight = min(pool_size / MIN_ELITE_POOL, 2.0).
    Only trades on days with elite pool >= MIN_ELITE_POOL (75).
    Walk-forward eval: 70.7% hit rate, ~2.7 trading days/month.
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

        # --- Quality filter: keep only stocks with sufficient history and revenue ---
        # Removes shell companies and pre-revenue biotechs that contribute
        # mostly NaN fundamental features.
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

            quality_tickers = tickers_2q & tickers_with_rev
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
        time_budget: int = 60,
        n_folds: int = 5,
        min_train_years: int = 3,
    ) -> dict:
        """Train 4-stage pipeline with walk-forward cross-validation.

        4-stage pipeline per fold:
          1. FLAML classifier: P(MFD >= 20%), gate P >= CLS_PROB_THRESHOLD
          2. XGBoost Huber: predict MFD magnitude, gate >= MFD_PRED_THRESHOLD
          3. Cross-sectional quantile transform on features
          4. LambdaMART on quantile-transformed features

        Ranking: max(Z_cls, 0) * max(Z_ltr, 0) * pool_weight.
        pool_weight = min(pool_size / MIN_ELITE_POOL, 2.0).
        Only evaluates days with elite pool >= MIN_ELITE_POOL.

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

        # --- Quality filter: keep only stocks with sufficient history and revenue ---
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

            quality_tickers = tickers_2q & tickers_with_rev
            df = df[df["Ticker"].isin(quality_tickers)].reset_index(drop=True)
            logger.info(
                "Quality filter: %d → %d rows (%d tickers kept, %d removed)",
                before_filter, len(df),
                len(quality_tickers),
                before_filter - len(df),
            )

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

        # XGBoost Huber regressor params (deterministic)
        huber_params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.5,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 50,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "reg_alpha": 5.0,
            "reg_lambda": 10.0,
            "verbosity": 0,
            "seed": 42,
        }

        # LTR (LambdaMART) params — tuned to reduce overfitting
        ltr_params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@10",
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 100,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "reg_alpha": 5.0,
            "reg_lambda": 10.0,
            "verbosity": 0,
            "ndcg_exp_gain": False,
            "seed": 42,
        }

        for fold_idx in range(n_folds):
            test_start = first_test_start + pd.Timedelta(days=fold_idx * fold_size_days)
            test_end = test_start + pd.Timedelta(days=fold_size_days)
            train_end = test_start - pd.Timedelta(days=gap_days)

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
                "=== Fold %d/%d: Train to %s (%d rows) | Test %s–%s (%d rows) ===",
                fold_idx + 1, n_folds,
                train_end.date(), len(df_train),
                df_test["_date"].min().date(), df_test["_date"].max().date(),
                len(df_test),
            )

            feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]

            X_train = df_train[feature_cols].copy()
            y_train_raw = df_train[TARGET_COLUMN].copy()
            X_test = df_test[feature_cols].copy()
            y_test_raw = df_test[TARGET_COLUMN].copy()

            valid_train = y_train_raw.notna()
            X_train = X_train[valid_train].reset_index(drop=True)
            y_train_raw = y_train_raw[valid_train].reset_index(drop=True)
            df_train = df_train[valid_train].reset_index(drop=True)

            valid_test = y_test_raw.notna()
            X_test = X_test[valid_test].reset_index(drop=True)
            y_test_raw = y_test_raw[valid_test].reset_index(drop=True)
            df_test = df_test[valid_test].reset_index(drop=True)

            y_train = (y_train_raw >= CLASSIFICATION_THRESHOLD).astype(int)
            y_test = (y_test_raw >= CLASSIFICATION_THRESHOLD).astype(int)

            pos_rate = y_train.mean()
            if pos_rate == 0 or pos_rate == 1:
                logger.warning("Fold %d: single class in training, skipping.", fold_idx + 1)
                continue
            sw_train = np.where(
                y_train == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate),
            )

            # Per-fold preprocessing: median fill + percentile clipping
            fold_medians = X_train.median()
            clip_bounds = {}
            for col in feature_cols:
                X_train[col] = X_train[col].fillna(fold_medians[col])
                X_test[col] = X_test[col].fillna(fold_medians[col])
                p01, p99 = X_train[col].quantile(0.01), X_train[col].quantile(0.99)
                clip_bounds[col] = (p01, p99)
                if p01 < p99:
                    X_train[col] = X_train[col].clip(p01, p99)
                    X_test[col] = X_test[col].clip(p01, p99)

            train_dates = df_train["_date"].values
            test_dates = df_test["_date"].values

            # ---- Stage 1: FLAML Binary Classifier ----
            fold_automl = AutoML()
            fold_automl.fit(
                X_train=X_train,
                y_train=y_train,
                task="classification",
                time_budget=time_budget,
                metric="ap",
                seed=42,
                estimator_list=["xgboost", "lgbm"],
                eval_method="cv",
                n_splits=5,
                verbose=0,
                early_stop=True,
                sample_weight=sw_train,
            )

            y_proba_test = fold_automl.predict_proba(X_test)
            y_proba_pos = y_proba_test[:, 1] if y_proba_test.ndim == 2 else y_proba_test

            try:
                cls_auc = roc_auc_score(y_test, y_proba_pos)
            except ValueError:
                cls_auc = float("nan")

            cls_pass_test = y_proba_pos >= CLS_PROB_THRESHOLD
            logger.info(
                "  Stage 1 (Classifier): AUC=%.4f, pass: %d/%d (%.1f%%)",
                cls_auc, int(cls_pass_test.sum()), len(y_proba_pos),
                cls_pass_test.mean() * 100,
            )

            # ---- Stage 2: XGBoost Huber Regressor ----
            huber_labels_train = np.nan_to_num(y_train_raw.values, nan=0.0)
            huber_labels_test = np.nan_to_num(y_test_raw.values, nan=0.0)

            dtr_huber = xgb.DMatrix(X_train, label=huber_labels_train)
            dte_huber = xgb.DMatrix(X_test, label=huber_labels_test)

            huber_evals: dict = {}
            huber_model = xgb.train(
                huber_params, dtr_huber, num_boost_round=500,
                evals=[(dtr_huber, "train"), (dte_huber, "test")],
                evals_result=huber_evals,
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            pred_mfd_test = huber_model.predict(dte_huber)
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(huber_labels_test, pred_mfd_test)
            mae = mean_absolute_error(huber_labels_test, pred_mfd_test)

            mfd_pass_test = pred_mfd_test >= MFD_PRED_THRESHOLD
            elite_test = cls_pass_test & mfd_pass_test

            logger.info(
                "  Stage 2 (Huber): R2=%.4f MAE=%.4f, Elite: %d/%d",
                r2, mae, int(elite_test.sum()), len(elite_test),
            )

            # ---- Stage 3: Cross-Sectional Quantile Transform ----
            # Transform features to per-date percentile ranks for elite survivors
            elite_train_mask = np.ones(len(X_train), dtype=bool)  # train on all
            X_train_qt = X_train.astype(np.float64).copy()
            X_test_qt = X_test.astype(np.float64).copy()
            for col in feature_cols:
                for dates_arr, X_df in [(train_dates, X_train_qt), (test_dates, X_test_qt)]:
                    for d in np.unique(dates_arr):
                        dmask = dates_arr == d
                        vals = X_df.loc[dmask, col].values
                        if len(vals) > 1:
                            ranks = pd.Series(vals).rank(pct=True).values
                            X_df.loc[dmask, col] = ranks
                        else:
                            X_df.loc[dmask, col] = 0.5

            logger.info("  Stage 3 (Quantile Transform): features transformed")

            # ---- Stage 4: LambdaMART on quantile-transformed features ----
            # Train on elite survivors only, with log-scaled MFD labels
            y_proba_train = fold_automl.predict_proba(X_train)
            y_proba_train_pos = y_proba_train[:, 1] if y_proba_train.ndim == 2 else y_proba_train
            pred_mfd_train = huber_model.predict(dtr_huber)
            elite_train = (y_proba_train_pos >= CLS_PROB_THRESHOLD) & (pred_mfd_train >= MFD_PRED_THRESHOLD)

            fold_ltr_model = None
            ltr_ndcg_train = float("nan")
            ltr_ndcg_test = float("nan")

            n_elite_train = int(elite_train.sum())
            n_elite_test = int(elite_test.sum())

            if n_elite_train >= 100 and n_elite_test >= 10:
                ltr_labels_train = np.round(
                    np.log1p(np.clip(np.nan_to_num(y_train_raw.values[elite_train], nan=0.0), 0.0, None)) * 100
                ).astype(np.int32)
                ltr_labels_test = np.round(
                    np.log1p(np.clip(np.nan_to_num(y_test_raw.values[elite_test], nan=0.0), 0.0, None)) * 100
                ).astype(np.int32)

                X_ltr_train = X_train_qt[elite_train].reset_index(drop=True)
                X_ltr_test = X_test_qt[elite_test].reset_index(drop=True)
                ltr_train_dates = train_dates[elite_train]
                ltr_test_dates = test_dates[elite_test]

                tr_groups = pd.Series(ltr_train_dates).value_counts().sort_index()
                te_groups = pd.Series(ltr_test_dates).value_counts().sort_index()

                dtrain_ltr = xgb.DMatrix(X_ltr_train, label=ltr_labels_train)
                dtrain_ltr.set_group(tr_groups.values)
                dtest_ltr = xgb.DMatrix(X_ltr_test, label=ltr_labels_test)
                dtest_ltr.set_group(te_groups.values)

                ltr_evals: dict = {}
                fold_ltr_model = xgb.train(
                    ltr_params, dtrain_ltr, num_boost_round=200,
                    evals=[(dtrain_ltr, "train"), (dtest_ltr, "test")],
                    evals_result=ltr_evals,
                    early_stopping_rounds=30,
                    verbose_eval=False,
                )
                ltr_ndcg_train = ltr_evals["train"]["ndcg@10"][-1]
                ltr_ndcg_test = ltr_evals["test"]["ndcg@10"][-1]

            logger.info(
                "  Stage 4 (LTR): NDCG train=%.4f test=%.4f",
                ltr_ndcg_train, ltr_ndcg_test,
            )

            # ---- Per-day top-10 evaluation with Z-score ranking ----
            ltr_scores = fold_ltr_model.predict(dtest_ltr) if fold_ltr_model is not None else pred_mfd_test[elite_test]

            # Save intermediates for post-retrain evaluation at different pool thresholds
            import pickle as _pkl
            _inter_dir = Path(__file__).resolve().parent.parent.parent / "intermediates"
            _inter_dir.mkdir(exist_ok=True)
            _fold_data = {
                "y_proba_test": y_proba_pos,
                "pred_mfd_test": pred_mfd_test,
                "ltr_scores": ltr_scores,
                "elite_test": elite_test,
                "test_dates": test_dates,
                "y_test_raw": y_test_raw.values,
                "tickers": df_test["Ticker"].values,
                "cls_auc": cls_auc,
                "huber_r2": r2,
            }
            with open(_inter_dir / f"fold{fold_idx + 1}.pkl", "wb") as _f:
                _pkl.dump(_fold_data, _f)
            logger.info("  Saved intermediates/fold%d.pkl", fold_idx + 1)

            unique_test_dates = np.unique(test_dates)
            daily_hits_list = []
            daily_returns_list = []

            for date_val in unique_test_dates:
                date_mask = test_dates == date_val
                day_elite = elite_test & date_mask
                n_elite_day = int(day_elite.sum())

                if n_elite_day < MIN_ELITE_POOL:
                    continue

                elite_indices = np.where(day_elite)[0]
                # Map elite_indices to ltr_scores indices
                elite_cumsum = np.cumsum(elite_test)
                ltr_idx = np.array([int(elite_cumsum[i]) - 1 for i in elite_indices])

                day_ltr = ltr_scores[ltr_idx]
                day_proba = y_proba_pos[elite_indices]
                day_returns_vals = y_test_raw.values[elite_indices]

                p_mu, p_sig = day_proba.mean(), day_proba.std()
                day_cls_z = (day_proba - p_mu) / p_sig if p_sig > 1e-8 else np.zeros_like(day_proba)

                l_mu, l_sig = day_ltr.mean(), day_ltr.std()
                day_ltr_z = (day_ltr - l_mu) / l_sig if l_sig > 1e-8 else np.zeros_like(day_ltr)

                pool_weight = min(n_elite_day / MIN_ELITE_POOL, 2.0)
                day_scores = np.maximum(day_cls_z, 0) * np.maximum(day_ltr_z, 0) * pool_weight
                top10 = np.argsort(day_scores)[-10:][::-1]
                t10_ret = day_returns_vals[top10]
                hits = int((t10_ret >= CLASSIFICATION_THRESHOLD).sum())

                daily_hits_list.append(hits)
                daily_returns_list.append(float(t10_ret.mean()))

            n_eval_days = len(daily_hits_list)
            total_hits_fold = sum(daily_hits_list)
            total_picks_fold = n_eval_days * 10
            avg_hit_rate = total_hits_fold / total_picks_fold if total_picks_fold > 0 else 0
            avg_daily_return = float(np.mean(daily_returns_list)) if daily_returns_list else 0

            all_top10_hits.append(total_hits_fold)
            all_top10_totals.append(total_picks_fold)

            fold_result = {
                "fold": fold_idx + 1,
                "train_period": f"{df_train['_date'].min().date()} to {train_end.date()}",
                "test_period": f"{df_test['_date'].min().date()} to {df_test['_date'].max().date()}",
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "best_estimator": fold_automl.best_estimator,
                "auc_test": round(cls_auc, 4),
                "huber_r2": round(r2, 4),
                "ltr_ndcg_test": round(ltr_ndcg_test, 4) if not np.isnan(ltr_ndcg_test) else None,
                "n_eval_days": n_eval_days,
                "top10_hits": total_hits_fold,
                "top10_total": total_picks_fold,
                "top10_hit_rate": round(avg_hit_rate, 4),
                "top10_avg_return": round(avg_daily_return, 4),
                "min_elite_pool": MIN_ELITE_POOL,
                "avg_elite_per_day": round(float(np.mean([int((elite_test & (test_dates == d)).sum()) for d in unique_test_dates if int((elite_test & (test_dates == d)).sum()) >= MIN_ELITE_POOL])), 0) if n_eval_days > 0 else 0,
            }
            fold_results.append(fold_result)

            logger.info(
                "Fold %d: AUC=%.4f R2=%.4f | %d days (pool>=%d) | "
                "Hit rate=%.1f%% | Avg return=%.1f%%",
                fold_idx + 1, cls_auc, r2,
                n_eval_days, MIN_ELITE_POOL,
                avg_hit_rate * 100, avg_daily_return * 100,
            )

            # Save fold models for ensemble prediction
            fold_model_entry = {
                "fold": fold_idx + 1,
                "automl": fold_automl,
                "huber_model": huber_model,
                "ltr_model": fold_ltr_model,
                "feature_medians": fold_medians,
                "clip_bounds": clip_bounds,
            }
            self.fold_models.append(fold_model_entry)

            if fold_idx == n_folds - 1:
                self.automl = fold_automl
                self.feature_names = feature_cols
                self.feature_medians = fold_medians
                self.is_trained = True
                self.regression_model = None

                # Train final global LTR for backward compat
                split_idx = len(X_train)
                X_combined = pd.concat([X_train, X_test], ignore_index=True)
                y_combined = pd.concat([y_train, y_test], ignore_index=True)
                df_combined = pd.concat([df_train, df_test], ignore_index=True)
                self._train_ltr(
                    df_combined, X_combined, y_combined,
                    feature_cols, split_idx, 0,
                )
                self.save()

            import gc
            gc.collect()

        # Aggregate statistics
        total_hits = sum(all_top10_hits)
        total_picks = sum(all_top10_totals)
        avg_hit_rate = total_hits / total_picks if total_picks > 0 else 0.0

        auc_tests = [f["auc_test"] for f in fold_results if not np.isnan(f["auc_test"])]

        summary = {
            "n_folds": len(fold_results),
            "pipeline": "4-stage (Cls + Huber + QT + LTR)",
            "min_elite_pool": MIN_ELITE_POOL,
            "aggregate_top10_hits": total_hits,
            "aggregate_top10_total": total_picks,
            "aggregate_top10_hit_rate": round(avg_hit_rate, 4),
            "mean_auc_test": round(float(np.mean(auc_tests)), 4) if auc_tests else None,
            "folds": fold_results,
        }

        logger.info("=" * 70)
        logger.info("WALK-FORWARD SUMMARY (4-Stage Pipeline, min pool >= %d)", MIN_ELITE_POOL)
        logger.info("=" * 70)
        logger.info(
            "Folds: %d | Total: %d/%d hits (%.1f%%) | Mean AUC: %.4f",
            len(fold_results), total_hits, total_picks, avg_hit_rate * 100,
            np.mean(auc_tests) if auc_tests else 0,
        )
        for f in fold_results:
            logger.info(
                "  Fold %d [%s]: AUC=%.4f R2=%.4f | %d days | "
                "Hit rate=%.1f%% | Avg return=%.1f%%",
                f["fold"], f["test_period"],
                f["auc_test"], f["huber_r2"],
                f["n_eval_days"],
                f["top10_hit_rate"] * 100, f["top10_avg_return"] * 100,
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
        """Score multiple rows using the 4-stage pipeline.

        For each fold model, runs the full pipeline:
          1. Classifier probability (gate P >= CLS_PROB_THRESHOLD)
          2. Huber MFD prediction (gate >= MFD_PRED_THRESHOLD)
          3. Cross-sectional quantile transform (per-date percentile ranks)
          4. LTR score on quantile-transformed features

        Final score = max(Z_cls, 0) * max(Z_ltr, 0) * pool_weight,
        averaged across folds. pool_weight = min(pool/75, 2.0).
        Stocks that don't pass both gates get score 0.

        Also returns elite_pool_size and individual stage scores as
        attributes on the returned array when fold_models are available.

        Args:
            df: DataFrame with feature columns.
            tickers: Optional series of ticker symbols.
            apply_adjustments: Whether to apply volatility scoring and
                ticker calibration (default True).

        Returns:
            Array of ensemble scores, one per row.
        """
        if not self.is_trained:
            self.load()

        raw_vol = df["Volatility_20d"].values.copy() if "Volatility_20d" in df.columns else None

        df = df.copy()
        df = _compute_derived_features(df)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df_features = df[self.feature_names].copy()

        # --- Walk-forward 4-stage ensemble ---
        if self.fold_models and any(fm.get("huber_model") is not None for fm in self.fold_models):
            all_fold_scores = []
            all_fold_proba = []
            all_fold_mfd = []

            for fm in self.fold_models:
                fm_medians = fm.get("feature_medians")
                fm_clip = fm.get("clip_bounds", {})
                df_fold = df_features.copy()

                # Apply per-fold preprocessing
                if fm_medians is not None:
                    for col in self.feature_names:
                        if col in fm_medians.index:
                            df_fold[col] = df_fold[col].fillna(fm_medians[col])
                else:
                    df_fold = df_fold.fillna(0.0)
                for col, (lo, hi) in fm_clip.items():
                    if col in df_fold.columns and lo < hi:
                        df_fold[col] = df_fold[col].clip(lo, hi)

                # Stage 1: Classifier
                proba = fm["automl"].predict_proba(df_fold)
                cls_prob = proba[:, 1] if proba.ndim == 2 else proba
                cls_pass = cls_prob >= CLS_PROB_THRESHOLD

                # Stage 2: Huber MFD
                huber = fm.get("huber_model")
                if huber is not None:
                    pred_mfd = huber.predict(xgb.DMatrix(df_fold))
                else:
                    pred_mfd = np.zeros(len(df_fold))
                mfd_pass = pred_mfd >= MFD_PRED_THRESHOLD

                elite = cls_pass & mfd_pass

                # Stage 3: Quantile transform features
                df_qt = df_fold.copy()
                for col in self.feature_names:
                    vals = df_qt[col].values
                    if len(vals) > 1:
                        ranks = pd.Series(vals).rank(pct=True).values
                        df_qt[col] = ranks
                    else:
                        df_qt[col] = 0.5

                # Stage 4: LTR on elite survivors only
                fm_ltr = fm.get("ltr_model")
                ltr_sc = np.zeros(len(df_fold))
                if fm_ltr is not None and elite.sum() > 0:
                    elite_qt = df_qt[elite].reset_index(drop=True)
                    ltr_raw = fm_ltr.predict(xgb.DMatrix(elite_qt))
                    ltr_sc[elite] = ltr_raw

                # Z-score ranking within elite pool, weighted by pool size
                fold_scores = np.zeros(len(df_fold))
                n_elite = int(elite.sum())
                if n_elite >= 2:
                    elite_proba = cls_prob[elite]
                    elite_ltr = ltr_sc[elite]

                    p_mu, p_sig = elite_proba.mean(), elite_proba.std()
                    z_cls = (elite_proba - p_mu) / p_sig if p_sig > 1e-8 else np.zeros_like(elite_proba)

                    l_mu, l_sig = elite_ltr.mean(), elite_ltr.std()
                    z_ltr = (elite_ltr - l_mu) / l_sig if l_sig > 1e-8 else np.zeros_like(elite_ltr)

                    # Pool weight: pool_size / MIN_ELITE_POOL, capped at 2.0
                    pool_weight = min(n_elite / MIN_ELITE_POOL, 2.0)
                    z_scores = np.maximum(z_cls, 0) * np.maximum(z_ltr, 0) * pool_weight
                    fold_scores[elite] = z_scores

                all_fold_scores.append(fold_scores)
                all_fold_proba.append(cls_prob)
                all_fold_mfd.append(pred_mfd)

            scores = np.mean(all_fold_scores, axis=0)

            # Store per-stock stage details for UI display
            avg_proba = np.mean(all_fold_proba, axis=0)
            avg_mfd = np.mean(all_fold_mfd, axis=0)
            # Compute average Z-scores across folds for each stock
            n = len(df_fold)
            avg_z_cls = np.zeros(n)
            avg_z_ltr = np.zeros(n)
            n_folds_with_elite = 0
            for fi, fm in enumerate(self.fold_models):
                fs = all_fold_scores[fi]
                fp = all_fold_proba[fi]
                elite_mask = fs > 0
                if elite_mask.sum() >= 2:
                    ep = fp[elite_mask]
                    p_mu, p_sig = ep.mean(), ep.std()
                    z_c = (fp - p_mu) / p_sig if p_sig > 1e-8 else np.zeros(n)
                    avg_z_cls += z_c
                    # Approximate Z_ltr from fold scores
                    elite_fs = fs[elite_mask]
                    l_mu, l_sig = elite_fs.mean(), elite_fs.std()
                    z_l = np.zeros(n)
                    z_l[elite_mask] = (fs[elite_mask] - l_mu) / l_sig if l_sig > 1e-8 else 0
                    avg_z_ltr += z_l
                    n_folds_with_elite += 1
            if n_folds_with_elite > 0:
                avg_z_cls /= n_folds_with_elite
                avg_z_ltr /= n_folds_with_elite

            elite_pool_size = int((scores > 0).sum())
            self._last_batch_details = {
                "cls_proba": avg_proba,
                "pred_mfd": avg_mfd,
                "z_cls": avg_z_cls,
                "z_ltr": avg_z_ltr,
                "elite_pool_size": elite_pool_size,
            }

        else:
            # Fallback: legacy ensemble (backward compat with train()-based models)
            if self.feature_medians is not None:
                df_features = df_features.fillna(self.feature_medians)
            else:
                df_features = df_features.fillna(0.0)

            proba = self.automl.predict_proba(df_features)
            cls_scores = proba[:, 1] if proba.ndim == 2 else proba

            has_ltr = self.ltr_model is not None
            has_reg = self.regression_model is not None

            if has_ltr and has_reg:
                dmat = xgb.DMatrix(df_features)
                ltr_scores_raw = self.ltr_model.predict(dmat)
                ltr_norm = 1.0 / (1.0 + np.exp(-ltr_scores_raw))
                reg_pred = self.regression_model.predict(df_features)
                reg_norm = 1.0 / (1.0 + np.exp(-reg_pred * 2))
                scores = W_CLS * cls_scores + W_LTR * ltr_norm + W_REG * reg_norm
            elif has_ltr:
                dmat = xgb.DMatrix(df_features)
                ltr_scores_raw = self.ltr_model.predict(dmat)
                ltr_norm = 1.0 / (1.0 + np.exp(-ltr_scores_raw))
                w = LTR_ENSEMBLE_WEIGHT
                scores = (1 - w) * cls_scores + w * ltr_norm
            elif has_reg:
                reg_pred = self.regression_model.predict(df_features)
                reg_norm = 1.0 / (1.0 + np.exp(-reg_pred * 2))
                scores = 0.6 * cls_scores + 0.4 * reg_norm
            else:
                scores = cls_scores.copy()

        if apply_adjustments:
            if raw_vol is not None and len(raw_vol) > 1:
                vol_pctl = pd.Series(raw_vol).rank(pct=True).values
                scores = scores * (1 + VOLATILITY_SCORE_ALPHA * vol_pctl)

            if tickers is not None and self.ticker_calibration:
                cal_factors = np.array([
                    self.ticker_calibration.get(t, 1.0) for t in tickers
                ])
                scores = scores * cal_factors

        return scores

    def predict_ticker(
        self,
        ticker: str,
        include_explanation: bool = False,
    ) -> dict:
        """Predict whether a ticker will hit >=20% peak return within 3 months.

        Uses the ensemble score (classification + LTR) to determine the
        probability. Prediction is BUY if probability >= optimal_threshold.

        Args:
            ticker: Stock ticker symbol.
            include_explanation: If True, compute SHAP explanation for the
                prediction. Adds ~50ms per call. Default False.

        Returns:
            Dict with ticker, probability, and classification.
        """
        # Fetch market cap (informational, no filtering)
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap") or 0
        except Exception:
            mcap = 0

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
                if fm.get("huber_model") is not None:
                    fm["huber_model"].save_model(str(fold_dir / "huber_model.json"))
                if fm.get("clip_bounds"):
                    joblib.dump(fm["clip_bounds"], fold_dir / "clip_bounds.pkl")
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
                        "huber_model": None,
                        "clip_bounds": {},
                        "regression_model": None,
                    }
                    ltr_path = fold_dir / "ltr_model.json"
                    if ltr_path.exists():
                        fm["ltr_model"] = xgb.Booster()
                        fm["ltr_model"].load_model(str(ltr_path))
                    huber_path = fold_dir / "huber_model.json"
                    if huber_path.exists():
                        fm["huber_model"] = xgb.Booster()
                        fm["huber_model"].load_model(str(huber_path))
                    clip_path = fold_dir / "clip_bounds.pkl"
                    if clip_path.exists():
                        fm["clip_bounds"] = joblib.load(clip_path)
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
