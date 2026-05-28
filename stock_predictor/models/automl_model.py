"""AutoML model for stock return classification using FLAML.

Predicts whether a stock will achieve >=30% peak return at any point
within a 3-month window.  FLAML (Fast Lightweight AutoML) automatically
selects the best model and hyperparameters from XGBoost, LightGBM, etc.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML, tune
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction / derived features from existing columns.

    These features combine multiple raw signals into higher-level
    indicators that capture multi-factor breakout patterns.
    """
    # Earnings momentum: quarter-over-quarter EPS change
    if "hist_earnings_growth_qoq" in df.columns:
        df["Earnings_Momentum"] = df["hist_earnings_growth_qoq"]
    else:
        df["Earnings_Momentum"] = 0.0

    # Fundamental surprise: companies beating estimates while growing
    if "hist_revenue_growth_qoq" in df.columns and "earnings_surprise_pct" in df.columns:
        df["Fundamental_Surprise"] = (
            df["hist_revenue_growth_qoq"] * df["earnings_surprise_pct"]
        )
    else:
        df["Fundamental_Surprise"] = 0.0

    # Excess return vs market (stock alpha relative to S&P 500)
    if "Return_20d" in df.columns and "sp500_return_20d" in df.columns:
        df["Excess_Return_20d"] = df["Return_20d"] - df["sp500_return_20d"]
    else:
        df["Excess_Return_20d"] = 0.0

    if "Return_60d" in df.columns and "sp500_return_60d" in df.columns:
        df["Excess_Return_60d"] = df["Return_60d"] - df["sp500_return_60d"]
    else:
        df["Excess_Return_60d"] = 0.0

    return df


MODEL_DIR = Path(__file__).parent / "saved"
MODEL_PATH = MODEL_DIR / "stock_predictor_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MEDIANS_PATH = MODEL_DIR / "feature_medians.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
CLIP_BOUNDS_PATH = MODEL_DIR / "feature_clip_bounds.pkl"


# Classification threshold: predict class 1 when the stock achieves
# >=30% peak return at any point within the 3-month forward window.
CLASSIFICATION_THRESHOLD = 0.30


class StockReturnPredictor:
    """AutoML-based stock return classifier.

    Predicts class 1 (>=30% peak return within 3 months) vs class 0.
    Uses class weights to handle class imbalance and AUC as metric.
    """

    def __init__(self) -> None:
        self.automl = AutoML()
        self.feature_names: list[str] = []
        self.feature_medians: pd.Series | None = None
        self.scaler: StandardScaler | None = None
        self.clip_lower: pd.Series | None = None
        self.clip_upper: pd.Series | None = None
        self.is_trained = False

    def train(
        self,
        tickers: list[str] | None = None,
        time_budget: int = 120,
        include_sentiment: bool = True,
    ) -> dict:
        """Train the AutoML model on historical stock data.

        Args:
            tickers: List of tickers for training data. Defaults to top NASDAQ.
            time_budget: Time budget in seconds for AutoML search.
            include_sentiment: Whether to include sentiment features.

        Returns:
            Dictionary with training metrics.
        """
        if tickers is None:
            tickers = NASDAQ_TOP_TICKERS[:30]

        logger.info("Building training dataset for %d tickers...", len(tickers))
        df = build_training_dataset(tickers, include_sentiment=include_sentiment)

        if df.empty:
            raise ValueError("Training dataset is empty — no valid data collected.")

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
        # 1 = return >= 30%, 0 = otherwise
        y_binary = (y >= CLASSIFICATION_THRESHOLD).astype(int)
        n_pos = int(y_binary.sum())
        n_neg = len(y_binary) - n_pos
        logger.info(
            "Class balance: %d positive (%.1f%%) / %d negative (%.1f%%)",
            n_pos, n_pos / len(y_binary) * 100,
            n_neg, n_neg / len(y_binary) * 100,
        )
        y = y_binary

        # Compute class weights for imbalanced data
        weight_neg = 1.0
        weight_pos = n_neg / max(n_pos, 1)
        sample_weight = y.map({0: weight_neg, 1: weight_pos}).values

        # Semantically fill profit-related NaN values: these are NaN
        # because the company is unprofitable, not because data is
        # missing.  Using meaningful defaults preserves this signal.
        X = _fill_semantic_nan(X)

        # Log-transform raw dollar features to compress scale differences
        X = _log_transform(X)

        # Fill remaining NaN features with median and save medians for prediction
        self.feature_medians = X.median()
        X = X.fillna(self.feature_medians)

        # Winsorize features at 1st/99th percentiles to clip extreme
        # outliers while preserving the overall distribution shape.
        # Bounds are saved for consistent clipping at prediction time.
        self.clip_lower = X.quantile(0.01)
        self.clip_upper = X.quantile(0.99)
        X = X.clip(lower=self.clip_lower, upper=self.clip_upper, axis=1)

        # --- Temporal train/test split to avoid data leakage ---
        # Sort by date so split is chronological, not random
        if "_date" in df.columns:
            date_series = pd.to_datetime(df["_date"], errors="coerce")
            sort_order = date_series.sort_values().index
            X = X.loc[sort_order]
            y = y.loc[sort_order]

        # Hold out the last 20% as a test set with a 63-day gap
        # (gap prevents overlapping forward-return windows from leaking)
        gap_rows = max(1, int(len(X) * 0.05))  # ~5% gap
        split_idx = int(len(X) * 0.75)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx + gap_rows:]
        y_test = y.iloc[split_idx + gap_rows:]

        # Standardize features so no single feature dominates by scale
        self.scaler = StandardScaler()
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=feature_cols,
            index=X_test.index,
        )

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
        sw_train = sample_weight[:split_idx]
        sw_test = sample_weight[split_idx + gap_rows:len(sample_weight)]

        self.automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            time_budget=time_budget,
            metric="roc_auc",
            estimator_list=["xgboost", "lgbm"],
            eval_method="cv",
            n_splits=5,
            verbose=0,
            custom_hp=custom_hp,
            early_stop=True,
            sample_weight=sw_train,
        )

        self.is_trained = True

        # --- Out-of-sample metrics on held-out test set ---
        y_pred_test = self.automl.predict(X_test)
        y_proba_test = self.automl.predict_proba(X_test)
        # Use probability of class 1
        if y_proba_test.ndim == 2:
            y_proba_pos = y_proba_test[:, 1]
        else:
            y_proba_pos = y_proba_test

        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba_pos)
        except ValueError:
            auc = float("nan")

        # Training-set metrics for comparison (to detect overfitting)
        y_pred_train = self.automl.predict(X_train)
        y_proba_train = self.automl.predict_proba(X_train)
        if y_proba_train.ndim == 2:
            y_proba_train_pos = y_proba_train[:, 1]
        else:
            y_proba_train_pos = y_proba_train
        try:
            auc_train = roc_auc_score(y_train, y_proba_train_pos)
        except ValueError:
            auc_train = float("nan")
        accuracy_train = accuracy_score(y_train, y_pred_train)

        # Logistic Regression baseline for comparison
        lr = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        )
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_proba = lr.predict_proba(X_test)[:, 1]
        lr_accuracy = accuracy_score(y_test, lr_pred)
        try:
            lr_auc = roc_auc_score(y_test, lr_proba)
        except ValueError:
            lr_auc = float("nan")

        metrics = {
            "best_estimator": self.automl.best_estimator,
            "best_config": self.automl.best_config,
            "best_loss": self.automl.best_loss,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_cols),
            "class_balance": f"{n_pos}/{n_neg} ({n_pos/(n_pos+n_neg)*100:.1f}%)",
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
            "accuracy_train": round(accuracy_train, 4),
            "auc_train": round(auc_train, 4),
            "lr_accuracy": round(lr_accuracy, 4),
            "lr_auc": round(lr_auc, 4),
        }

        logger.info(
            "Training complete. Best: %s | Test AUC=%.4f | Train AUC=%.4f | "
            "Accuracy=%.4f | F1=%.4f | LR AUC=%.4f",
            self.automl.best_estimator, auc, auc_train, accuracy, f1, lr_auc,
        )
        self.save()
        return metrics

    def predict(self, features: dict | pd.DataFrame) -> float:
        """Predict probability of >=30% peak return within 3 months.

        Args:
            features: Feature dict or DataFrame row.

        Returns:
            Probability of class 1 (>=30% peak return) as a float [0, 1].
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

        if self.clip_lower is not None and self.clip_upper is not None:
            df = df.clip(lower=self.clip_lower, upper=self.clip_upper, axis=1)

        if self.scaler is not None:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=self.feature_names,
                index=df.index,
            )

        proba = self.automl.predict_proba(df)
        if proba.ndim == 2:
            return float(proba[0, 1])
        return float(proba[0])

    def predict_ticker(self, ticker: str) -> dict:
        """Predict whether a ticker will hit >=30% peak return within 3 months.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with ticker, probability, and classification.
        """
        row = build_training_row(ticker, include_sentiment=True)
        if row is None:
            return {
                "ticker": ticker,
                "probability_30pct_gain": None,
                "prediction": None,
                "error": "Could not build features for ticker.",
            }

        probability = self.predict(row)
        prediction = 1 if probability >= 0.5 else 0
        return {
            "ticker": ticker,
            "probability_30pct_gain": round(probability, 4),
            "probability_pct": f"{probability * 100:.1f}%",
            "prediction": prediction,
            "signal": "BUY" if prediction == 1 else "HOLD",
        }

    def save(self) -> None:
        """Save model and feature names to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.automl, MODEL_PATH)
        joblib.dump(self.feature_names, FEATURE_NAMES_PATH)
        joblib.dump(self.feature_medians, MEDIANS_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump((self.clip_lower, self.clip_upper), CLIP_BOUNDS_PATH)
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
        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
        if CLIP_BOUNDS_PATH.exists():
            self.clip_lower, self.clip_upper = joblib.load(CLIP_BOUNDS_PATH)
        self.is_trained = True
        logger.info("Model loaded from %s", MODEL_PATH)

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
            pairs = list(zip(self.feature_names, importances))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_n]
        return []
