"""AutoML model for stock return prediction using FLAML.

FLAML (Fast Lightweight AutoML) automatically selects the best model
and hyperparameters from XGBoost, LightGBM, Random Forest, etc.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


MODEL_DIR = Path(__file__).parent / "saved"
MODEL_PATH = MODEL_DIR / "stock_predictor_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MEDIANS_PATH = MODEL_DIR / "feature_medians.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"


class StockReturnPredictor:
    """AutoML-based stock return predictor."""

    def __init__(self) -> None:
        self.automl = AutoML()
        self.feature_names: list[str] = []
        self.feature_medians: pd.Series | None = None
        self.scaler: StandardScaler | None = None
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

        # Prepare features and target
        feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
        self.feature_names = feature_cols

        X = df[feature_cols].copy()
        y = df[TARGET_COLUMN].copy()

        # Drop rows with missing target
        valid = y.notna()
        X = X[valid]
        y = y[valid]

        # Semantically fill profit-related NaN values: these are NaN
        # because the company is unprofitable, not because data is
        # missing.  Using meaningful defaults preserves this signal.
        X = _fill_semantic_nan(X)

        # Fill remaining NaN features with median and save medians for prediction
        self.feature_medians = X.median()
        X = X.fillna(self.feature_medians)

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

        self.automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            time_budget=time_budget,
            metric="r2",
            estimator_list=["xgboost", "lgbm", "rf", "extra_tree"],
            eval_method="cv",
            n_splits=5,
            verbose=0,
        )

        self.is_trained = True

        # --- Out-of-sample metrics on held-out test set ---
        y_pred_test = self.automl.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        nonzero_mask = y_test.abs() > 1e-8
        if nonzero_mask.sum() > 0:
            mape = float(
                ((y_test[nonzero_mask] - y_pred_test[nonzero_mask]).abs()
                 / y_test[nonzero_mask].abs()).mean() * 100
            )
        else:
            mape = float("nan")

        # Training-set metrics for comparison (to detect overfitting)
        y_pred_train = self.automl.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)

        metrics = {
            "best_estimator": self.automl.best_estimator,
            "best_config": self.automl.best_config,
            "best_loss": self.automl.best_loss,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_cols),
            "r2_score": round(r2, 4),
            "r2_train": round(r2_train, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
        }

        logger.info(
            "Training complete. Best: %s | Test R²=%.4f | Train R²=%.4f | "
            "MAE=%.4f | RMSE=%.4f",
            self.automl.best_estimator, r2, r2_train, mae, rmse,
        )
        self.save()
        return metrics

    def predict(self, features: dict | pd.DataFrame) -> float:
        """Predict 3-month forward return.

        Args:
            features: Feature dict or DataFrame row.

        Returns:
            Predicted return as a float (e.g. 0.25 means +25%).
        """
        if not self.is_trained:
            self.load()

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        # Ensure columns match training features
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names]
        df = _fill_semantic_nan(df)
        if self.feature_medians is not None:
            df = df.fillna(self.feature_medians)
        else:
            df = df.fillna(0.0)

        if self.scaler is not None:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=self.feature_names,
                index=df.index,
            )

        prediction = self.automl.predict(df)
        return float(prediction[0])

    def predict_ticker(self, ticker: str) -> dict:
        """Predict 3-month return for a given ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with ticker, predicted return, and confidence info.
        """
        row = build_training_row(ticker, include_sentiment=True)
        if row is None:
            return {
                "ticker": ticker,
                "predicted_return_3m": None,
                "error": "Could not build features for ticker.",
            }

        predicted_return = self.predict(row)
        return {
            "ticker": ticker,
            "predicted_return_3m": round(predicted_return, 4),
            "predicted_return_3m_pct": f"{predicted_return * 100:.2f}%",
        }

    def save(self) -> None:
        """Save model and feature names to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.automl, MODEL_PATH)
        joblib.dump(self.feature_names, FEATURE_NAMES_PATH)
        joblib.dump(self.feature_medians, MEDIANS_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
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
