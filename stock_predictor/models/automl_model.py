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
from sklearn.model_selection import TimeSeriesSplit

from stock_predictor.data.feature_engineering import (
    ALL_FEATURE_NAMES,
    TARGET_COLUMN,
    build_training_dataset,
    build_training_row,
)
from stock_predictor.data.yfinance_client import NASDAQ_TOP_TICKERS

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_PATH = MODEL_DIR / "stock_predictor_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"


class StockReturnPredictor:
    """AutoML-based stock return predictor."""

    def __init__(self) -> None:
        self.automl = AutoML()
        self.feature_names: list[str] = []
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

        # Fill remaining NaN features with median
        X = X.fillna(X.median())

        logger.info(
            "Training AutoML on %d samples, %d features (budget=%ds)",
            len(X), len(feature_cols), time_budget,
        )

        self.automl.fit(
            X_train=X,
            y_train=y,
            task="regression",
            time_budget=time_budget,
            metric="r2",
            estimator_list=["xgboost", "lgbm", "rf", "extra_tree"],
            eval_method="cv",
            n_splits=5,
            verbose=0,
        )

        self.is_trained = True
        metrics = {
            "best_estimator": self.automl.best_estimator,
            "best_config": self.automl.best_config,
            "best_loss": self.automl.best_loss,
            "training_samples": len(X),
            "num_features": len(feature_cols),
        }

        logger.info("Training complete. Best estimator: %s", self.automl.best_estimator)
        self.save()
        return metrics

    def predict(self, features: dict | pd.DataFrame) -> float:
        """Predict 6-month forward return.

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
        df = df.fillna(0.0)

        prediction = self.automl.predict(df)
        return float(prediction[0])

    def predict_ticker(self, ticker: str) -> dict:
        """Predict 6-month return for a given ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with ticker, predicted return, and confidence info.
        """
        row = build_training_row(ticker, include_sentiment=True)
        if row is None:
            return {
                "ticker": ticker,
                "predicted_return_6m": None,
                "error": "Could not build features for ticker.",
            }

        predicted_return = self.predict(row)
        return {
            "ticker": ticker,
            "predicted_return_6m": round(predicted_return, 4),
            "predicted_return_6m_pct": f"{predicted_return * 100:.2f}%",
        }

    def save(self) -> None:
        """Save model and feature names to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.automl, MODEL_PATH)
        joblib.dump(self.feature_names, FEATURE_NAMES_PATH)
        logger.info("Model saved to %s", MODEL_PATH)

    def load(self) -> None:
        """Load model and feature names from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No saved model found at {MODEL_PATH}. Train the model first."
            )
        self.automl = joblib.load(MODEL_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
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
