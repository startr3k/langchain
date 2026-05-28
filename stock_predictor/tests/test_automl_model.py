"""Unit tests for the AutoML stock return prediction model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from stock_predictor.data.feature_engineering import (
    ALL_FEATURE_NAMES,
    FUNDAMENTAL_FEATURES,
    SENTIMENT_FEATURES,
    TARGET_COLUMN,
    TECHNICAL_FEATURES,
)
from stock_predictor.data.historical_fundamentals import HIST_FUNDAMENTAL_FEATURES
from stock_predictor.data.macro_data import MACRO_FEATURES
from stock_predictor.data.earnings_data import EARNINGS_FEATURES
from stock_predictor.data.google_trends import TRENDS_FEATURES
from stock_predictor.data.sec_edgar import SEC_FEATURES
from stock_predictor.models.automl_model import StockReturnPredictor


def _make_training_data(n_rows: int = 200) -> pd.DataFrame:
    """Create synthetic training data for testing."""
    np.random.seed(42)
    data = {"Ticker": ["TEST"] * n_rows}

    for col in TECHNICAL_FEATURES:
        data[col] = np.random.randn(n_rows)
    for col in FUNDAMENTAL_FEATURES:
        data[col] = np.random.rand(n_rows) * 100
    for col in SENTIMENT_FEATURES:
        data[col] = np.random.rand(n_rows)

    # Target: noisy function of a few features
    data[TARGET_COLUMN] = (
        0.5 * data["Return_60d"]
        + 0.3 * data["RSI_14"]
        + 0.2 * data["sentiment_mean_polarity"]
        + np.random.randn(n_rows) * 0.1
    )

    return pd.DataFrame(data)


class TestStockReturnPredictor:
    def test_train_and_predict(self):
        predictor = StockReturnPredictor()
        df = _make_training_data()

        with patch.object(predictor, "save"):  # Don't save to disk in test
            with patch(
                "stock_predictor.models.automl_model.build_training_dataset",
                return_value=df,
            ):
                metrics = predictor.train(
                    tickers=["TEST"], time_budget=10, include_sentiment=True
                )

        assert "best_estimator" in metrics
        assert metrics["training_samples"] > 0
        assert predictor.is_trained

        # Test prediction
        test_features = {col: 0.5 for col in ALL_FEATURE_NAMES}
        prediction = predictor.predict(test_features)
        assert isinstance(prediction, float)
        assert np.isfinite(prediction)

    def test_predict_with_dataframe(self):
        predictor = StockReturnPredictor()
        df = _make_training_data()

        with patch.object(predictor, "save"):
            with patch(
                "stock_predictor.models.automl_model.build_training_dataset",
                return_value=df,
            ):
                predictor.train(tickers=["TEST"], time_budget=10)

        test_df = pd.DataFrame([{col: 0.5 for col in ALL_FEATURE_NAMES}])
        prediction = predictor.predict(test_df)
        assert isinstance(prediction, float)

    def test_predict_without_training_raises(self, tmp_path):
        predictor = StockReturnPredictor()
        with patch(
            "stock_predictor.models.automl_model.MODEL_PATH",
            tmp_path / "nonexistent_model.pkl",
        ):
            with pytest.raises(FileNotFoundError):
                predictor.predict({"feature": 1.0})

    def test_save_and_load(self, tmp_path):
        predictor = StockReturnPredictor()
        df = _make_training_data()

        with patch(
            "stock_predictor.models.automl_model.MODEL_DIR", tmp_path
        ), patch(
            "stock_predictor.models.automl_model.MODEL_PATH",
            tmp_path / "model.pkl",
        ), patch(
            "stock_predictor.models.automl_model.FEATURE_NAMES_PATH",
            tmp_path / "features.pkl",
        ), patch(
            "stock_predictor.models.automl_model.MEDIANS_PATH",
            tmp_path / "medians.pkl",
        ), patch(
            "stock_predictor.models.automl_model.build_training_dataset",
            return_value=df,
        ):
            predictor.train(tickers=["TEST"], time_budget=10)

            # Load in a new predictor
            predictor2 = StockReturnPredictor()
            predictor2.load()
            assert predictor2.is_trained
            assert predictor2.feature_names == predictor.feature_names
            assert predictor2.feature_medians is not None

    def test_predict_ticker(self):
        predictor = StockReturnPredictor()
        df = _make_training_data()

        with patch.object(predictor, "save"):
            with patch(
                "stock_predictor.models.automl_model.build_training_dataset",
                return_value=df,
            ):
                predictor.train(tickers=["TEST"], time_budget=10)

        mock_row = {col: 0.5 for col in ALL_FEATURE_NAMES}
        mock_row["Ticker"] = "AAPL"

        with patch(
            "stock_predictor.models.automl_model.build_training_row",
            return_value=mock_row,
        ):
            result = predictor.predict_ticker("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["predicted_return_3m"] is not None
        assert "predicted_return_3m_pct" in result

    def test_predict_ticker_no_data(self):
        predictor = StockReturnPredictor()
        predictor.is_trained = True
        predictor.feature_names = ALL_FEATURE_NAMES

        with patch(
            "stock_predictor.models.automl_model.build_training_row",
            return_value=None,
        ):
            result = predictor.predict_ticker("INVALID")

        assert result["predicted_return_3m"] is None
        assert "error" in result

    def test_get_feature_importance(self):
        predictor = StockReturnPredictor()
        df = _make_training_data()

        with patch.object(predictor, "save"):
            with patch(
                "stock_predictor.models.automl_model.build_training_dataset",
                return_value=df,
            ):
                predictor.train(tickers=["TEST"], time_budget=10)

        importance = predictor.get_feature_importance(top_n=5)
        # May or may not have importances depending on estimator type
        assert isinstance(importance, list)

    def test_empty_training_data_raises(self):
        predictor = StockReturnPredictor()

        with patch.object(predictor, "save"):
            with patch(
                "stock_predictor.models.automl_model.build_training_dataset",
                return_value=pd.DataFrame(),
            ):
                with pytest.raises(ValueError, match="empty"):
                    predictor.train(tickers=["TEST"], time_budget=10)


class TestFeatureEngineering:
    def test_all_feature_names_complete(self):
        assert len(TECHNICAL_FEATURES) > 10
        assert len(FUNDAMENTAL_FEATURES) > 5
        assert len(SENTIMENT_FEATURES) > 5
        assert len(HIST_FUNDAMENTAL_FEATURES) > 5
        assert len(MACRO_FEATURES) > 5
        assert len(EARNINGS_FEATURES) > 0
        assert len(TRENDS_FEATURES) > 0
        assert len(SEC_FEATURES) > 0
        # Google Trends excluded from ALL_FEATURE_NAMES (rate-limited)
        expected_total = (
            len(TECHNICAL_FEATURES)
            + len(FUNDAMENTAL_FEATURES)
            + len(HIST_FUNDAMENTAL_FEATURES)
            + len(MACRO_FEATURES)
            + len(EARNINGS_FEATURES)
            + len(SEC_FEATURES)
            + len(SENTIMENT_FEATURES)
        )
        assert len(ALL_FEATURE_NAMES) == expected_total

    def test_target_column_defined(self):
        assert TARGET_COLUMN == "Forward_Return_3M"
