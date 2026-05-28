"""Integration / system tests for the stock predictor pipeline.

These tests verify that the components work together correctly using
mocked external APIs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from stock_predictor.data.feature_engineering import (
    ALL_FEATURE_NAMES,
    TARGET_COLUMN,
    TECHNICAL_FEATURES,
    build_training_row,
)
from stock_predictor.models.automl_model import StockReturnPredictor


def _make_price_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - np.random.rand(n) * 0.5,
            "High": close + np.random.rand(n) * 1.0,
            "Low": close - np.random.rand(n) * 1.0,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }
    )


class TestEndToEndPipeline:
    """Test the full pipeline: data collection -> feature engineering -> prediction."""

    @patch("stock_predictor.data.feature_engineering.get_stock_data")
    def test_build_training_row(self, mock_data):
        mock_data.return_value = _make_price_df(300)

        row = build_training_row("AAPL")
        assert row is not None
        assert row["Ticker"] == "AAPL"

        # Check technical features are present
        for col in TECHNICAL_FEATURES:
            assert col in row, f"Missing technical feature: {col}"

        # Snapshot fundamentals and sentiment should NOT be in the row
        assert "marketCap" not in row
        assert "sentiment_mean_polarity" not in row

    @patch("stock_predictor.data.feature_engineering.get_stock_data")
    def test_full_train_predict_cycle(self, mock_data):
        """Test training a model and making predictions end-to-end."""
        mock_data.return_value = _make_price_df(300)

        # Build a row
        row = build_training_row("TEST")
        assert row is not None

        # Create synthetic training data
        np.random.seed(42)
        rows = []
        for i in range(100):
            r = {col: np.random.randn() for col in ALL_FEATURE_NAMES}
            r["Ticker"] = "TEST"
            r[TARGET_COLUMN] = 0.3 * r.get("RSI_14", 0) + np.random.randn() * 0.1
            rows.append(r)
        df = pd.DataFrame(rows)

        # Train
        predictor = StockReturnPredictor()
        with patch.object(predictor, "save"):
            with patch(
                "stock_predictor.models.automl_model.build_training_dataset",
                return_value=df,
            ):
                metrics = predictor.train(tickers=["TEST"], time_budget=10)

        assert predictor.is_trained
        assert metrics["training_samples"] > 0

        # Predict
        prediction = predictor.predict(row)
        assert isinstance(prediction, float)
        assert np.isfinite(prediction)

    def test_tools_return_valid_json(self):
        """Verify that YFinance tool output is valid JSON."""
        from stock_predictor.agent.tools import yfinance_tool

        with patch("stock_predictor.agent.tools.get_stock_info") as mock_info, \
             patch("stock_predictor.agent.tools.get_fundamentals_features") as mock_fund, \
             patch("stock_predictor.agent.tools.get_stock_data") as mock_data, \
             patch("stock_predictor.agent.tools.compute_technical_features") as mock_tech:

            mock_info.return_value = {"shortName": "Test Corp"}
            mock_fund.return_value = {"marketCap": 1e9}

            df = pd.DataFrame({
                "Close": [100.0], "Volume": [1000000],
                "RSI_14": [50.0], "MACD": [0.5], "MACD_Signal": [0.4],
                "BB_Position": [0.5], "BB_Width": [0.04],
                "Volatility_20d": [0.02], "Volume_Ratio": [1.0],
                "Price_to_SMA_20": [1.0], "Price_to_SMA_50": [1.0],
                "Price_to_SMA_200": [1.0], "ATR_14": [2.0],
                "Return_1d": [0.01], "Return_5d": [0.02],
                "Return_20d": [0.05], "Return_60d": [0.10],
            })
            mock_data.return_value = df
            mock_tech.return_value = df

            result = yfinance_tool.invoke({"ticker": "TEST"})
            parsed = json.loads(result)
            assert "ticker" in parsed

    def test_sentiment_tool_returns_string(self):
        """Verify the social media listener tool returns a formatted string."""
        from stock_predictor.agent.tools import social_media_listener_tool

        with patch("stock_predictor.agent.tools.get_sentiment_summary") as mock_sum, \
             patch("stock_predictor.agent.tools.get_sentiment_features") as mock_feat, \
             patch("stock_predictor.agent.tools.get_trending_tickers_from_social") as mock_trend:

            mock_sum.return_value = "Positive sentiment"
            mock_feat.return_value = {"sentiment_mean_polarity": 0.3}
            mock_trend.return_value = ["AAPL"]

            result = social_media_listener_tool.invoke({"ticker": "AAPL"})
            assert isinstance(result, str)
            assert "Positive" in result
