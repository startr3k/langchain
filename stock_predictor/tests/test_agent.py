"""Unit tests for the LLM agent and tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from stock_predictor.agent.tools import (
    social_media_listener_tool,
    stock_predictor_tool,
    yfinance_tool,
)


class TestYFinanceTool:
    @patch("stock_predictor.agent.tools.get_stock_info")
    @patch("stock_predictor.agent.tools.get_fundamentals_features")
    @patch("stock_predictor.agent.tools.get_stock_data")
    @patch("stock_predictor.agent.tools.compute_technical_features")
    def test_returns_json(self, mock_tech, mock_data, mock_fund, mock_info):
        import numpy as np
        import pandas as pd

        mock_info.return_value = {"shortName": "NVIDIA Corp", "sector": "Technology"}
        mock_fund.return_value = {"marketCap": 3e12, "trailingPE": 60.0}

        df = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0],
                "Volume": [1000000, 1100000, 1200000],
                "RSI_14": [55.0, 56.0, 57.0],
                "MACD": [1.0, 1.1, 1.2],
                "MACD_Signal": [0.9, 1.0, 1.1],
                "BB_Position": [0.6, 0.65, 0.7],
                "BB_Width": [0.05, 0.04, 0.06],
                "Volatility_20d": [0.02, 0.021, 0.019],
                "Volume_Ratio": [1.1, 1.2, 0.9],
                "Price_to_SMA_20": [1.01, 1.02, 1.03],
                "Price_to_SMA_50": [1.05, 1.06, 1.07],
                "Price_to_SMA_200": [1.15, 1.16, 1.17],
                "ATR_14": [2.5, 2.6, 2.7],
                "Return_1d": [0.01, 0.01, 0.01],
                "Return_5d": [0.03, 0.03, 0.03],
                "Return_20d": [0.08, 0.08, 0.08],
                "Return_60d": [0.15, 0.15, 0.15],
            }
        )
        mock_data.return_value = df
        mock_tech.return_value = df

        result = yfinance_tool.invoke({"ticker": "NVDA"})
        parsed = json.loads(result)
        assert parsed["ticker"] == "NVDA"
        assert "company_info" in parsed
        assert "latest_price" in parsed
        assert "technical_indicators" in parsed
        assert "fundamentals" in parsed


class TestSocialMediaListenerTool:
    @patch("stock_predictor.agent.tools.get_trending_tickers_from_social")
    @patch("stock_predictor.agent.tools.get_sentiment_features")
    @patch("stock_predictor.agent.tools.get_sentiment_summary")
    def test_returns_summary(self, mock_summary, mock_features, mock_trending):
        mock_summary.return_value = "=== Sentiment for TSLA ===\nPositive"
        mock_features.return_value = {"sentiment_mean_polarity": 0.25}
        mock_trending.return_value = ["TSLA", "NVDA"]

        result = social_media_listener_tool.invoke({"ticker": "TSLA"})
        assert "TSLA" in result
        assert "sentiment_mean_polarity" in result


class TestStockPredictorTool:
    @patch("stock_predictor.agent.tools._get_predictor")
    def test_returns_prediction(self, mock_get_pred):
        mock_predictor = MagicMock()
        mock_predictor.is_trained = True
        mock_predictor.predict_ticker.return_value = {
            "ticker": "AAPL",
            "probability_gain": 0.72,
            "probability_pct": "72.0%",
            "prediction": 1,
            "signal": "BUY",
        }
        mock_predictor.get_feature_importance.return_value = [
            ("RSI_14", 0.15),
            ("Return_60d", 0.12),
        ]
        mock_get_pred.return_value = mock_predictor

        result = stock_predictor_tool.invoke({"ticker": "AAPL"})
        parsed = json.loads(result)
        assert parsed["ticker"] == "AAPL"
        assert parsed["probability_gain"] == 0.72

    @patch("stock_predictor.agent.tools._get_predictor")
    def test_untrained_model_error(self, mock_get_pred):
        mock_predictor = MagicMock()
        mock_predictor.is_trained = False
        mock_get_pred.return_value = mock_predictor

        result = stock_predictor_tool.invoke({"ticker": "AAPL"})
        parsed = json.loads(result)
        assert "error" in parsed


class TestAgentCreation:
    @patch("stock_predictor.agent.agent.ChatOpenAI")
    def test_create_agent(self, mock_chat):
        from stock_predictor.agent.agent import create_agent

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        llm, tools, sys_msg = create_agent(api_key="test-key")
        assert len(tools) == 4
        assert sys_msg.content  # System prompt not empty

    def test_create_agent_no_key_raises(self):
        from stock_predictor.agent.agent import create_agent

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                create_agent(api_key=None)
