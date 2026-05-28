"""Unit tests for the YFinance data client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from stock_predictor.data.yfinance_client import (
    NASDAQ_TOP_TICKERS,
    compute_technical_features,
    get_fundamentals_features,
    get_stock_data,
    get_stock_info,
)


def _make_price_df(n: int = 300) -> pd.DataFrame:
    """Create a synthetic price DataFrame for testing."""
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


class TestComputeTechnicalFeatures:
    def test_returns_expected_columns(self):
        df = _make_price_df()
        result = compute_technical_features(df)
        expected_cols = [
            "Return_1d", "Return_5d", "Return_20d", "Return_60d",
            "SMA_5", "SMA_20", "SMA_50", "SMA_200",
            "EMA_5", "EMA_20",
            "Price_to_SMA_20", "Price_to_SMA_50", "Price_to_SMA_200",
            "Volatility_20d", "Volatility_60d",
            "Volume_Ratio",
            "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Lower", "BB_Width", "BB_Position",
            "ATR_14", "OBV", "OBV_SMA_20",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_range(self):
        df = _make_price_df()
        result = compute_technical_features(df)
        rsi = result["RSI_14"].dropna()
        assert rsi.min() >= 0, "RSI should be >= 0"
        assert rsi.max() <= 100, "RSI should be <= 100"

    def test_sma_values(self):
        df = _make_price_df()
        result = compute_technical_features(df)
        # SMA_5 at row 10 should be mean of rows 6-10
        sma5_at_10 = result["SMA_5"].iloc[10]
        expected = df["Close"].iloc[6:11].mean()
        assert abs(sma5_at_10 - expected) < 0.01

    def test_empty_df(self):
        df = pd.DataFrame()
        result = compute_technical_features(df)
        assert result.empty

    def test_volume_ratio(self):
        df = _make_price_df()
        result = compute_technical_features(df)
        vol_ratio = result["Volume_Ratio"].dropna()
        assert not vol_ratio.empty
        # Volume ratio should be positive
        assert (vol_ratio > 0).all()


class TestGetStockData:
    @patch("stock_predictor.data.yfinance_client.yf.Ticker")
    def test_returns_dataframe_with_ticker(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_price_df().set_index("Date")
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_data("AAPL", period="1y")
        assert "Ticker" in result.columns
        assert result["Ticker"].iloc[0] == "AAPL"
        assert not result.empty

    @patch("stock_predictor.data.yfinance_client.yf.Ticker")
    def test_returns_empty_for_bad_ticker(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_data("XXXXX")
        assert result.empty


class TestGetStockInfo:
    @patch("stock_predictor.data.yfinance_client.yf.Ticker")
    def test_returns_dict_with_known_keys(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "shortName": "Apple Inc.",
            "sector": "Technology",
            "marketCap": 3_000_000_000_000,
            "trailingPE": 30.5,
            "beta": 1.2,
        }
        mock_ticker_cls.return_value = mock_ticker

        result = get_stock_info("AAPL")
        assert result["shortName"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["marketCap"] == 3_000_000_000_000


class TestGetFundamentalsFeatures:
    @patch("stock_predictor.data.yfinance_client.get_stock_info")
    def test_returns_float_values(self, mock_info):
        mock_info.return_value = {
            "marketCap": 3_000_000_000_000,
            "trailingPE": 30.5,
            "forwardPE": 28.0,
            "beta": 1.2,
            "revenueGrowth": 0.15,
        }
        result = get_fundamentals_features("AAPL")
        assert isinstance(result["marketCap"], float)
        assert isinstance(result["trailingPE"], float)
        assert result["beta"] == 1.2


class TestNasdaqTickers:
    def test_list_not_empty(self):
        assert len(NASDAQ_TOP_TICKERS) >= 20

    def test_known_tickers_present(self):
        for ticker in ["AAPL", "MSFT", "NVDA", "TSLA"]:
            assert ticker in NASDAQ_TOP_TICKERS
