"""YFinance data client for fetching stock market data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

NASDAQ_TOP_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "ADBE", "PEP", "CSCO", "TMUS", "INTC",
    "INTU", "CMCSA", "TXN", "QCOM", "AMGN", "AMAT", "ISRG", "HON",
    "BKNG", "LRCX", "SBUX", "VRTX", "MU", "ADI", "GILD", "MDLZ",
    "PANW", "REGN", "KLAC", "SNPS", "CDNS", "PYPL", "MELI", "CRWD",
    "MAR", "CTAS", "ABNB", "ORLY", "MRVL", "FTNT", "CEG", "DASH",
    "WDAY", "MNST",
]


def get_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical stock data from YFinance.

    Args:
        ticker: Stock ticker symbol.
        period: Data period (e.g. '1y', '2y', '5y').
        interval: Data interval (e.g. '1d', '1wk').

    Returns:
        DataFrame with OHLCV data and computed features.
    """
    logger.info("Fetching stock data for %s (period=%s)", ticker, period)
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        logger.warning("No data returned for %s", ticker)
        return df
    df = df.reset_index()
    df["Ticker"] = ticker
    return df


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features to a price DataFrame.

    Args:
        df: DataFrame with at least 'Close' and 'Volume' columns.

    Returns:
        DataFrame augmented with technical features.
    """
    if df.empty:
        return df

    df = df.copy()

    # Returns
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_20d"] = df["Close"].pct_change(20)
    df["Return_60d"] = df["Close"].pct_change(60)

    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

    # Relative position to moving averages
    df["Price_to_SMA_20"] = df["Close"] / df["SMA_20"]
    df["Price_to_SMA_50"] = df["Close"] / df["SMA_50"]
    df["Price_to_SMA_200"] = df["Close"] / df["SMA_200"]

    # Volatility
    df["Volatility_20d"] = df["Return_1d"].rolling(window=20).std()
    df["Volatility_60d"] = df["Return_1d"].rolling(window=60).std()

    # Volume features
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    bb_sma = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = bb_sma + 2 * bb_std
    df["BB_Lower"] = bb_sma - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_sma
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
        df["BB_Upper"] - df["BB_Lower"]
    ).replace(0, np.nan)

    # Average True Range (ATR)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14).mean()

    # On-Balance Volume (OBV)
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv
    df["OBV_SMA_20"] = obv.rolling(window=20).mean()

    # Volume spike detection
    vol_mean = df["Volume"].rolling(window=20).mean()
    vol_std = df["Volume"].rolling(window=20).std()
    df["Volume_Spike"] = (df["Volume"] > vol_mean * 1.5).astype(float)
    df["Volume_Spike_Magnitude"] = (
        (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
    )

    return df


def get_stock_info(ticker: str) -> dict:
    """Fetch company info / fundamentals from YFinance.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary of company fundamental data.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    keys_of_interest = [
        "shortName", "sector", "industry", "marketCap", "trailingPE",
        "forwardPE", "priceToBook", "dividendYield", "beta",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "averageVolume",
        "revenueGrowth", "earningsGrowth", "profitMargins",
        "returnOnEquity", "debtToEquity", "currentRatio",
        "freeCashflow", "totalRevenue", "targetMeanPrice",
        "recommendationKey", "numberOfAnalystOpinions",
    ]
    return {k: info.get(k) for k in keys_of_interest if info.get(k) is not None}


def get_fundamentals_features(ticker: str) -> dict:
    """Extract numerical fundamental features for model input.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary of fundamental feature values.
    """
    info = get_stock_info(ticker)
    feature_keys = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook",
        "dividendYield", "beta", "revenueGrowth", "earningsGrowth",
        "profitMargins", "returnOnEquity", "debtToEquity",
        "currentRatio", "numberOfAnalystOpinions",
    ]
    features = {}
    for k in feature_keys:
        val = info.get(k)
        if val is not None:
            try:
                features[k] = float(val)
            except (ValueError, TypeError):
                pass
    return features


def get_nasdaq_trending_tickers(top_n: int = 20) -> list[str]:
    """Return the top NASDAQ tickers by recent volume activity.

    Args:
        top_n: Number of tickers to return.

    Returns:
        List of ticker symbols.
    """
    results = []
    for ticker in NASDAQ_TOP_TICKERS[:top_n]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            results.append(
                {
                    "ticker": ticker,
                    "name": info.get("shortName", ticker),
                    "market_cap": info.get("marketCap", 0),
                    "volume": info.get("averageVolume", 0),
                }
            )
        except Exception:
            logger.warning("Failed to fetch info for %s", ticker)
    results.sort(key=lambda x: x["volume"], reverse=True)
    return [r["ticker"] for r in results[:top_n]]
