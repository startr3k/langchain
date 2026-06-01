"""YFinance data client for fetching stock market data."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# Regex for valid ticker symbols (1-5 uppercase letters, optional dot/hyphen suffix)
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?(-[A-Z]{1,2})?$")


def fetch_all_nasdaq_tickers(
    min_market_cap: int = 500_000_000,
    cache_path: str | None = None,
) -> list[str]:
    """Fetch the full NASDAQ-listed ticker universe from the NASDAQ API.

    Args:
        min_market_cap: Minimum market cap in USD to include.  Defaults to
            $500M for institutional-grade liquidity.
        cache_path: If provided, write the market cap cache to this JSON file
            so future runs can resume without re-fetching.

    Returns:
        Sorted list of ticker symbols passing the market cap filter.
    """
    import json

    headers = {"User-Agent": "Mozilla/5.0 (StockPredictor/1.0)"}

    # ── Step 1: Get all NASDAQ-listed symbols from the NASDAQ screener API ──
    all_symbols: set[str] = set()
    try:
        resp = requests.get(
            "https://api.nasdaq.com/api/screener/stocks"
            "?tableType=traded&exchange=nasdaq&limit=10000",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()["data"]["table"]["rows"]
        for r in rows:
            sym = r["symbol"].strip()
            if _TICKER_RE.match(sym):
                all_symbols.add(sym)
        logger.info("NASDAQ API returned %d valid tickers", len(all_symbols))
    except Exception:
        logger.warning("NASDAQ API failed — falling back to Wikipedia + hardcoded list")
        all_symbols = set(NASDAQ_TOP_TICKERS)

    if not all_symbols:
        logger.warning("No tickers fetched — using NASDAQ_TOP_TICKERS fallback")
        all_symbols = set(NASDAQ_TOP_TICKERS)

    # ── Step 2: Filter by market cap via yFinance ──
    logger.info(
        "Filtering %d tickers by market cap >= $%dM...",
        len(all_symbols),
        min_market_cap // 1_000_000,
    )
    mcap_cache: dict[str, int | None] = {}

    # Load existing cache if available
    if cache_path:
        try:
            with open(cache_path) as f:
                mcap_cache = json.load(f)
            logger.info("Loaded %d entries from existing market cap cache", len(mcap_cache))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # Only fetch market cap for tickers not already in cache
    to_fetch = sorted(all_symbols - set(mcap_cache.keys()))
    logger.info("Need market cap for %d new tickers", len(to_fetch))

    for i, sym in enumerate(to_fetch):
        try:
            info = yf.Ticker(sym).info
            mcap_cache[sym] = info.get("marketCap", 0) or 0
        except Exception:
            mcap_cache[sym] = None
        if (i + 1) % 50 == 0:
            logger.info("Market cap progress: %d / %d", i + 1, len(to_fetch))
            time.sleep(1)  # rate limit
            # Checkpoint the cache
            if cache_path:
                try:
                    with open(cache_path, "w") as f:
                        json.dump(mcap_cache, f, indent=2)
                except Exception:
                    pass

    # Save final cache
    if cache_path:
        try:
            with open(cache_path, "w") as f:
                json.dump(mcap_cache, f, indent=2)
            logger.info("Saved market cap cache to %s", cache_path)
        except Exception:
            logger.warning("Could not save market cap cache")

    filtered = sorted(
        sym
        for sym in all_symbols
        if (mcap_cache.get(sym) or 0) >= min_market_cap
    )
    logger.info(
        "Filtered to %d tickers with market cap >= $%dM",
        len(filtered),
        min_market_cap // 1_000_000,
    )
    return filtered

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
    start: str | None = None,
) -> pd.DataFrame:
    """Fetch historical stock data from YFinance.

    Args:
        ticker: Stock ticker symbol.
        period: Data period (e.g. '1y', '2y', '5y'). Ignored when *start* is set.
        interval: Data interval (e.g. '1d', '1wk').
        start: Optional start date (YYYY-MM-DD).  When provided, ``period``
            is ignored and data is fetched from *start* to today.

    Returns:
        DataFrame with OHLCV data and computed features.
    """
    if start is not None:
        logger.info("Fetching stock data for %s (start=%s)", ticker, start)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, interval=interval)
    else:
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

    # SMA 200 cross signal: detect when price crosses the 200-day MA
    above_sma200 = (df["Close"] > df["SMA_200"]).astype(int)
    cross = above_sma200.diff()
    # +1 = bullish cross (price moved above), -1 = bearish cross (below)
    df["SMA_200_Cross"] = cross.fillna(0)
    # Days since last cross event
    cross_occurred = cross.abs() > 0
    cross_groups = cross_occurred.cumsum()
    df["Days_Since_SMA200_Cross"] = cross_groups.groupby(cross_groups).cumcount()

    # Volatility
    df["Volatility_20d"] = df["Return_1d"].rolling(window=20).std()
    df["Volatility_60d"] = df["Return_1d"].rolling(window=60).std()

    # Volume features
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]
    # 3-day volume surge: avg volume over last 3 days / 20-day avg
    df["Volume_Surge_3d"] = (
        df["Volume"].rolling(window=3).mean() / df["Volume_SMA_20"]
    )

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
    # Normalize MACD by price so it's comparable across stocks
    df["MACD"] = (ema_12 - ema_26) / df["Close"]
    macd_signal = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = df["MACD"] - macd_signal

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

    # --- Engineered breakout features ---

    # 1. Volatility contraction ratio: low short-term vol relative to
    #    long-term vol signals compression before a breakout.
    df["Volatility_Contraction"] = df["Volatility_20d"] / df["Volatility_60d"].replace(0, np.nan)

    # 2. Momentum acceleration: difference between short-term and
    #    medium-term returns — captures accelerating moves.
    df["Momentum_Accel"] = df["Return_5d"] - df["Return_20d"]

    # 3. Volume-price confirmation: positive returns on high volume
    #    signal conviction behind the move.
    df["Volume_Price_Confirm"] = df["Return_5d"] * df["Volume_Ratio"]

    # 4. Distance from 52-week high and low (normalised 0-1).
    high_252 = df["Close"].rolling(window=252, min_periods=63).max()
    low_252 = df["Close"].rolling(window=252, min_periods=63).min()
    df["Dist_52w_High"] = df["Close"] / high_252.replace(0, np.nan)
    df["Dist_52w_Low"] = df["Close"] / low_252.replace(0, np.nan)

    # 5. BB squeeze duration: consecutive days BB_Width is below its
    #    20-day average — longer squeezes precede bigger breakouts.
    bb_width_avg = df["BB_Width"].rolling(window=20).mean()
    squeeze = (df["BB_Width"] < bb_width_avg).astype(int)
    # Count consecutive squeeze days (resets on non-squeeze)
    squeeze_groups = (squeeze != squeeze.shift()).cumsum()
    df["BB_Squeeze_Duration"] = squeeze.groupby(squeeze_groups).cumcount()
    df.loc[squeeze == 0, "BB_Squeeze_Duration"] = 0

    # 6. RSI divergence: price makes new 14-day low but RSI doesn't.
    #    Positive value = bullish divergence (reversal signal).
    price_14d_low = df["Close"].rolling(window=14).min()
    rsi_14d_low = df["RSI_14"].rolling(window=14).min()
    price_at_new_low = df["Close"] <= price_14d_low * 1.001
    rsi_higher = df["RSI_14"] > rsi_14d_low + 2
    df["RSI_Divergence"] = (price_at_new_low & rsi_higher).astype(float)

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
        # Valuation metrics
        "pegRatio", "priceToSalesTrailing12Months",
        "enterpriseToEbitda", "enterpriseToRevenue",
        "enterpriseValue", "bookValue",
        "trailingEps", "forwardEps",
        # Profitability / efficiency
        "returnOnAssets", "grossMargins", "operatingMargins",
        "ebitdaMargins",
        # Liquidity / leverage
        "quickRatio", "totalDebt", "totalCash",
        # Ownership / short interest
        "shortRatio", "shortPercentOfFloat",
        "heldPercentInsiders", "heldPercentInstitutions",
        # Growth
        "revenuePerShare", "earningsQuarterlyGrowth",
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
        # Valuation ratios
        "marketCap", "trailingPE", "forwardPE", "priceToBook",
        "pegRatio", "priceToSalesTrailing12Months",
        "enterpriseToEbitda", "enterpriseToRevenue",
        # Per-share data
        "trailingEps", "forwardEps", "bookValue", "revenuePerShare",
        # Profitability
        "dividendYield", "profitMargins", "grossMargins",
        "operatingMargins", "ebitdaMargins",
        "returnOnEquity", "returnOnAssets",
        # Growth
        "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        # Risk / leverage
        "beta", "debtToEquity", "currentRatio", "quickRatio",
        # Ownership / sentiment
        "shortRatio", "shortPercentOfFloat",
        "heldPercentInsiders", "heldPercentInstitutions",
        "numberOfAnalystOpinions",
    ]
    features = {}
    for k in feature_keys:
        val = info.get(k)
        if val is not None:
            try:
                features[k] = float(val)
            except (ValueError, TypeError):
                pass

    # Days until next earnings
    features["daysToEarnings"] = _get_days_to_earnings(ticker)

    return features


def _get_days_to_earnings(ticker: str) -> float:
    """Return calendar days until the next quarterly earnings date.

    Returns NaN if the date cannot be determined.
    """
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is None or cal.empty if isinstance(cal, pd.DataFrame) else not cal:
            return float("nan")

        # calendar can be a dict or DataFrame depending on yfinance version
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                earnings_date = pd.Timestamp(cal.loc["Earnings Date"].iloc[0])
            else:
                return float("nan")
        elif isinstance(cal, dict):
            ed = cal.get("Earnings Date") or cal.get("earningsDate")
            if ed is None:
                return float("nan")
            if isinstance(ed, list):
                ed = ed[0] if ed else None
            if ed is None:
                return float("nan")
            earnings_date = pd.Timestamp(ed)
        else:
            return float("nan")

        today = pd.Timestamp.now().normalize()
        delta = (earnings_date - today).days
        return float(max(delta, 0))
    except Exception:
        logger.debug("Could not fetch earnings date for %s", ticker)
        return float("nan")


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
