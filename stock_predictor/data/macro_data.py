"""Macroeconomic data from FRED and YFinance (VIX, interest rates, etc.).

All data is properly time-indexed so it can be aligned to any historical
date without look-ahead bias.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MACRO_FEATURES = [
    "vix_close",
    "treasury_10y",
    "treasury_3m",
    "yield_curve_spread",
    "sp500_return_20d",
    "sp500_return_60d",
    "sp500_volatility_20d",
    "dollar_index_return_20d",
    "gold_return_20d",
    "oil_return_20d",
]


def _fetch_series(ticker: str, period: str = "6y") -> pd.DataFrame:
    """Fetch a YFinance series and return date-indexed Close prices."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Handle multi-level columns from yf.download
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df[["Close"]].rename(columns={"Close": ticker})
    except Exception:
        logger.debug("Could not fetch %s", ticker)
        return pd.DataFrame()


def get_macro_data(period: str = "6y") -> pd.DataFrame:
    """Fetch macroeconomic time series and compute derived features.

    Returns a date-indexed DataFrame with MACRO_FEATURES columns.
    """
    # VIX (CBOE Volatility Index)
    vix = _fetch_series("^VIX", period)
    # S&P 500
    sp500 = _fetch_series("^GSPC", period)
    # US 10-Year Treasury Yield
    tnx = _fetch_series("^TNX", period)
    # US 2-Year Treasury Yield
    irx = _fetch_series("^IRX", period)  # 13-week T-bill as proxy for short end
    # US Dollar Index
    dxy = _fetch_series("DX-Y.NYB", period)
    # Gold
    gold = _fetch_series("GC=F", period)
    # Crude Oil
    oil = _fetch_series("CL=F", period)

    # Merge all on date index
    frames = [vix, sp500, tnx, irx, dxy, gold, oil]
    merged = pd.DataFrame(index=pd.DatetimeIndex([]))
    for frame in frames:
        if not frame.empty:
            merged = merged.join(frame, how="outer")

    if merged.empty:
        logger.warning("No macro data could be fetched")
        return pd.DataFrame()

    merged = merged.sort_index().ffill()

    result = pd.DataFrame(index=merged.index)

    # VIX
    if "^VIX" in merged.columns:
        result["vix_close"] = merged["^VIX"]
        result["vix_sma_20"] = merged["^VIX"].rolling(20).mean()
    else:
        result["vix_close"] = np.nan
        result["vix_sma_20"] = np.nan

    # Treasury yields
    if "^TNX" in merged.columns:
        result["treasury_10y"] = merged["^TNX"]
    else:
        result["treasury_10y"] = np.nan

    if "^IRX" in merged.columns:
        result["treasury_3m"] = merged["^IRX"]
    else:
        result["treasury_3m"] = np.nan

    # Yield curve spread (10y - 2y)
    result["yield_curve_spread"] = result["treasury_10y"] - result["treasury_3m"]

    # S&P 500 momentum and volatility
    if "^GSPC" in merged.columns:
        sp = merged["^GSPC"]
        result["sp500_return_20d"] = sp.pct_change(20)
        result["sp500_return_60d"] = sp.pct_change(60)
        result["sp500_volatility_20d"] = sp.pct_change().rolling(20).std() * np.sqrt(252)
    else:
        result["sp500_return_20d"] = np.nan
        result["sp500_return_60d"] = np.nan
        result["sp500_volatility_20d"] = np.nan

    # Dollar index
    if "DX-Y.NYB" in merged.columns:
        result["dollar_index_return_20d"] = merged["DX-Y.NYB"].pct_change(20)
    else:
        result["dollar_index_return_20d"] = np.nan

    # Gold
    if "GC=F" in merged.columns:
        result["gold_return_20d"] = merged["GC=F"].pct_change(20)
    else:
        result["gold_return_20d"] = np.nan

    # Oil
    if "CL=F" in merged.columns:
        result["oil_return_20d"] = merged["CL=F"].pct_change(20)
    else:
        result["oil_return_20d"] = np.nan

    return result


def align_macro_to_dates(
    macro_df: pd.DataFrame,
    dates: pd.DatetimeIndex | pd.Index,
) -> pd.DataFrame:
    """Align macro data to given dates using as-of join (no lookahead).

    For each target date, uses the most recent available macro data point.
    """
    if macro_df.empty:
        return pd.DataFrame(
            {col: np.nan for col in MACRO_FEATURES},
            index=range(len(dates)),
        )

    feature_cols = [c for c in MACRO_FEATURES if c in macro_df.columns]
    macro_dates = macro_df.index.values

    aligned_rows: list[dict] = []
    for d in dates:
        ts = pd.Timestamp(d)
        mask = macro_dates <= ts
        if mask.any():
            idx = np.where(mask)[0][-1]
            row = macro_df.iloc[idx]
            aligned_rows.append({col: row.get(col, np.nan) for col in feature_cols})
        else:
            aligned_rows.append({col: np.nan for col in feature_cols})

    return pd.DataFrame(aligned_rows)
