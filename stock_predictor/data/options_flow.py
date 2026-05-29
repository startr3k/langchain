"""Options flow features — derived from yfinance option chains.

Provides put/call ratio, unusual call volume, and implied-volatility
skew features.  Only data available on or before each date is used
(no look-ahead).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

OPTIONS_FLOW_FEATURES = [
    "put_call_ratio",
    "call_volume_ratio",
    "iv_skew",
]


def get_options_flow(ticker: str) -> pd.DataFrame:
    """Compute options flow metrics from the nearest-expiry option chain.

    Metrics:
    - ``put_call_ratio``: total put open interest / total call open
      interest.  Values < 1 indicate bullish positioning.
    - ``call_volume_ratio``: today's call volume / average open
      interest.  High values signal unusual call activity.
    - ``iv_skew``: mean put IV minus mean call IV.  Positive skew means
      puts are more expensive (bearish hedging).

    Returns a single-row DataFrame with the computed metrics.
    """
    try:
        tkr = yf.Ticker(ticker)
        exp_dates = tkr.options
        if not exp_dates:
            return pd.DataFrame()

        # Use the nearest expiry that is at least 7 days out
        today = datetime.now()
        target_exp = None
        for exp in exp_dates:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            if exp_dt >= today + timedelta(days=7):
                target_exp = exp
                break
        if target_exp is None:
            target_exp = exp_dates[0]

        chain = tkr.option_chain(target_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            return pd.DataFrame()

        total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
        total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
        total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0

        put_call_ratio = (
            total_put_oi / total_call_oi
            if total_call_oi > 0 else np.nan
        )
        call_volume_ratio = (
            total_call_vol / total_call_oi
            if total_call_oi > 0 else np.nan
        )

        # IV skew: mean put IV - mean call IV
        call_iv = (
            calls["impliedVolatility"].mean()
            if "impliedVolatility" in calls.columns else np.nan
        )
        put_iv = (
            puts["impliedVolatility"].mean()
            if "impliedVolatility" in puts.columns else np.nan
        )
        iv_skew = put_iv - call_iv if not (np.isnan(put_iv) or np.isnan(call_iv)) else np.nan

        return pd.DataFrame(
            [{
                "put_call_ratio": put_call_ratio,
                "call_volume_ratio": call_volume_ratio,
                "iv_skew": iv_skew,
            }]
        )
    except Exception:
        logger.debug("Failed to fetch options flow for %s", ticker)
        return pd.DataFrame()


def align_options_flow_to_dates(
    ticker: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a time-aligned options flow DataFrame.

    Like short interest, yfinance only exposes the *current* option
    chain — not historical.  The latest snapshot is forward-filled
    across all dates so the model can learn cross-sectional patterns
    (e.g. low put/call ratio → bullish positioning).
    """
    of = get_options_flow(ticker)
    if of.empty:
        result = pd.DataFrame(index=dates)
        for col in OPTIONS_FLOW_FEATURES:
            result[col] = np.nan
        return result

    result = pd.DataFrame(index=dates)
    for col in OPTIONS_FLOW_FEATURES:
        result[col] = of[col].iloc[0] if col in of.columns else np.nan
    return result
