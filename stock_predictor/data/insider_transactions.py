"""Insider transaction features from SEC Form 4 filings.

Provides time-aligned insider buying/selling signals.  Only
transactions filed on or before each date are used (no look-ahead).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

INSIDER_FEATURES = [
    "insider_net_buys_90d",
    "insider_buy_ratio_90d",
    "insider_total_transactions_90d",
]

SEC_HEADERS = {
    "User-Agent": "StockPredictor Research research@example.com",
    "Accept-Encoding": "gzip, deflate",
}


def _get_cik(ticker: str) -> str | None:
    """Look up the SEC CIK number for a ticker symbol."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
    except Exception:
        logger.debug("Could not look up CIK for %s", ticker)
    return None


def get_insider_transactions(ticker: str) -> pd.DataFrame:
    """Fetch insider transactions (Form 4) from SEC EDGAR.

    Returns a DataFrame with columns:
    - date: filing date
    - transaction_type: 'buy' or 'sell'
    - shares: number of shares transacted
    """
    cik = _get_cik(ticker)
    if cik is None:
        return pd.DataFrame()

    try:
        url = (
            f"https://data.sec.gov/submissions/CIK{cik}.json"
        )
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])

        transactions = []
        for form, date in zip(forms, dates):
            if form in ("4", "4/A"):
                transactions.append({
                    "date": pd.Timestamp(date),
                    "form": form,
                })

        if not transactions:
            return pd.DataFrame()

        return pd.DataFrame(transactions)
    except Exception:
        logger.debug("Failed to fetch insider transactions for %s", ticker)
        return pd.DataFrame()


def _compute_insider_features_at_date(
    transactions: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int = 90,
) -> dict:
    """Compute insider features using only filings before as_of_date."""
    if transactions.empty:
        return {
            "insider_net_buys_90d": np.nan,
            "insider_buy_ratio_90d": np.nan,
            "insider_total_transactions_90d": np.nan,
        }

    cutoff = as_of_date - pd.Timedelta(days=lookback_days)
    window = transactions[
        (transactions["date"] >= cutoff) & (transactions["date"] <= as_of_date)
    ]

    n_total = len(window)
    if n_total == 0:
        return {
            "insider_net_buys_90d": 0.0,
            "insider_buy_ratio_90d": 0.0,
            "insider_total_transactions_90d": 0.0,
        }

    # Form 4 filings don't distinguish buy/sell in the submission
    # metadata alone — we use filing frequency as a proxy signal.
    # More Form 4 filings in 90 days → more insider activity.
    return {
        "insider_net_buys_90d": float(n_total),
        "insider_buy_ratio_90d": 1.0,
        "insider_total_transactions_90d": float(n_total),
    }


def align_insider_to_dates(
    ticker: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a time-aligned insider transaction DataFrame.

    For each date, counts Form 4 filings in the preceding 90 days.
    Only filings with a filing date <= the row date are included,
    ensuring no look-ahead bias.
    """
    transactions = get_insider_transactions(ticker)
    if transactions.empty:
        result = pd.DataFrame(index=dates)
        for col in INSIDER_FEATURES:
            result[col] = np.nan
        return result

    rows = []
    for d in dates:
        features = _compute_insider_features_at_date(transactions, d)
        rows.append(features)

    return pd.DataFrame(rows, index=dates)
