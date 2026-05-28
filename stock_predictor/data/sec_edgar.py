"""SEC EDGAR data — historical 10-Q/10-K filing features.

Uses the SEC's free EDGAR XBRL API to fetch filing dates and key
financial data directly from regulatory filings.  This provides the
most authoritative historical fundamental data without look-ahead bias.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

SEC_EDGAR_BASE = "https://data.sec.gov"
SEC_HEADERS = {
    "User-Agent": "StockPredictor Research research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# CIK mapping for major NASDAQ tickers (SEC uses CIK, not ticker)
_CIK_CACHE: dict[str, str] = {}

SEC_FEATURES = [
    "sec_revenue",
    "sec_net_income",
    "sec_eps",
    "sec_total_assets",
    "sec_total_liabilities",
    "sec_stockholders_equity",
    "sec_operating_cash_flow",
    "sec_filing_age_days",
]


def _get_cik(ticker: str) -> str | None:
    """Look up the SEC CIK number for a ticker symbol."""
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    try:
        url = f"{SEC_EDGAR_BASE}/submissions/CIK{ticker.upper()}.json"
        # Try ticker-based lookup first (doesn't always work)
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2020-01-01&forms=10-K,10-Q".format(ticker),
            headers=SEC_HEADERS,
            timeout=10,
        )

        # Use the company tickers JSON for reliable lookup
        tickers_url = f"{SEC_EDGAR_BASE}/files/company_tickers.json"
        resp = requests.get(tickers_url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    _CIK_CACHE[ticker] = cik
                    return cik
    except Exception:
        logger.debug("Could not look up CIK for %s", ticker)

    return None


def get_sec_filings(ticker: str) -> pd.DataFrame:
    """Fetch key financials from SEC EDGAR XBRL API.

    Uses the companyfacts endpoint which provides structured XBRL data
    for all historical filings.

    Returns:
        DataFrame indexed by filing period end date with SEC_FEATURES columns.
    """
    cik = _get_cik(ticker)
    if cik is None:
        logger.debug("No CIK found for %s", ticker)
        return pd.DataFrame()

    try:
        url = f"{SEC_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        time.sleep(0.15)  # SEC rate limit: 10 req/sec

        if resp.status_code != 200:
            logger.debug("SEC API returned %d for %s", resp.status_code, ticker)
            return pd.DataFrame()

        data = resp.json()
        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        if not us_gaap:
            return pd.DataFrame()

        def _extract_quarterly(concept: str) -> dict[str, float]:
            """Extract quarterly values keyed by period-end date."""
            entries = us_gaap.get(concept, {}).get("units", {})
            values: dict[str, float] = {}
            for unit_entries in entries.values():
                for e in unit_entries:
                    form = e.get("form", "")
                    if form in ("10-Q", "10-K"):
                        end = e.get("end", "")
                        val = e.get("val")
                        filed = e.get("filed", "")
                        if end and val is not None:
                            values[end] = float(val)
            return values

        revenue = _extract_quarterly("Revenues") or _extract_quarterly("RevenueFromContractWithCustomerExcludingAssessedTax")
        net_income = _extract_quarterly("NetIncomeLoss")
        eps = _extract_quarterly("EarningsPerShareDiluted")
        total_assets = _extract_quarterly("Assets")
        total_liab = _extract_quarterly("Liabilities")
        equity = _extract_quarterly("StockholdersEquity")
        op_cf = _extract_quarterly("NetCashProvidedByOperatingActivities")

        # Combine all dates
        all_dates = sorted(set(
            list(revenue.keys()) + list(net_income.keys()) +
            list(total_assets.keys())
        ))

        if not all_dates:
            return pd.DataFrame()

        records: list[dict] = []
        for d in all_dates:
            records.append({
                "_filing_date": d,
                "sec_revenue": revenue.get(d, np.nan),
                "sec_net_income": net_income.get(d, np.nan),
                "sec_eps": eps.get(d, np.nan),
                "sec_total_assets": total_assets.get(d, np.nan),
                "sec_total_liabilities": total_liab.get(d, np.nan),
                "sec_stockholders_equity": equity.get(d, np.nan),
                "sec_operating_cash_flow": op_cf.get(d, np.nan),
            })

        result = pd.DataFrame(records)
        result["_filing_date"] = pd.to_datetime(result["_filing_date"])
        result = result.sort_values("_filing_date").reset_index(drop=True)
        return result

    except Exception:
        logger.exception("Error fetching SEC data for %s", ticker)
        return pd.DataFrame()


def align_sec_to_dates(
    sec_df: pd.DataFrame,
    dates: pd.DatetimeIndex | pd.Index,
) -> pd.DataFrame:
    """Align SEC filing data to training dates (most recent filing before each date).

    Also computes sec_filing_age_days — how old the most recent filing is.
    """
    if sec_df.empty:
        return pd.DataFrame(
            {col: np.nan for col in SEC_FEATURES},
            index=range(len(dates)),
        )

    filing_dates = sec_df["_filing_date"].values
    feature_cols = [c for c in SEC_FEATURES if c in sec_df.columns and c != "sec_filing_age_days"]

    aligned_rows: list[dict] = []
    for d in dates:
        ts = pd.Timestamp(d)
        mask = filing_dates <= ts
        if mask.any():
            idx = np.where(mask)[0][-1]
            row = sec_df.iloc[idx]
            rec = {col: row.get(col, np.nan) for col in feature_cols}
            rec["sec_filing_age_days"] = float((ts - pd.Timestamp(filing_dates[idx])).days)
        else:
            rec = {col: np.nan for col in feature_cols}
            rec["sec_filing_age_days"] = np.nan

        aligned_rows.append(rec)

    return pd.DataFrame(aligned_rows)
