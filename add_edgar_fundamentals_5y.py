"""Add EDGAR XBRL fundamentals to the 5-year dataset.

Fetches all historical quarterly financials from SEC EDGAR for each ticker,
computes derived features (margins, ratios, growth rates), and time-aligns
to each training row date.
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("edgar_fundamentals_5y.log"),
    ],
)
logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "StockPredictor Research research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

CHECKPOINT_FILE = "edgar_fund_checkpoint.csv"
PROGRESS_FILE = "edgar_fund_progress.txt"
OUTPUT_FILE = "training_data_5y_full.csv"

# XBRL concepts to extract.
# Multiple tag variants per concept — verified against EDGAR API for
# AAPL, AMZN, GOOG, META, NFLX, AAON, and 30+ NaN-producing tickers.
CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "Revenues",
        # Banks / financial companies use interest income as primary revenue
        "InterestIncomeExpenseNet",
        "InterestAndDividendIncomeOperating",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "total_assets": ["Assets"],
    "total_liabilities": [
        "Liabilities",
        "LiabilitiesAndStockholdersEquity",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByOperatingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditureDiscontinuedOperations",
    ],
}


def load_cik_map() -> dict[str, str]:
    """Load full ticker->CIK mapping from SEC."""
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    cik_map = {}
    if resp.status_code == 200:
        for entry in resp.json().values():
            t = entry.get("ticker", "").upper()
            cik = str(entry["cik_str"]).zfill(10)
            cik_map[t] = cik
    logger.info("Loaded %d CIK mappings", len(cik_map))
    return cik_map


def extract_quarterly(us_gaap: dict, concept_names: list[str]) -> list[dict]:
    """Extract quarterly values for a concept (try multiple XBRL names).
    
    Returns list of {end, filed, val} dicts sorted by end date.
    Uses the 'filed' date to ensure time-alignment (no look-ahead).
    """
    for concept in concept_names:
        entries = us_gaap.get(concept, {}).get("units", {})
        results = []
        seen = set()
        for unit_entries in entries.values():
            for e in unit_entries:
                form = e.get("form", "")
                if form in ("10-Q", "10-K"):
                    end = e.get("end", "")
                    filed = e.get("filed", "")
                    val = e.get("val")
                    # Use (end, form) as dedup key — keep first occurrence
                    key = (end, form)
                    if end and val is not None and filed and key not in seen:
                        seen.add(key)
                        results.append({
                            "end": end,
                            "filed": filed,
                            "val": float(val),
                        })
        if results:
            results.sort(key=lambda x: x["end"])
            return results
    return []


def fetch_edgar_fundamentals(cik: str) -> pd.DataFrame:
    """Fetch all quarterly fundamentals from EDGAR XBRL API for a CIK.
    
    Returns DataFrame with columns: _filing_date (when filed, for time-alignment),
    _period_end (fiscal period end), and all raw fundamental values.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
    time.sleep(0.12)  # SEC rate limit: 10 req/sec

    if resp.status_code != 200:
        return pd.DataFrame()

    data = resp.json()
    us_gaap = data.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return pd.DataFrame()

    # Extract all concepts
    extracted = {}
    for feat_name, concept_names in CONCEPTS.items():
        entries = extract_quarterly(us_gaap, concept_names)
        if entries:
            extracted[feat_name] = {e["end"]: e for e in entries}

    if not extracted:
        return pd.DataFrame()

    # Collect all unique period-end dates
    all_dates = set()
    for feat_entries in extracted.values():
        all_dates.update(feat_entries.keys())
    all_dates = sorted(all_dates)

    if not all_dates:
        return pd.DataFrame()

    records = []
    for d in all_dates:
        rec = {"_period_end": d}
        # Use the latest 'filed' date across concepts for this period
        filed_dates = []
        for feat_name, feat_entries in extracted.items():
            entry = feat_entries.get(d)
            if entry:
                rec[feat_name] = entry["val"]
                filed_dates.append(entry["filed"])
            else:
                rec[feat_name] = np.nan
        rec["_filing_date"] = max(filed_dates) if filed_dates else d
        records.append(rec)

    result = pd.DataFrame(records)
    result["_filing_date"] = pd.to_datetime(result["_filing_date"])
    result["_period_end"] = pd.to_datetime(result["_period_end"])
    result = result.sort_values("_filing_date").reset_index(drop=True)
    return result


def compute_derived_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived fundamental features from raw EDGAR data."""
    df = fund_df.copy()

    # Margins
    if "revenue" in df.columns and "operating_income" in df.columns:
        df["operating_margin"] = np.where(
            df["revenue"] > 0,
            df["operating_income"] / df["revenue"],
            np.nan,
        )
    if "revenue" in df.columns and "net_income" in df.columns:
        df["profit_margin"] = np.where(
            df["revenue"] > 0,
            df["net_income"] / df["revenue"],
            np.nan,
        )

    # Returns
    if "net_income" in df.columns and "stockholders_equity" in df.columns:
        df["roe"] = np.where(
            df["stockholders_equity"].abs() > 0,
            df["net_income"] / df["stockholders_equity"],
            np.nan,
        )
    if "net_income" in df.columns and "total_assets" in df.columns:
        df["roa"] = np.where(
            df["total_assets"] > 0,
            df["net_income"] / df["total_assets"],
            np.nan,
        )

    # Leverage
    if "long_term_debt" in df.columns and "stockholders_equity" in df.columns:
        df["debt_to_equity"] = np.where(
            df["stockholders_equity"].abs() > 0,
            df["long_term_debt"] / df["stockholders_equity"],
            np.nan,
        )

    # Liquidity
    if "current_assets" in df.columns and "current_liabilities" in df.columns:
        df["current_ratio"] = np.where(
            df["current_liabilities"] > 0,
            df["current_assets"] / df["current_liabilities"],
            np.nan,
        )

    # QoQ growth rates
    if "eps_diluted" in df.columns:
        prev_eps = df["eps_diluted"].shift(1)
        df["earnings_growth_qoq"] = np.where(
            prev_eps.abs() > 0.01,
            (df["eps_diluted"] - prev_eps) / prev_eps.abs(),
            np.nan,
        )
    if "revenue" in df.columns:
        prev_rev = df["revenue"].shift(1)
        df["revenue_growth_qoq"] = np.where(
            prev_rev.abs() > 0,
            (df["revenue"] - prev_rev) / prev_rev.abs(),
            np.nan,
        )

    return df


# Features to output (after derivation)
OUTPUT_FEATURES = [
    "hist_total_revenue",
    "hist_net_income",
    "hist_diluted_eps",
    "hist_operating_income",
    "hist_operating_margin",
    "hist_profit_margin",
    "hist_total_assets",
    "hist_stockholders_equity",
    "hist_current_ratio",
    "hist_debt_to_equity",
    "hist_roe",
    "hist_roa",
    "hist_earnings_growth_qoq",
    "hist_revenue_growth_qoq",
    "hist_capex",
    "sec_operating_cash_flow",
    "sec_filing_age_days",
]

# Mapping from derived columns to output names
COL_RENAME = {
    "revenue": "hist_total_revenue",
    "net_income": "hist_net_income",
    "eps_diluted": "hist_diluted_eps",
    "operating_income": "hist_operating_income",
    "operating_margin": "hist_operating_margin",
    "profit_margin": "hist_profit_margin",
    "total_assets": "hist_total_assets",
    "stockholders_equity": "hist_stockholders_equity",
    "current_ratio": "hist_current_ratio",
    "debt_to_equity": "hist_debt_to_equity",
    "roe": "hist_roe",
    "roa": "hist_roa",
    "earnings_growth_qoq": "hist_earnings_growth_qoq",
    "revenue_growth_qoq": "hist_revenue_growth_qoq",
    "capex": "hist_capex",
    "operating_cash_flow": "sec_operating_cash_flow",
}


def align_fundamentals_to_dates(
    fund_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Time-align fundamentals: for each date, use most recent filing BEFORE that date.
    
    Uses _filing_date (when the SEC received the filing) to prevent look-ahead bias.
    """
    if fund_df.empty:
        return pd.DataFrame(
            {feat: np.nan for feat in OUTPUT_FEATURES},
            index=range(len(dates)),
        )

    # Compute derived features
    fund_df = compute_derived_features(fund_df)
    
    # Rename columns to output names
    fund_df = fund_df.rename(columns=COL_RENAME)

    filing_dates = fund_df["_filing_date"].values
    available_output = [c for c in OUTPUT_FEATURES if c in fund_df.columns and c != "sec_filing_age_days"]

    aligned_rows = []
    for d in dates:
        ts = pd.Timestamp(d)
        mask = filing_dates <= ts
        if mask.any():
            idx = np.where(mask)[0][-1]
            row = fund_df.iloc[idx]
            rec = {col: row.get(col, np.nan) for col in available_output}
            rec["sec_filing_age_days"] = float((ts - pd.Timestamp(filing_dates[idx])).days)
        else:
            rec = {col: np.nan for col in available_output}
            rec["sec_filing_age_days"] = np.nan
        aligned_rows.append(rec)

    result = pd.DataFrame(aligned_rows)
    # Fill missing output features with NaN
    for feat in OUTPUT_FEATURES:
        if feat not in result.columns:
            result[feat] = np.nan
    return result[OUTPUT_FEATURES]


def main():
    # Load the base dataset (5-year with insider + mcap filter)
    input_file = "training_data_5y_insider_mcap100m.csv"
    logger.info("Loading %s ...", input_file)
    df = pd.read_csv(input_file)
    df["_date"] = pd.to_datetime(df["_date"])
    logger.info("Loaded %d rows, %d tickers", len(df), df["Ticker"].nunique())

    tickers = sorted(df["Ticker"].dropna().unique())
    logger.info("Total tickers: %d", len(tickers))

    # Load CIK map
    cik_map = load_cik_map()

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
        checkpoint_df["_date"] = pd.to_datetime(checkpoint_df["_date"])
        completed_tickers = set(checkpoint_df["Ticker"].unique())
        logger.info("Resuming from checkpoint: %d tickers done", len(completed_tickers))
    else:
        checkpoint_df = pd.DataFrame()
        completed_tickers = set()

    remaining = [t for t in tickers if t not in completed_tickers]
    logger.info("Remaining: %d tickers", len(remaining))

    batch_rows = []
    no_cik = 0
    no_data = 0
    too_few_quarters = 0

    for i, ticker in enumerate(remaining):
        cik = cik_map.get(ticker)
        if not cik:
            no_cik += 1
            # Fill with NaN
            ticker_df = df[df["Ticker"] == ticker].copy()
            for feat in OUTPUT_FEATURES:
                ticker_df[feat] = np.nan
            batch_rows.append(ticker_df)
            if (i + 1) % 50 == 0:
                logger.info("Progress: %d / %d (no CIK: %d)", 
                           len(completed_tickers) + i + 1, len(tickers), no_cik)
            continue

        try:
            fund_df = fetch_edgar_fundamentals(cik)
        except Exception as e:
            logger.debug("Error fetching %s: %s", ticker, e)
            fund_df = pd.DataFrame()

        ticker_df = df[df["Ticker"] == ticker].copy()
        dates = ticker_df["_date"]

        if fund_df.empty or len(fund_df) < 2:
            if fund_df.empty:
                no_data += 1
            else:
                too_few_quarters += 1
            for feat in OUTPUT_FEATURES:
                ticker_df[feat] = np.nan
        else:
            aligned = align_fundamentals_to_dates(fund_df, dates)
            for feat in OUTPUT_FEATURES:
                ticker_df[feat] = aligned[feat].values

        batch_rows.append(ticker_df)

        done = len(completed_tickers) + i + 1
        if done % 25 == 0:
            logger.info("Progress: %d / %d tickers (%.1f%%) | no_cik=%d no_data=%d <2Q=%d",
                        done, len(tickers), done/len(tickers)*100,
                        no_cik, no_data, too_few_quarters)
            with open(PROGRESS_FILE, "a") as f:
                f.write(f"{done}\n")

        # Checkpoint every 50 tickers
        if done % 50 == 0 and batch_rows:
            batch = pd.concat(batch_rows, ignore_index=True)
            if not checkpoint_df.empty:
                checkpoint_df = pd.concat([checkpoint_df, batch], ignore_index=True)
            else:
                checkpoint_df = batch
            checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)
            batch_rows = []
            logger.info("Checkpointed at %d tickers (%d rows)", done, len(checkpoint_df))

    # Final merge
    if batch_rows:
        batch = pd.concat(batch_rows, ignore_index=True)
        if not checkpoint_df.empty:
            checkpoint_df = pd.concat([checkpoint_df, batch], ignore_index=True)
        else:
            checkpoint_df = batch

    result = checkpoint_df
    logger.info("Final dataset: %d rows, %d columns", len(result), len(result.columns))
    
    # Report NaN rates for new features
    logger.info("NaN rates for EDGAR fundamental features:")
    for feat in OUTPUT_FEATURES:
        if feat in result.columns:
            nan_pct = result[feat].isna().mean() * 100
            logger.info("  %s: %.1f%% NaN", feat, nan_pct)

    result.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved to %s", OUTPUT_FILE)

    # Stats
    logger.info("Stats: no_cik=%d no_data=%d too_few_quarters=%d", no_cik, no_data, too_few_quarters)


if __name__ == "__main__":
    main()
