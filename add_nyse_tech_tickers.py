"""Append NYSE/missing tech tickers to the existing 10-year training dataset.

Generates the same features as generate_10y_dataset.py for new tickers
and appends them to training_data_10y_full.csv.

Usage:
    python add_nyse_tech_tickers.py
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("add_nyse_tech.log"),
    ],
)
logger = logging.getLogger(__name__)

from stock_predictor.data.yfinance_client import get_stock_data, compute_technical_features
from stock_predictor.data.macro_data import get_macro_data, align_macro_to_dates

from add_edgar_fundamentals_5y import (
    load_cik_map,
    fetch_edgar_fundamentals,
    align_fundamentals_to_dates,
    OUTPUT_FEATURES as EDGAR_FEATURES,
)
from stock_predictor.data.insider_transactions import (
    get_insider_transactions,
    align_insider_to_dates,
    INSIDER_FEATURES,
)

OUTPUT_FILE = "training_data_10y_full.csv"
CHECKPOINT_FILE = "nyse_tech_checkpoint.csv"

# NYSE / missing tech tickers to add
NEW_TICKERS = [
    # AI / Software / Cloud
    "PLTR", "PATH", "SNOW", "AI", "S", "TWLO", "NET", "SHOP",
    "UBER", "LYFT", "RBLX", "U", "MDB", "ESTC", "GTLB", "DT", "FROG",
    # Legacy tech / enterprise
    "IBM", "ORCL", "HPE", "DELL", "WDAY",
    # Semiconductors
    "TSM", "GFS", "IONQ", "RGTI",
    # Space / defense tech
    "RKLB", "RDW", "BKSY",
    # Fintech
    "UPST", "SOFI", "HOOD", "NU",
    # EV / clean energy tech
    "RIVN", "LCID", "QS", "CHPT",
    # Cybersecurity
    "ZS", "OKTA", "TENB", "VRNS",
    # E-commerce / digital
    "ETSY", "W", "PINS", "SNAP", "SPOT",
    # Biotech (tech-adjacent)
    "NTLA",
    # Data / Analytics
    "TYL",
    # Asia tech
    "SE", "GRAB", "BABA", "JD", "PDD",
    # Space / quantum
    "ASTS",
]

# Columns to keep (same as generate_10y_dataset.py)
KEEP_COLS = {
    "BB_Position", "BB_Squeeze_Duration", "BB_Width", "Days_Since_SMA200_Cross",
    "Dist_52w_High", "Dist_52w_Low", "MACD", "MACD_Hist", "Momentum_Accel",
    "Price_to_SMA_20", "Price_to_SMA_200", "Price_to_SMA_50", "RSI_14",
    "Return_1d", "Return_20d", "Return_5d", "Return_60d",
    "Volatility_20d", "Volatility_60d", "Volatility_Contraction",
    "Volume_Price_Confirm", "Volume_Ratio", "Volume_Spike_Magnitude",
    "Volume_Surge_3d",
    "dollar_index_return_20d", "gold_return_20d", "oil_return_20d",
    "sp500_return_20d", "sp500_return_60d", "sp500_volatility_20d",
    "treasury_10y", "treasury_3m", "vix_close", "yield_curve_spread",
    "hist_capex", "hist_current_ratio", "hist_debt_to_equity",
    "hist_diluted_eps", "hist_earnings_growth_qoq", "hist_net_income",
    "hist_operating_income", "hist_operating_margin", "hist_profit_margin",
    "hist_revenue_growth_qoq", "hist_roa", "hist_roe",
    "hist_stockholders_equity", "hist_total_assets", "hist_total_revenue",
    "sec_filing_age_days", "sec_operating_cash_flow",
    "insider_buy_ratio_90d", "insider_net_buys_90d",
    "insider_total_transactions_90d",
    "Ticker", "_date", "Forward_Max_Return_3M",
}


def compute_forward_return(close_prices: np.ndarray, window: int = 63) -> np.ndarray:
    n = len(close_prices)
    fwd = np.full(n, np.nan)
    for i in range(n - 1):
        end = min(i + window + 1, n)
        future = close_prices[i + 1 : end]
        if len(future) > 0:
            fwd[i] = (future.max() / close_prices[i]) - 1.0
    return fwd


def main():
    # Load existing dataset to find which tickers are already present
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE, usecols=["Ticker"])
        existing_tickers = set(existing_df["Ticker"].unique())
        logger.info("Existing dataset has %d tickers", len(existing_tickers))
    else:
        existing_tickers = set()
        logger.warning("No existing dataset found at %s", OUTPUT_FILE)

    # Filter to only new tickers
    tickers_to_add = [t for t in NEW_TICKERS if t not in existing_tickers]
    already_present = [t for t in NEW_TICKERS if t in existing_tickers]
    if already_present:
        logger.info("Already in dataset (skipping): %s", already_present)
    logger.info("New tickers to generate: %d — %s", len(tickers_to_add), tickers_to_add)

    if not tickers_to_add:
        logger.info("No new tickers to add. Exiting.")
        return

    # Load CIK map for EDGAR
    cik_map = load_cik_map()

    # Fetch macro data (11 years)
    logger.info("Fetching macro data...")
    macro_df = get_macro_data(period="11y")
    macro_df.index = pd.to_datetime(macro_df.index)
    logger.info("Macro data: %d rows", len(macro_df))

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
        checkpoint_df["_date"] = pd.to_datetime(checkpoint_df["_date"])
        completed = set(checkpoint_df["Ticker"].unique())
        logger.info("Resuming: %d tickers done (%d rows)", len(completed), len(checkpoint_df))
    else:
        checkpoint_df = pd.DataFrame()
        completed = set()

    remaining = [t for t in tickers_to_add if t not in completed]
    logger.info("Remaining: %d / %d tickers", len(remaining), len(tickers_to_add))

    batch_rows = []
    errors = 0

    for i, ticker in enumerate(remaining):
        try:
            hist = get_stock_data(ticker, period="10y")
            if hist.empty or len(hist) < 126:
                logger.warning("Insufficient data for %s (%d rows)", ticker, len(hist) if not hist.empty else 0)
                errors += 1
                continue

            tech_df = compute_technical_features(hist)
            if tech_df.empty or len(tech_df) < 126:
                errors += 1
                continue

            if "Date" in tech_df.columns:
                tech_df["Date"] = pd.to_datetime(tech_df["Date"]).dt.tz_localize(None)
                tech_df = tech_df.set_index("Date")
            else:
                tech_df.index = pd.to_datetime(tech_df.index)

            aligned_macro = align_macro_to_dates(macro_df, tech_df.index)
            merged = tech_df.copy()
            for col in aligned_macro.columns:
                merged[col] = aligned_macro[col].values

            merged["Forward_Max_Return_3M"] = compute_forward_return(merged["Close"].values)

            cik = cik_map.get(ticker)
            if cik:
                try:
                    fund_df = fetch_edgar_fundamentals(cik)
                    if not fund_df.empty and len(fund_df) >= 2:
                        aligned = align_fundamentals_to_dates(fund_df, merged.index)
                        for feat in EDGAR_FEATURES:
                            merged[feat] = aligned[feat].values
                    else:
                        for feat in EDGAR_FEATURES:
                            merged[feat] = np.nan
                except Exception:
                    for feat in EDGAR_FEATURES:
                        merged[feat] = np.nan
            else:
                for feat in EDGAR_FEATURES:
                    merged[feat] = np.nan

            try:
                aligned_ins = align_insider_to_dates(ticker, merged.index)
                for feat in INSIDER_FEATURES:
                    if feat in aligned_ins.columns:
                        merged[feat] = aligned_ins[feat].values
                    else:
                        merged[feat] = np.nan
            except Exception:
                for feat in INSIDER_FEATURES:
                    merged[feat] = np.nan

            merged["Ticker"] = ticker
            merged["_date"] = merged.index.strftime("%Y-%m-%d")
            merged = merged.reset_index(drop=True)

            available_keep = [c for c in KEEP_COLS if c in merged.columns]
            merged = merged[available_keep]
            merged = merged.dropna(subset=["Forward_Max_Return_3M"])

            rows_added = len(merged)
            batch_rows.append(merged)
            logger.info("[%d/%d] %s: %d rows generated", i + 1, len(remaining), ticker, rows_added)

        except Exception as e:
            logger.warning("Error processing %s: %s", ticker, e)
            errors += 1

        # Checkpoint every 10 tickers
        done = len(completed) + i + 1
        if done % 10 == 0 and batch_rows:
            batch = pd.concat(batch_rows, ignore_index=True)
            if not checkpoint_df.empty:
                checkpoint_df = pd.concat([checkpoint_df, batch], ignore_index=True)
            else:
                checkpoint_df = batch
            checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)
            batch_rows = []
            logger.info("Checkpointed at %d tickers (%d rows)", done, len(checkpoint_df))

        time.sleep(0.5)  # Rate limit

    # Final concat
    if batch_rows:
        batch = pd.concat(batch_rows, ignore_index=True)
        if not checkpoint_df.empty:
            checkpoint_df = pd.concat([checkpoint_df, batch], ignore_index=True)
        else:
            checkpoint_df = batch

    if checkpoint_df.empty:
        logger.warning("No new data generated.")
        return

    new_data = checkpoint_df
    new_data["_date"] = pd.to_datetime(new_data["_date"])
    new_data = new_data.sort_values(["Ticker", "_date"]).reset_index(drop=True)

    logger.info("=" * 60)
    logger.info("NEW DATA GENERATED")
    logger.info("Rows: %d | Tickers: %d", len(new_data), new_data["Ticker"].nunique())
    logger.info("=" * 60)

    # Append to existing dataset
    if os.path.exists(OUTPUT_FILE):
        logger.info("Appending to %s ...", OUTPUT_FILE)
        existing = pd.read_csv(OUTPUT_FILE)
        existing["_date"] = pd.to_datetime(existing["_date"])
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.sort_values(["Ticker", "_date"]).reset_index(drop=True)
        combined.to_csv(OUTPUT_FILE, index=False)
        logger.info("Updated dataset: %d rows | %d tickers",
                     len(combined), combined["Ticker"].nunique())
    else:
        new_data.to_csv(OUTPUT_FILE, index=False)
        logger.info("Created new dataset: %d rows | %d tickers",
                     len(new_data), new_data["Ticker"].nunique())

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Cleaned up checkpoint file")


if __name__ == "__main__":
    main()
