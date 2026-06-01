"""Generate 10-year training dataset (2016-2026) from scratch.

All features generated fresh for the full 10-year period — no merging.

Features:
- Technical indicators (from price/volume)
- Macro features (VIX, rates, SP500, gold, oil, dollar)
- EDGAR fundamentals (revenue, EPS, margins, ratios, growth)
- Insider transactions (Form 4)
- Target: Forward_Max_Return_3M (>=20%)

Fetches the full NASDAQ-listed ticker universe from the NASDAQ API and
filters by market cap (default $500M).  Pass ``--min-mcap`` to override.
"""

import argparse
import json
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
        logging.FileHandler("generate_10y.log"),
    ],
)
logger = logging.getLogger(__name__)

from stock_predictor.data.yfinance_client import (
    get_stock_data,
    compute_technical_features,
    fetch_all_nasdaq_tickers,
)
from stock_predictor.data.macro_data import get_macro_data, align_macro_to_dates
from stock_predictor.data.feature_engineering import TECHNICAL_FEATURES

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

CHECKPOINT_FILE = "10y_checkpoint.csv"
OUTPUT_FILE = "training_data_10y_full.csv"
MCAP_CACHE_FILE = "market_cap_cache.json"

# Columns to keep in the final dataset (matching 5-year dataset structure)
KEEP_COLS = {
    # Technical
    "BB_Position", "BB_Squeeze_Duration", "BB_Width", "Days_Since_SMA200_Cross",
    "Dist_52w_High", "Dist_52w_Low", "MACD", "MACD_Hist", "Momentum_Accel",
    "Price_to_SMA_20", "Price_to_SMA_200", "Price_to_SMA_50", "RSI_14",
    "Return_1d", "Return_20d", "Return_5d", "Return_60d",
    "Volatility_20d", "Volatility_60d", "Volatility_Contraction",
    "Volume_Price_Confirm", "Volume_Ratio", "Volume_Spike_Magnitude",
    "Volume_Surge_3d",
    # Macro
    "dollar_index_return_20d", "gold_return_20d", "oil_return_20d",
    "sp500_return_20d", "sp500_return_60d", "sp500_volatility_20d",
    "treasury_10y", "treasury_3m", "vix_close", "yield_curve_spread",
    # EDGAR fundamentals
    "hist_capex", "hist_current_ratio", "hist_debt_to_equity",
    "hist_diluted_eps", "hist_earnings_growth_qoq", "hist_net_income",
    "hist_operating_income", "hist_operating_margin", "hist_profit_margin",
    "hist_revenue_growth_qoq", "hist_roa", "hist_roe",
    "hist_stockholders_equity", "hist_total_assets", "hist_total_revenue",
    "sec_filing_age_days", "sec_operating_cash_flow",
    # Insider
    "insider_buy_ratio_90d", "insider_net_buys_90d",
    "insider_total_transactions_90d",
    # Meta + target
    "Ticker", "_date", "Forward_Max_Return_3M",
}


def compute_forward_return(close_prices: np.ndarray, window: int = 63) -> np.ndarray:
    """Compute forward max return in next `window` trading days."""
    n = len(close_prices)
    fwd = np.full(n, np.nan)
    for i in range(n - 1):
        end = min(i + window + 1, n)
        future = close_prices[i + 1 : end]
        if len(future) > 0:
            fwd[i] = (future.max() / close_prices[i]) - 1.0
    return fwd


def main():
    parser = argparse.ArgumentParser(description="Generate 10-year training dataset")
    parser.add_argument(
        "--min-mcap",
        type=int,
        default=500_000_000,
        help="Minimum market cap in USD (default: 500000000 = $500M)",
    )
    args = parser.parse_args()

    # Fetch full NASDAQ ticker list and filter by market cap
    logger.info("Fetching full NASDAQ ticker universe...")
    tickers = fetch_all_nasdaq_tickers(
        min_market_cap=args.min_mcap,
        cache_path=MCAP_CACHE_FILE,
    )
    logger.info(
        "Using %d tickers with market cap >= $%dM",
        len(tickers),
        args.min_mcap // 1_000_000,
    )

    # Load CIK map for EDGAR
    cik_map = load_cik_map()

    # Fetch macro data (11 years)
    logger.info("Fetching macro data...")
    macro_df = get_macro_data(period="11y")
    macro_df.index = pd.to_datetime(macro_df.index)
    logger.info("Macro data: %d rows, %s to %s",
                len(macro_df), macro_df.index.min().date(), macro_df.index.max().date())

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
        checkpoint_df["_date"] = pd.to_datetime(checkpoint_df["_date"])
        completed = set(checkpoint_df["Ticker"].unique())
        logger.info("Resuming: %d tickers done (%d rows)", len(completed), len(checkpoint_df))
    else:
        checkpoint_df = pd.DataFrame()
        completed = set()

    remaining = [t for t in tickers if t not in completed]
    logger.info("Remaining: %d / %d tickers", len(remaining), len(tickers))

    batch_rows = []
    errors = 0

    for i, ticker in enumerate(remaining):
        try:
            # 1. Fetch 10-year price history
            hist = get_stock_data(ticker, period="10y")
            if hist.empty or len(hist) < 126:
                errors += 1
                continue

            # 2. Compute technical features
            tech_df = compute_technical_features(hist)
            if tech_df.empty or len(tech_df) < 126:
                errors += 1
                continue

            # 3. Set DatetimeIndex for alignment
            if "Date" in tech_df.columns:
                tech_df["Date"] = pd.to_datetime(tech_df["Date"]).dt.tz_localize(None)
                tech_df = tech_df.set_index("Date")
            else:
                tech_df.index = pd.to_datetime(tech_df.index)

            # 4. Align macro data
            aligned_macro = align_macro_to_dates(macro_df, tech_df.index)
            merged = tech_df.copy()
            for col in aligned_macro.columns:
                merged[col] = aligned_macro[col].values

            # 5. Compute forward return (target)
            merged["Forward_Max_Return_3M"] = compute_forward_return(merged["Close"].values)

            # 6. Add EDGAR fundamentals
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

            # 7. Add insider transactions
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

            # 8. Add metadata
            merged["Ticker"] = ticker
            merged["_date"] = merged.index.strftime("%Y-%m-%d")
            merged = merged.reset_index(drop=True)

            # 9. Keep only the columns we need
            available_keep = [c for c in KEEP_COLS if c in merged.columns]
            merged = merged[available_keep]

            # 10. Drop rows without target
            merged = merged.dropna(subset=["Forward_Max_Return_3M"])

            batch_rows.append(merged)

        except Exception as e:
            logger.debug("Error processing %s: %s", ticker, e)
            errors += 1

        done = len(completed) + i + 1
        if done % 25 == 0:
            logger.info("Progress: %d / %d tickers (%.1f%%) | errors=%d",
                        done, len(tickers), done / len(tickers) * 100, errors)

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

    # Final concat
    if batch_rows:
        batch = pd.concat(batch_rows, ignore_index=True)
        if not checkpoint_df.empty:
            checkpoint_df = pd.concat([checkpoint_df, batch], ignore_index=True)
        else:
            checkpoint_df = batch

    result = checkpoint_df
    result["_date"] = pd.to_datetime(result["_date"])
    result = result.sort_values(["Ticker", "_date"]).reset_index(drop=True)

    logger.info("=" * 80)
    logger.info("FINAL 10-YEAR DATASET")
    logger.info("=" * 80)
    logger.info("Rows: %d | Columns: %d | Tickers: %d",
                len(result), len(result.columns), result["Ticker"].nunique())
    logger.info("Date range: %s to %s",
                result["_date"].min().date(), result["_date"].max().date())

    # Target stats
    target = result["Forward_Max_Return_3M"] >= 0.20
    logger.info("Class balance: %d positive (%.1f%%) / %d negative",
                target.sum(), target.mean() * 100, (~target).sum())

    # Date gap check
    all_dates = sorted(result["_date"].unique())
    gaps = []
    for j in range(1, len(all_dates)):
        gap = (all_dates[j] - all_dates[j - 1]).days
        if gap > 4:
            gaps.append((all_dates[j - 1], all_dates[j], gap))
    if gaps:
        logger.warning("Date gaps > 4 days: %d", len(gaps))
        for prev, nxt, g in gaps[:5]:
            logger.warning("  %s -> %s (%d days)", prev.date(), nxt.date(), g)
    else:
        logger.info("No date gaps > 4 days — clean")

    # Duplicate check
    dupes = result.duplicated(subset=["Ticker", "_date"], keep=False).sum()
    if dupes > 0:
        logger.warning("Duplicate (Ticker, _date) rows: %d — removing", dupes)
        result = result.drop_duplicates(subset=["Ticker", "_date"], keep="last")
    else:
        logger.info("No duplicate rows — clean")

    # NaN rates
    logger.info("NaN rates (>1%%):")
    meta = {"Ticker", "_date", "Forward_Max_Return_3M"}
    for col in sorted(result.columns):
        if col in meta:
            continue
        nan_pct = result[col].isna().mean() * 100
        if nan_pct > 1:
            logger.info("  %s: %.1f%%", col, nan_pct)

    result.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved to %s", OUTPUT_FILE)
    logger.info("Errors (tickers skipped): %d", errors)


if __name__ == "__main__":
    main()
