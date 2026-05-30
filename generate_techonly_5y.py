"""Generate a 5-year technical+macro only training dataset.

Uses only data sources with full historical coverage:
- Price/volume technical indicators (yfinance, 20+ years)
- Macroeconomic data (VIX, rates, SP500, gold, oil, dollar — 20+ years)

Excludes:
- Historical fundamentals (yfinance only has ~5 quarters)
- Earnings surprise (yfinance only has ~4 quarters)
- SEC EDGAR filings
- Google Trends
- Short interest, options flow, insider, Reddit sentiment

This gives us data covering COVID crash (2020), 2022 bear market,
and current bull — much more diverse market conditions.
"""

import gc
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from stock_predictor.data.yfinance_client import get_stock_data, compute_technical_features
from stock_predictor.data.macro_data import get_macro_data, align_macro_to_dates, MACRO_FEATURES
from stock_predictor.data.feature_engineering import TECHNICAL_FEATURES, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_CSV = "training_data_techonly_5y.csv"
CHECKPOINT_CSV = "training_data_techonly_5y_checkpoint.csv"
PROGRESS_FILE = "techonly_progress.txt"
CHECKPOINT_INTERVAL = 50


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(done):
    with open(PROGRESS_FILE, "w") as f:
        for t in sorted(done):
            f.write(t + "\n")


def main():
    # Load NASDAQ tickers
    if os.path.exists("nasdaq_all_tickers.txt"):
        with open("nasdaq_all_tickers.txt") as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        logger.error("nasdaq_all_tickers.txt not found")
        return

    logger.info("Fetching macroeconomic data (6y)...")
    macro_df = get_macro_data(period="10y")

    done = load_progress()
    logger.info("Total tickers: %d, already done: %d", len(tickers), len(done))

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_CSV) and len(done) > 0:
        all_rows_df = pd.read_csv(CHECKPOINT_CSV)
        all_rows = all_rows_df.to_dict("records")
        logger.info("Loaded checkpoint: %d rows", len(all_rows))
    else:
        all_rows = []

    processed_since_checkpoint = 0

    for ticker in tickers:
        if ticker in done:
            continue

        try:
            # 5y price history
            df = get_stock_data(ticker, period="5y")
            if df.empty or len(df) < 250:
                logger.warning("Skipping %s — insufficient history (%d rows)", ticker, len(df))
                done.add(ticker)
                continue

            df = compute_technical_features(df)

            # Compute target: max forward return within 63 trading days
            rolling_max = (
                df["Close"]
                .iloc[::-1]
                .rolling(window=63, min_periods=1)
                .max()
                .iloc[::-1]
            )
            df[TARGET_COLUMN] = rolling_max / df["Close"] - 1
            df.loc[df.index[-63:], TARGET_COLUMN] = np.nan

            # Filter valid rows
            valid_mask = df[TARGET_COLUMN].notna() & df["SMA_200"].notna()
            valid_df = df[valid_mask]

            if valid_df.empty:
                done.add(ticker)
                continue

            # Extract dates for macro alignment
            if "Date" in valid_df.columns:
                sample_dates = pd.to_datetime(valid_df["Date"])
                if sample_dates.dt.tz is not None:
                    sample_dates = sample_dates.dt.tz_localize(None)
            else:
                sample_dates = pd.to_datetime(valid_df.index)

            # Time-align macro data
            aligned_macro = align_macro_to_dates(macro_df, sample_dates)

            for i, (idx, row) in enumerate(valid_df.iterrows()):
                data_point = {"Ticker": ticker}
                date_val = sample_dates.iloc[i] if hasattr(sample_dates, "iloc") else sample_dates[i]
                data_point["_date"] = str(date_val)

                # Technical features
                for col in TECHNICAL_FEATURES:
                    data_point[col] = row.get(col, np.nan)

                # Macro features (time-aligned)
                for col in MACRO_FEATURES:
                    data_point[col] = (
                        aligned_macro[col].iloc[i]
                        if col in aligned_macro.columns
                        else np.nan
                    )

                data_point[TARGET_COLUMN] = row[TARGET_COLUMN]
                all_rows.append(data_point)

            logger.info("Processed %s: %d samples (total: %d)", ticker, len(valid_df), len(all_rows))

        except Exception:
            logger.exception("Error processing %s", ticker)

        done.add(ticker)
        processed_since_checkpoint += 1

        if processed_since_checkpoint >= CHECKPOINT_INTERVAL:
            logger.info("Checkpointing at %d tickers, %d rows...", len(done), len(all_rows))
            save_progress(done)
            pd.DataFrame(all_rows).to_csv(CHECKPOINT_CSV, index=False)
            processed_since_checkpoint = 0
            gc.collect()

    # Final save
    logger.info("Saving final dataset...")
    save_progress(done)
    result = pd.DataFrame(all_rows)
    result.to_csv(OUTPUT_CSV, index=False)
    logger.info("Done. %d rows, %d columns saved to %s", len(result), len(result.columns), OUTPUT_CSV)


if __name__ == "__main__":
    main()
