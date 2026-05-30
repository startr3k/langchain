"""MLOps pipeline for recording and evaluating daily top-10 stock picks.

Records daily predictions to a CSV with model probability, SHAP explanations,
sentiment scores, volume surge, and other metadata.  Provides ground-truth
evaluation by checking whether each pick achieved ≥20% upside from the
recorded closing price within 3 months.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default path for the daily picks CSV.
DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "daily_picks.csv"

CSV_COLUMNS = [
    "date",
    "rank",
    "ticker",
    "close_price",
    "probability",
    "signal",
    "ensemble_score",
    "ltr_score",
    "classification_score",
    "volume_surge_3d",
    "regime_confidence",
    "ticker_calibration",
    "volatility_20d",
    "sentiment_score",
    "sentiment_mentions",
    "shap_top_features",
    "market_cap",
    "sector",
    "max_upside_pct",
    "hit_20pct",
    "ground_truth_date",
]


def _ensure_csv(path: Path) -> None:
    """Create the CSV with headers if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def run_daily_picks(
    *,
    csv_path: Path | str | None = None,
    top_k: int = 10,
    min_market_cap: float = 100_000_000,
) -> pd.DataFrame:
    """Generate today's top-K stock picks and append to the CSV.

    Returns a DataFrame of the picks that were recorded.
    """
    from stock_predictor.data.sentiment import get_sentiment_features
    from stock_predictor.data.yfinance_client import get_stock_data, get_stock_info
    from stock_predictor.models.automl_model import StockReturnPredictor

    csv_path = Path(csv_path or DEFAULT_CSV_PATH)
    _ensure_csv(csv_path)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Check if today's picks already exist
    existing = pd.read_csv(csv_path)
    if not existing.empty and today_str in existing["date"].values:
        logger.info("Picks for %s already recorded — skipping.", today_str)
        return existing[existing["date"] == today_str]

    predictor = StockReturnPredictor()
    predictor.load()

    # Scan a broad universe
    from stock_predictor.data.yfinance_client import NASDAQ_TOP_TICKERS

    # Build a larger ticker list from the training data cache if available
    tickers_to_scan: list[str] = list(NASDAQ_TOP_TICKERS)

    # Also add S&P 500 / Dow tickers from a known list
    try:
        sp500 = _get_major_index_tickers()
        tickers_to_scan = list(set(tickers_to_scan) | set(sp500))
    except Exception:
        logger.warning("Could not fetch index tickers, using NASDAQ list only")

    logger.info("Scanning %d tickers for daily picks...", len(tickers_to_scan))

    results: list[dict] = []
    for ticker in tickers_to_scan:
        try:
            result = predictor.predict_ticker(ticker, min_market_cap=min_market_cap)
            if result.get("probability_gain") is not None:
                results.append(result)
        except Exception:
            continue

    if not results:
        logger.warning("No valid predictions — pipeline produced 0 picks.")
        return pd.DataFrame(columns=CSV_COLUMNS)

    # Sort by probability (or ensemble score if available)
    results.sort(key=lambda x: x.get("probability_gain", 0), reverse=True)
    top_picks = results[:top_k]

    rows: list[dict] = []
    for rank, r in enumerate(top_picks, 1):
        ticker = r["ticker"]

        # Fetch sentiment
        sentiment_data = _safe_sentiment(ticker)
        sentiment_score = sentiment_data.get("sentiment_mean_polarity", 0.0)
        sentiment_mentions = sentiment_data.get("sentiment_total_mentions", 0)

        # SHAP explanations
        shap_str = _get_shap_explanation(predictor, ticker)

        # Stock info
        info = _safe_stock_info(ticker)
        market_cap = info.get("marketCap", info.get("market_cap"))
        sector = info.get("sector", "N/A")

        # Get closing price
        close_price = r.get("last_close")
        if close_price is None:
            try:
                df_price = get_stock_data(ticker, period="5d")
                if df_price is not None and not df_price.empty:
                    close_price = float(df_price["Close"].iloc[-1])
            except Exception:
                close_price = None

        row = {
            "date": today_str,
            "rank": rank,
            "ticker": ticker,
            "close_price": round(close_price, 2) if close_price else None,
            "probability": round(r.get("probability_gain", 0), 4),
            "signal": r.get("signal", "HOLD"),
            "ensemble_score": round(r.get("ensemble_score", r.get("probability_gain", 0)), 4),
            "ltr_score": round(r.get("ltr_score", 0), 4),
            "classification_score": round(r.get("classification_score", r.get("probability_gain", 0)), 4),
            "volume_surge_3d": round(r.get("volume_surge_3d", 0), 2) if r.get("volume_surge_3d") else None,
            "regime_confidence": round(r.get("regime_confidence", 0.5), 4),
            "ticker_calibration": round(r.get("ticker_calibration", 1.0), 4),
            "volatility_20d": round(r.get("volatility_20d", 0), 4) if r.get("volatility_20d") else None,
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_mentions": sentiment_mentions,
            "shap_top_features": shap_str,
            "market_cap": market_cap,
            "sector": sector,
            "max_upside_pct": None,
            "hit_20pct": None,
            "ground_truth_date": None,
        }
        rows.append(row)

    # Append to CSV
    df_new = pd.DataFrame(rows)
    df_new.to_csv(csv_path, mode="a", header=False, index=False)
    logger.info("Recorded %d picks for %s", len(rows), today_str)
    return df_new


def evaluate_ground_truth(
    *,
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Check historical picks for ≥20% upside and update ground truth.

    For each recorded pick, fetch the max high price from the pick date to
    now (or 3 months later, whichever is earlier).  If the max upside from
    the recorded close_price is ≥20%, mark as a hit.  Only overwrites
    existing values if the latest upside is higher.
    """
    from stock_predictor.data.yfinance_client import get_stock_data

    csv_path = Path(csv_path or DEFAULT_CSV_PATH)
    if not csv_path.exists():
        logger.warning("No picks CSV found at %s", csv_path)
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    today = datetime.now()
    updated = False

    for idx, row in df.iterrows():
        pick_date_str = row["date"]
        ticker = row["ticker"]
        close_price = row.get("close_price")

        if pd.isna(close_price) or close_price is None or close_price <= 0:
            continue

        pick_date = pd.to_datetime(pick_date_str)
        end_date = min(pick_date + timedelta(days=90), today)

        # Skip if evaluation window hasn't started yet (same day)
        if (today - pick_date).days < 1:
            continue

        try:
            df_price = get_stock_data(
                ticker,
                start=pick_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
            if df_price is None or df_price.empty:
                continue

            max_high = float(df_price["High"].max())
            max_upside = (max_high - close_price) / close_price * 100

            # Only overwrite if new upside is higher
            existing_upside = row.get("max_upside_pct")
            if pd.notna(existing_upside) and existing_upside >= max_upside:
                continue

            df.at[idx, "max_upside_pct"] = round(max_upside, 2)
            df.at[idx, "hit_20pct"] = 1 if max_upside >= 20 else 0
            df.at[idx, "ground_truth_date"] = today.strftime("%Y-%m-%d")
            updated = True

        except Exception:
            logger.exception("Error evaluating ground truth for %s on %s", ticker, pick_date_str)

    if updated:
        df.to_csv(csv_path, index=False)
        logger.info("Updated ground truth in %s", csv_path)

    return df


def get_precision_over_time(
    *,
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Compute daily top-10 precision over time from the picks CSV.

    Returns a DataFrame with columns: date, total_picks, hits, precision.
    """
    csv_path = Path(csv_path or DEFAULT_CSV_PATH)
    if not csv_path.exists():
        return pd.DataFrame(columns=["date", "total_picks", "hits", "precision"])

    df = pd.read_csv(csv_path)
    if df.empty or "hit_20pct" not in df.columns:
        return pd.DataFrame(columns=["date", "total_picks", "hits", "precision"])

    # Only include dates where ground truth has been evaluated
    evaluated = df[df["hit_20pct"].notna()].copy()
    if evaluated.empty:
        return pd.DataFrame(columns=["date", "total_picks", "hits", "precision"])

    result = (
        evaluated.groupby("date")
        .agg(
            total_picks=("ticker", "count"),
            hits=("hit_20pct", "sum"),
        )
        .reset_index()
    )
    result["precision"] = result["hits"] / result["total_picks"]
    result = result.sort_values("date")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_sentiment(ticker: str) -> dict:
    """Fetch sentiment, returning empty dict on failure."""
    try:
        from stock_predictor.data.sentiment import get_sentiment_features
        return get_sentiment_features(ticker)
    except Exception:
        logger.debug("Sentiment fetch failed for %s", ticker)
        return {}


def _safe_stock_info(ticker: str) -> dict:
    """Fetch stock info, returning empty dict on failure."""
    try:
        from stock_predictor.data.yfinance_client import get_stock_info
        return get_stock_info(ticker)
    except Exception:
        return {}


def _get_shap_explanation(predictor, ticker: str) -> str:
    """Get top SHAP features as a compact string."""
    try:
        explanation = predictor.explain_prediction(ticker)
        if explanation and "top_features" in explanation:
            features = explanation["top_features"][:5]
            parts = [f"{f['feature']}={f['contribution']:+.3f}" for f in features]
            return "; ".join(parts)
    except Exception:
        pass
    return ""


def _get_major_index_tickers() -> list[str]:
    """Return a list of tickers from major US indices (S&P 500, Dow, NASDAQ-100)."""
    # Use yfinance or a static list — prefer fetching dynamically
    import yfinance as yf

    tickers: set[str] = set()

    for index_ticker in ["^GSPC", "^DJI", "^NDX"]:
        try:
            idx = yf.Ticker(index_ticker)
            # yfinance doesn't directly expose components for all indices,
            # so we fall back to a curated list
        except Exception:
            pass

    # Curated S&P 500 + Dow + NASDAQ-100 tickers (top ~150)
    major_tickers = [
        # Dow 30
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
        # Top S&P 500 / NASDAQ-100 additions
        "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST", "NFLX", "AMD",
        "ADBE", "PEP", "TMUS", "INTU", "CMCSA", "TXN", "QCOM", "AMAT", "ISRG",
        "BKNG", "LRCX", "SBUX", "VRTX", "MU", "ADI", "GILD", "MDLZ", "PANW",
        "REGN", "KLAC", "SNPS", "CDNS", "PYPL", "MELI", "CRWD", "MAR", "CTAS",
        "ABNB", "ORLY", "MRVL", "FTNT", "CEG", "DASH", "WDAY", "MNST",
        "BRK-B", "LLY", "XOM", "UNP", "RTX", "LOW", "SPGI", "BLK", "SCHW",
        "C", "BMY", "PFE", "ABBV", "TMO", "DHR", "SYK", "ZTS", "BDX", "CI",
        "SO", "DUK", "NEE", "AEP", "D", "SRE", "EXC", "ED", "WEC", "ES",
        "PLTR", "SOFI", "RIVN", "LCID", "MARA", "COIN", "HOOD", "ARM", "SMCI",
        "GME", "AMC", "SNOW", "DDOG", "ZS", "NET", "MDB", "SHOP", "SQ", "ROKU",
    ]
    tickers.update(major_tickers)
    return list(tickers)
