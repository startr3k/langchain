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

# Path to the cached training data (10-year full dataset, ~950 MB).
_TRAINING_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "training_data_10y_full.csv"

CSV_COLUMNS = [
    "date",
    "rank",
    "ticker",
    "close_price",
    "probability",
    "signal",
    "ensemble_score",
    "elite_pool_size",
    "cls_proba",
    "pred_mfd",
    "z_cls",
    "z_ltr",
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
    "rsi_14",
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
    save_to_csv: bool = True,
) -> pd.DataFrame:
    """Generate today's top-K stock picks.

    When *save_to_csv* is True (default, used by the scheduler), picks are
    only recorded to the CSV when the elite pool >= 75 (MIN_ELITE_POOL).
    When *save_to_csv* is False (used by Top Recommendations in Streamlit),
    picks are always returned for display regardless of pool size, and the
    CSV is never touched.

    Returns a DataFrame of the picks.
    """
    from stock_predictor.models.automl_model import StockReturnPredictor

    csv_path = Path(csv_path or DEFAULT_CSV_PATH)
    _ensure_csv(csv_path)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Check if today's picks already exist in CSV
    if save_to_csv:
        try:
            existing = pd.read_csv(csv_path)
            if "date" in existing.columns and not existing.empty and today_str in existing["date"].values:
                logger.info("Picks for %s already recorded — skipping.", today_str)
                return existing[existing["date"] == today_str]
        except Exception:
            logger.warning("Could not read existing CSV at %s — will regenerate.", csv_path)
            csv_path.unlink(missing_ok=True)
            _ensure_csv(csv_path)

    predictor = StockReturnPredictor()
    predictor.load()

    # ── Fast batch scoring using cached training data ─────────────────
    top_picks = _batch_score_from_cache(
        predictor,
        top_k=top_k,
    )

    if not top_picks:
        logger.warning("No valid predictions — pipeline produced 0 picks.")
        return pd.DataFrame(columns=CSV_COLUMNS)

    from stock_predictor.models.automl_model import MIN_ELITE_POOL

    pool_size = top_picks[0].get("elite_pool_size", 0)

    # When saving to CSV (scheduler), gate on pool >= MIN_ELITE_POOL
    if save_to_csv and pool_size < MIN_ELITE_POOL:
        logger.info(
            "Elite pool %d < %d — skipping CSV for %s (weak signal day)",
            pool_size, MIN_ELITE_POOL, today_str,
        )
        return pd.DataFrame(columns=CSV_COLUMNS)

    rows: list[dict] = []
    for rank, r in enumerate(top_picks, 1):
        ticker = r["ticker"]

        sentiment_data = _safe_sentiment(ticker)
        sentiment_score = sentiment_data.get("sentiment_mean_polarity", 0.0)
        sentiment_mentions = sentiment_data.get("sentiment_total_mentions", 0)

        feature_row = r.get("_feature_row")
        shap_str = _get_shap_explanation(predictor, ticker, feature_row=feature_row)

        close_price = r.get("last_close")
        market_cap = r.get("market_cap")
        sector = r.get("sector", "N/A")

        row = {
            "date": today_str,
            "rank": rank,
            "ticker": ticker,
            "close_price": round(close_price, 2) if close_price else None,
            "probability": round(r.get("probability_gain", 0), 4),
            "signal": r.get("signal", "HOLD"),
            "ensemble_score": round(r.get("ensemble_score", r.get("probability_gain", 0)), 4),
            "elite_pool_size": r.get("elite_pool_size", 0),
            "cls_proba": r.get("cls_proba", 0),
            "pred_mfd": r.get("pred_mfd", 0),
            "z_cls": r.get("z_cls", 0),
            "z_ltr": r.get("z_ltr", 0),
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
            "rsi_14": round(r.get("rsi_14", 0), 2) if r.get("rsi_14") else None,
            "max_upside_pct": None,
            "hit_20pct": None,
            "ground_truth_date": None,
        }
        rows.append(row)

    df_new = pd.DataFrame(rows)
    # Ensure column order matches CSV_COLUMNS to prevent data corruption
    df_new = df_new.reindex(columns=CSV_COLUMNS)

    # Only write to CSV when save_to_csv=True and pool gate passed
    if save_to_csv:
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
        logger.info("Recorded %d picks for %s", len(rows), today_str)
    else:
        logger.info("Generated %d picks for display (pool=%d, not saved to CSV)", len(rows), pool_size)

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
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=CSV_COLUMNS)

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
    if df.empty or "date" not in df.columns or "hit_20pct" not in df.columns:
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

def _batch_score_from_cache(
    predictor,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Score all tickers from the cached training CSV in one batch.

    Steps:
    1. Load the latest row per ticker from training_data_10y_full.csv.
    2. Run ``predict_batch`` on the whole DataFrame (~670 rows).
    3. Rank by ensemble score and return the top-K results.

    Falls back to the slow per-ticker loop if the cache is missing.
    """
    if not _TRAINING_CSV_PATH.exists():
        logger.warning("Training cache not found at %s — falling back to slow scan", _TRAINING_CSV_PATH)
        return _slow_scan_tickers(predictor, top_k=top_k)

    logger.info("Loading cached training data from %s ...", _TRAINING_CSV_PATH)
    cache_df = pd.read_csv(_TRAINING_CSV_PATH)

    # Keep only the latest date per ticker — handle both '_date' and 'date' column names
    date_col = "_date" if "_date" in cache_df.columns else "date"
    if date_col not in cache_df.columns:
        logger.warning("Training CSV has no date column — falling back to slow scan")
        return _slow_scan_tickers(predictor, top_k=top_k)
    cache_df["_date"] = pd.to_datetime(cache_df[date_col])
    latest_idx = cache_df.groupby("Ticker")["_date"].idxmax()
    latest_df = cache_df.loc[latest_idx].copy().reset_index(drop=True)
    logger.info(
        "Loaded %d tickers, latest dates: %s to %s",
        len(latest_df),
        latest_df["_date"].min().date(),
        latest_df["_date"].max().date(),
    )

    # Warn if cache is stale (>7 days old)
    max_date = latest_df["_date"].max()
    staleness = (pd.Timestamp.now() - max_date).days
    if staleness > 7:
        logger.warning(
            "Training cache is %d days old (latest: %s) — predictions may be stale",
            staleness,
            max_date.date(),
        )

    tickers = latest_df["Ticker"].values

    # Batch score all tickers at once
    scores = predictor.predict_batch(
        latest_df,
        tickers=pd.Series(tickers),
        apply_adjustments=True,
    )

    # Extract per-stock stage details from 4-stage pipeline
    batch_details = getattr(predictor, "_last_batch_details", None)
    if batch_details:
        elite_pool_size = batch_details["elite_pool_size"]
        cls_proba = batch_details["cls_proba"]
        pred_mfd = batch_details["pred_mfd"]
        z_cls = batch_details["z_cls"]
        z_ltr = batch_details["z_ltr"]
    else:
        elite_pool_size = int((scores > 0).sum())
        cls_proba = np.zeros(len(scores))
        pred_mfd = np.zeros(len(scores))
        z_cls = np.zeros(len(scores))
        z_ltr = np.zeros(len(scores))
    logger.info("Elite pool size: %d / %d tickers", elite_pool_size, len(scores))

    # Build results with scores
    scored = pd.DataFrame({
        "ticker": tickers,
        "ensemble_score": scores,
        "cls_proba": cls_proba,
        "pred_mfd": pred_mfd,
        "z_cls": z_cls,
        "z_ltr": z_ltr,
    })

    # Merge back ALL feature columns (needed for SHAP explanations + output)
    for col in latest_df.columns:
        if col not in scored.columns and col not in ("_date",):
            scored[col] = latest_df[col].values

    # Sort by ensemble score descending
    scored = scored.sort_values("ensemble_score", ascending=False).reset_index(drop=True)

    import yfinance as yf

    all_tickers = scored["ticker"].tolist()

    # Take top_k from the scored list (no market cap filtering)
    top_candidates = scored.head(top_k)
    candidate_tickers = top_candidates["ticker"].tolist()

    # ── Batch-fetch close prices via yf.download (single request) ────
    close_prices: dict[str, float | None] = {}
    try:
        price_df = yf.download(
            candidate_tickers,
            period="5d",
            progress=False,
            threads=True,
        )
        if not price_df.empty:
            if isinstance(price_df.columns, pd.MultiIndex):
                # Multi-ticker download returns MultiIndex columns
                for t in candidate_tickers:
                    try:
                        closes = price_df[("Close", t)].dropna()
                        close_prices[t] = round(float(closes.iloc[-1]), 2) if not closes.empty else None
                    except (KeyError, IndexError):
                        close_prices[t] = None
            else:
                # Single ticker download returns flat columns
                closes = price_df["Close"].dropna()
                if not closes.empty and len(candidate_tickers) == 1:
                    close_prices[candidate_tickers[0]] = round(float(closes.iloc[-1]), 2)
        logger.info("Batch-fetched close prices for %d tickers", len(close_prices))
    except Exception as e:
        logger.warning("Batch price download failed: %s", e)

    # ── Fetch sector + market cap per ticker (with retry + delay) ────
    import time

    sector_cache: dict[str, str] = {}
    mcap_cache: dict[str, int] = {}

    for t in candidate_tickers:
        for attempt in range(3):
            try:
                info = yf.Ticker(t).info
                sector_cache[t] = info.get("sector", "N/A")
                mcap_cache[t] = info.get("marketCap", 0) or 0
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    sector_cache[t] = "N/A"
                    mcap_cache[t] = 0

    # ── Build final picks list ───────────────────────────────────────
    top_picks: list[dict] = []
    for _, row in top_candidates.iterrows():
        ticker = row["ticker"]

        regime_conf = predictor.predict_regime_confidence(
            {col: row.get(col, 0.0) for col in predictor.feature_names}
            if hasattr(predictor, "feature_names") else {}
        )
        cal_factor = predictor.ticker_calibration.get(ticker, 1.0)

        feature_row = {col: row.get(col, 0.0) for col in predictor.feature_names if col in row.index}

        close_price = close_prices.get(ticker)
        sector = sector_cache.get(ticker, "N/A")
        mcap = mcap_cache.get(ticker, 0)

        top_picks.append({
            "ticker": ticker,
            "probability_gain": round(float(row["ensemble_score"]), 4),
            "signal": "BUY",
            "ensemble_score": float(row["ensemble_score"]),
            "elite_pool_size": elite_pool_size,
            "cls_proba": round(float(row.get("cls_proba", 0)), 4),
            "pred_mfd": round(float(row.get("pred_mfd", 0)), 4),
            "z_cls": round(float(row.get("z_cls", 0)), 3),
            "z_ltr": round(float(row.get("z_ltr", 0)), 3),
            "ltr_score": 0.0,
            "classification_score": 0.0,
            "volume_surge_3d": round(float(row.get("Volume_Surge_3d", 0)), 2) if pd.notna(row.get("Volume_Surge_3d")) else None,
            "regime_confidence": round(regime_conf, 3),
            "ticker_calibration": cal_factor,
            "volatility_20d": round(float(row.get("Volatility_20d", 0)), 4) if pd.notna(row.get("Volatility_20d")) else None,
            "rsi_14": round(float(row.get("RSI_14", 0)), 2) if pd.notna(row.get("RSI_14")) else None,
            "market_cap": mcap,
            "sector": sector,
            "last_close": close_price,
            "_feature_row": feature_row,
        })
        logger.info("Pick %d: %s (score=%.4f, close=$%s, sector=%s, mcap=$%.1fB)",
                     len(top_picks), ticker, row["ensemble_score"],
                     close_price, sector, mcap / 1e9)

    logger.info("Batch scoring complete: %d picks selected from %d tickers", len(top_picks), len(all_tickers))
    return top_picks


def _slow_scan_tickers(
    predictor,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Fallback: scan tickers one-by-one via yFinance (slow)."""
    from stock_predictor.data.yfinance_client import NASDAQ_TOP_TICKERS

    tickers_to_scan: list[str] = list(NASDAQ_TOP_TICKERS)
    try:
        sp500 = _get_major_index_tickers()
        tickers_to_scan = list(set(tickers_to_scan) | set(sp500))
    except Exception:
        logger.warning("Could not fetch index tickers, using NASDAQ list only")

    logger.info("Slow scan: %d tickers via yFinance...", len(tickers_to_scan))

    results: list[dict] = []
    for ticker in tickers_to_scan:
        try:
            result = predictor.predict_ticker(ticker)
            if result.get("probability_gain") is not None:
                results.append(result)
        except Exception:
            continue

    results.sort(key=lambda x: x.get("probability_gain", 0), reverse=True)
    return results[:top_k]


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


def _get_shap_explanation(
    predictor, ticker: str, *, feature_row: dict | None = None,
) -> str:
    """Get top SHAP features as a compact string.

    Uses the pre-computed feature row from the training cache when available,
    falling back to building features from yFinance if not.
    """
    try:
        if feature_row:
            explanation = predictor.explain_prediction(feature_row)
        else:
            # Fallback: build features from scratch (slow)
            from stock_predictor.data.yfinance_client import get_stock_data
            from stock_predictor.data.feature_engineering import compute_features

            stock_data = get_stock_data(ticker, period="2y")
            if stock_data is None or stock_data.empty:
                return ""
            features_df = compute_features(stock_data, ticker)
            if features_df.empty:
                return ""
            explanation = predictor.explain_prediction(features_df.iloc[[-1]])

        if explanation:
            parts = [
                f"{f['feature']}={f['shap_value']:+.3f}"
                for f in explanation[:5]
            ]
            return "; ".join(parts)
    except Exception as exc:
        logger.debug("SHAP explanation failed for %s: %s", ticker, exc)
    return ""


def _get_major_index_tickers() -> list[str]:
    """Return a list of tickers from NASDAQ (fallback for eligible ticker fetch)."""
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
