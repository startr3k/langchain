"""LangChain tools for the stock recommendation agent.

Five tools:
1. YFinanceTool — fetches market data and fundamentals
2. SocialMediaListenerTool — fetches sentiment from Reddit / Finviz
3. StockPredictorTool — runs the AutoML model to predict 3-month returns
4. ScanTrendingStocksTool — scans trending stocks from social media
5. ScanFullUniverseTool — scans all 670 tickers from the training dataset
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from stock_predictor.data.sentiment import (
    get_sentiment_features,
    get_sentiment_summary,
    get_trending_tickers_from_social,
)
from stock_predictor.data.yfinance_client import (
    compute_technical_features,
    get_fundamentals_features,
    get_nasdaq_trending_tickers,
    get_stock_data,
    get_stock_info,
)
from stock_predictor.models.automl_model import StockReturnPredictor

logger = logging.getLogger(__name__)

# Shared predictor instance (loaded lazily)
_predictor: StockReturnPredictor | None = None


def _get_predictor() -> StockReturnPredictor:
    global _predictor
    if _predictor is None or not _predictor.is_trained:
        _predictor = StockReturnPredictor()
        try:
            _predictor.load()
        except FileNotFoundError:
            logger.warning("No saved model found — predictions will fail until trained.")
    return _predictor


@tool
def yfinance_tool(ticker: str) -> str:
    """Fetch stock market data and fundamentals from YFinance for a given ticker.

    Use this tool to get current price data, technical indicators, and fundamental
    metrics (P/E ratio, market cap, revenue growth, etc.) for any stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA').

    Returns:
        JSON string with stock info, latest technical data, and fundamentals.
    """
    result: dict = {"ticker": ticker}

    # Company info
    info = get_stock_info(ticker)
    result["company_info"] = info

    # Latest technical data
    df = get_stock_data(ticker, period="6mo")
    if not df.empty:
        df = compute_technical_features(df)
        latest = df.iloc[-1]
        result["latest_price"] = {
            "close": round(float(latest["Close"]), 2),
            "volume": int(latest["Volume"]),
        }
        technicals = {}
        for col in [
            "RSI_14", "MACD", "MACD_Signal", "BB_Position", "BB_Width",
            "Volatility_20d", "Volume_Ratio", "Price_to_SMA_20",
            "Price_to_SMA_50", "Price_to_SMA_200", "ATR_14",
            "Return_1d", "Return_5d", "Return_20d", "Return_60d",
        ]:
            val = latest.get(col)
            if val is not None and not (isinstance(val, float) and (val != val)):
                technicals[col] = round(float(val), 4)
        result["technical_indicators"] = technicals

    # Fundamentals
    fundamentals = get_fundamentals_features(ticker)
    result["fundamentals"] = fundamentals

    return json.dumps(result, indent=2, default=str)


@tool
def social_media_listener_tool(ticker: str) -> str:
    """Fetch social media sentiment for a stock from Reddit and Finviz.

    Use this tool to understand what retail investors and the public think about
    a stock. It aggregates sentiment from multiple social media and news sources.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA').

    Returns:
        Detailed sentiment summary including polarity scores, mention counts,
        and bullish/bearish ratios from Reddit and Finviz news.
    """
    features = get_sentiment_features(ticker)
    summary = get_sentiment_summary(ticker, features=features)

    trending = get_trending_tickers_from_social()
    is_trending = ticker in trending

    output_lines = [
        summary,
        "",
        f"Currently trending on social media: {'YES' if is_trending else 'No'}",
    ]
    if trending:
        output_lines.append(f"Top trending tickers: {', '.join(trending[:10])}")

    output_lines.extend([
        "",
        "Raw sentiment features (for model input):",
        json.dumps(features, indent=2),
    ])

    return "\n".join(output_lines)


@tool
def stock_predictor_tool(ticker: str, min_market_cap_millions: float = 100) -> str:
    """Predict the 3-month forward return for a stock using the trained AutoML model.

    This tool combines YFinance data and social media sentiment features, then
    runs them through the trained prediction model. Use this AFTER gathering
    data from the YFinance and Social Media Listener tools.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA').
        min_market_cap_millions: Minimum market cap in millions of dollars.
            Default 100 ($100M). Use 1000 for high-conviction large-cap mode.

    Returns:
        JSON with the predicted 3-month return percentage and model confidence.
    """
    predictor = _get_predictor()

    if not predictor.is_trained:
        return json.dumps({
            "error": "Model not trained yet. Please train the model first.",
            "ticker": ticker,
        })

    result = predictor.predict_ticker(
        ticker, min_market_cap=min_market_cap_millions * 1_000_000
    )

    # Add feature importance context
    importance = predictor.get_feature_importance(top_n=10)
    if importance:
        result["top_features"] = [
            {"feature": name, "importance": round(imp, 4)}
            for name, imp in importance
        ]

    return json.dumps(result, indent=2, default=str)


@tool
def scan_trending_stocks_tool(top_n: int = 10) -> str:
    """Scan trending NASDAQ stocks and predict which will gain >=20% in 3 months.

    Identifies trending stocks from social media and runs the ensemble
    model (classification + LTR ranking) on each to find
    high-probability candidates.

    Args:
        top_n: Number of trending stocks to scan (default 10).

    Returns:
        JSON with predicted probabilities for each trending stock, sorted by
        probability of >=20% gain.
    """
    predictor = _get_predictor()

    if not predictor.is_trained:
        return json.dumps({
            "error": "Model not trained yet. Please train the model first.",
        })

    # Get trending tickers from social media
    social_trending = get_trending_tickers_from_social()
    nasdaq_trending = get_nasdaq_trending_tickers(top_n=top_n)

    # Combine and deduplicate
    all_tickers = list(dict.fromkeys(social_trending + nasdaq_trending))[:top_n]

    results = []
    for ticker in all_tickers:
        try:
            prediction = predictor.predict_ticker(ticker)
            if prediction.get("probability_gain") is not None:
                results.append(prediction)
        except Exception:
            logger.warning("Failed to predict for %s", ticker)

    results.sort(key=lambda x: x.get("probability_gain", -1), reverse=True)

    return json.dumps(
        {
            "trending_stocks_scanned": len(results),
            "predictions": results,
            "tickers_sourced_from": "Reddit + NASDAQ top by volume",
        },
        indent=2,
        default=str,
    )


@tool
def scan_full_universe_tool(top_n: int = 10, min_market_cap_billions: float = 1.0) -> str:
    """Scan ALL 670 tickers (NASDAQ + NYSE tech) from the training dataset and rank by predicted returns.

    Uses the cached training data to batch-score every ticker in ~5 seconds,
    then filters by market cap and returns the top-N ranked picks. This is
    the same logic as the daily picks pipeline.

    Use this tool when the user wants comprehensive "best picks" across the
    entire stock universe — not just trending stocks.

    Args:
        top_n: Number of top picks to return (default 10).
        min_market_cap_billions: Minimum market cap filter in billions (default 1.0).

    Returns:
        JSON with the top-N ranked stocks by ensemble score, including
        ticker, ensemble score, and probability of >=20% gain.
    """
    from pathlib import Path

    import pandas as pd

    predictor = _get_predictor()

    if not predictor.is_trained:
        return json.dumps({
            "error": "Model not trained yet. Please train the model first.",
        })

    training_csv = Path(__file__).resolve().parent.parent.parent / "training_data_10y_full.csv"
    if not training_csv.exists():
        return json.dumps({
            "error": "Training data CSV not found. Cannot perform full universe scan.",
        })

    cache_df = pd.read_csv(training_csv)

    # Keep only the latest date per ticker
    date_col = "_date" if "_date" in cache_df.columns else "date"
    if date_col not in cache_df.columns:
        return json.dumps({"error": "Training CSV has no date column."})
    cache_df["_date"] = pd.to_datetime(cache_df[date_col])
    latest_idx = cache_df.groupby("Ticker")["_date"].idxmax()
    latest_df = cache_df.loc[latest_idx].copy().reset_index(drop=True)

    tickers = latest_df["Ticker"].values

    # Batch score all tickers
    scores = predictor.predict_batch(
        latest_df,
        tickers=pd.Series(tickers),
        apply_adjustments=True,
    )

    scored = pd.DataFrame({
        "ticker": tickers,
        "ensemble_score": scores,
    })
    scored = scored.sort_values("ensemble_score", ascending=False).reset_index(drop=True)

    # Filter by eligible ticker universe (>= min market cap)
    try:
        from stock_predictor.pipeline.social_listener import get_eligible_tickers
        eligible_set = get_eligible_tickers()
        scored = scored[scored["ticker"].isin(eligible_set)].reset_index(drop=True)
    except Exception:
        logger.warning("Could not load eligible ticker cache — returning unfiltered results")

    top_picks = scored.head(top_n)

    results = []
    for _, row in top_picks.iterrows():
        results.append({
            "ticker": row["ticker"],
            "ensemble_score": round(float(row["ensemble_score"]), 4),
            "rank": len(results) + 1,
        })

    return json.dumps(
        {
            "total_tickers_scored": len(scored),
            "top_picks": results,
            "min_market_cap_filter": f"${min_market_cap_billions}B",
            "source": "Full 670-ticker training universe (NASDAQ + NYSE tech)",
        },
        indent=2,
        default=str,
    )
