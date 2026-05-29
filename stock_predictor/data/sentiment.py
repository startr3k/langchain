"""Social media and earnings call sentiment analysis for stocks.

Aggregates sentiment signals from Reddit (via old.reddit.com scraping),
Finviz news, StockTwits, and earnings call transcripts (via DuckDuckGo +
web scraping) to generate sentiment features for stock prediction.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from textblob import TextBlob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reddit Sentiment (old.reddit.com scraping — no API key required)
# ---------------------------------------------------------------------------

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

SUBREDDITS = [
    "wallstreetbets", "stocks", "investing", "StockMarket",
    "options", "pennystocks", "Daytrading", "ValueInvesting",
]

_REQUEST_DELAY = 1.5  # seconds between requests to avoid rate limiting


def _get_user_agent() -> str:
    """Return a rotating user-agent string."""
    import random
    return random.choice(_USER_AGENTS)


def _analyze_text_sentiment(text: str) -> dict:
    """Return polarity and subjectivity using TextBlob."""
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }


def _scrape_old_reddit_search(
    subreddit: str,
    query: str,
    time_filter: str = "month",
    limit: int = 25,
) -> list[dict]:
    """Scrape search results from old.reddit.com for a subreddit.

    Args:
        subreddit: Subreddit name.
        query: Search query (ticker symbol).
        time_filter: Time window ('day', 'week', 'month', 'year').
        limit: Max posts to return.

    Returns:
        List of dicts with post data.
    """
    import time
    from bs4 import BeautifulSoup

    url = f"https://old.reddit.com/r/{subreddit}/search"
    params = {
        "q": query,
        "restrict_sr": "on",
        "sort": "relevance",
        "t": time_filter,
        "limit": str(limit),
    }
    headers = {"User-Agent": _get_user_agent()}
    posts: list[dict] = []

    try:
        time.sleep(_REQUEST_DELAY)
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(
                "old.reddit.com returned %d for r/%s query %s",
                resp.status_code, subreddit, query,
            )
            return posts

        soup = BeautifulSoup(resp.text, "html.parser")
        things = soup.find_all("div", class_="thing", attrs={"data-fullname": True})

        for thing in things[:limit]:
            title_el = thing.find("a", class_="title")
            title = title_el.get_text(strip=True) if title_el else ""

            score_el = thing.find("div", class_="score unvoted")
            score_text = score_el.get("title", "0") if score_el else "0"
            try:
                score = int(score_text)
            except (ValueError, TypeError):
                score = 0

            comments_el = thing.find("a", class_="comments")
            num_comments = 0
            if comments_el:
                comment_text = comments_el.get_text(strip=True)
                nums = re.findall(r"\d+", comment_text)
                if nums:
                    num_comments = int(nums[0])

            timestamp = thing.find("time")
            created_utc = 0.0
            if timestamp and timestamp.get("datetime"):
                try:
                    from datetime import timezone
                    dt = datetime.fromisoformat(timestamp["datetime"].replace("Z", "+00:00"))
                    created_utc = dt.replace(tzinfo=timezone.utc).timestamp()
                except (ValueError, TypeError):
                    pass

            posts.append({
                "source": "reddit",
                "subreddit": subreddit,
                "title": title,
                "score": score,
                "num_comments": num_comments,
                "created_utc": created_utc,
            })
    except Exception:
        logger.exception("Error scraping old.reddit.com for %s from r/%s", query, subreddit)

    return posts


def fetch_reddit_sentiment(
    ticker: str,
    limit: int = 100,
    time_filter: str = "month",
) -> list[dict]:
    """Fetch Reddit posts mentioning a ticker and compute sentiment.

    Scrapes old.reddit.com search results (no API key required).

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        limit: Max number of posts to fetch.
        time_filter: Time window ('day', 'week', 'month', 'year').

    Returns:
        List of dicts with post-level sentiment data.
    """
    posts: list[dict] = []
    per_sub = max(1, limit // len(SUBREDDITS))

    for subreddit in SUBREDDITS:
        raw_posts = _scrape_old_reddit_search(subreddit, ticker, time_filter, per_sub)
        for post in raw_posts:
            sentiment = _analyze_text_sentiment(post["title"])
            post["polarity"] = sentiment["polarity"]
            post["subjectivity"] = sentiment["subjectivity"]
            post["upvote_ratio"] = 0  # not available via scraping
            posts.append(post)

    return posts


# ---------------------------------------------------------------------------
# Finviz News Sentiment (public page scrape)
# ---------------------------------------------------------------------------

def fetch_finviz_sentiment(ticker: str) -> list[dict]:
    """Scrape recent news headlines from Finviz and compute sentiment.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        List of dicts with headline-level sentiment data.
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "StockPredictor/1.0"}
    results: list[dict] = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.warning("Finviz returned %d for %s", resp.status_code, ticker)
            return results

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
        news_table = soup.find(id="news-table")
        if not news_table:
            return results

        rows = news_table.find_all("tr")
        for row in rows[:30]:
            link = row.find("a")
            if not link:
                continue
            headline = link.text.strip()
            sentiment = _analyze_text_sentiment(headline)
            results.append(
                {
                    "source": "finviz",
                    "headline": headline,
                    "polarity": sentiment["polarity"],
                    "subjectivity": sentiment["subjectivity"],
                }
            )
    except Exception:
        logger.exception("Error fetching Finviz data for %s", ticker)

    return results


# ---------------------------------------------------------------------------
# StockTwits Sentiment (public API)
# ---------------------------------------------------------------------------

def fetch_stocktwits_sentiment(ticker: str) -> list[dict]:
    """Fetch StockTwits messages for a ticker and compute sentiment.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        List of dicts with message-level sentiment data.
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    results: list[dict] = []
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning("StockTwits returned %d for %s", resp.status_code, ticker)
            return results

        data = resp.json()
        messages = data.get("messages", [])
        for msg in messages:
            body = msg.get("body", "")
            st_sentiment = msg.get("entities", {}).get("sentiment", {})
            basic = st_sentiment.get("basic") if st_sentiment else None
            sentiment = _analyze_text_sentiment(body)
            results.append(
                {
                    "source": "stocktwits",
                    "body": body[:200],
                    "stocktwits_sentiment": basic,
                    "polarity": sentiment["polarity"],
                    "subjectivity": sentiment["subjectivity"],
                    "likes": msg.get("likes", {}).get("total", 0),
                }
            )
    except Exception:
        logger.exception("Error fetching StockTwits data for %s", ticker)

    return results


# ---------------------------------------------------------------------------
# Aggregate Sentiment Features
# ---------------------------------------------------------------------------

def get_sentiment_features(ticker: str) -> dict:
    """Aggregate sentiment from all sources into model-ready features.

    Sources: Reddit, Finviz news, StockTwits, and earnings call transcripts.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dictionary of aggregated sentiment features including source texts.
    """
    from stock_predictor.data.earnings_transcript import fetch_earnings_transcript

    reddit_posts = fetch_reddit_sentiment(ticker)
    finviz_news = fetch_finviz_sentiment(ticker)
    stocktwits_msgs = fetch_stocktwits_sentiment(ticker)
    transcript_data = fetch_earnings_transcript(ticker)

    all_polarities: list[float] = []
    all_subjectivities: list[float] = []

    # Collect source texts for display: (source, text, polarity)
    source_texts: list[tuple[str, str, float]] = []

    reddit_polarities: list[float] = []
    reddit_scores: list[float] = []
    reddit_comments: list[float] = []

    for post in reddit_posts:
        all_polarities.append(post["polarity"])
        all_subjectivities.append(post["subjectivity"])
        reddit_polarities.append(post["polarity"])
        reddit_scores.append(post["score"])
        reddit_comments.append(post["num_comments"])
        source_texts.append((
            "Reddit",
            post.get("title", post.get("text", ""))[:200],
            round(post["polarity"], 3),
        ))

    finviz_polarities: list[float] = []
    for item in finviz_news:
        all_polarities.append(item["polarity"])
        all_subjectivities.append(item["subjectivity"])
        finviz_polarities.append(item["polarity"])
        source_texts.append((
            "Finviz News",
            item.get("title", item.get("text", ""))[:200],
            round(item["polarity"], 3),
        ))

    stocktwits_polarities: list[float] = []
    stocktwits_bullish = 0
    stocktwits_bearish = 0
    for msg in stocktwits_msgs:
        all_polarities.append(msg["polarity"])
        all_subjectivities.append(msg["subjectivity"])
        stocktwits_polarities.append(msg["polarity"])
        if msg.get("stocktwits_sentiment") == "Bullish":
            stocktwits_bullish += 1
        elif msg.get("stocktwits_sentiment") == "Bearish":
            stocktwits_bearish += 1
        source_texts.append((
            "StockTwits",
            msg.get("body", msg.get("text", ""))[:200],
            round(msg["polarity"], 3),
        ))

    # Earnings call transcript sentiment
    transcript_polarity = transcript_data.get("transcript_polarity")
    if transcript_polarity is not None:
        all_polarities.append(transcript_polarity)
    # Add transcript source texts for display
    for src_text in transcript_data.get("transcript_source_texts", []):
        source_texts.append(src_text)

    features = {
        # Overall sentiment
        "sentiment_mean_polarity": float(np.mean(all_polarities)) if all_polarities else 0.0,
        "sentiment_std_polarity": float(np.std(all_polarities)) if all_polarities else 0.0,
        "sentiment_max_polarity": float(np.max(all_polarities)) if all_polarities else 0.0,
        "sentiment_min_polarity": float(np.min(all_polarities)) if all_polarities else 0.0,
        "sentiment_mean_subjectivity": float(np.mean(all_subjectivities)) if all_subjectivities else 0.0,
        "sentiment_total_mentions": len(all_polarities),
        # Reddit-specific
        "reddit_mention_count": len(reddit_posts),
        "reddit_mean_polarity": float(np.mean(reddit_polarities)) if reddit_polarities else 0.0,
        "reddit_mean_score": float(np.mean(reddit_scores)) if reddit_scores else 0.0,
        "reddit_total_comments": float(np.sum(reddit_comments)) if reddit_comments else 0.0,
        # Finviz news
        "finviz_mention_count": len(finviz_news),
        "finviz_mean_polarity": float(np.mean(finviz_polarities)) if finviz_polarities else 0.0,
        # StockTwits
        "stocktwits_mention_count": len(stocktwits_msgs),
        "stocktwits_mean_polarity": float(np.mean(stocktwits_polarities)) if stocktwits_polarities else 0.0,
        "stocktwits_bullish_count": stocktwits_bullish,
        "stocktwits_bearish_count": stocktwits_bearish,
        "stocktwits_bull_bear_ratio": (
            stocktwits_bullish / max(stocktwits_bearish, 1)
        ),
        # Earnings call transcript
        "transcript_sentiment": transcript_data.get("transcript_sentiment"),
        "transcript_polarity": transcript_data.get("transcript_polarity"),
        "transcript_url": transcript_data.get("transcript_url"),
        "transcript_date": transcript_data.get("transcript_date"),
        # All source texts for display
        "source_texts": source_texts,
    }

    return features


def get_sentiment_summary(ticker: str, features: dict | None = None) -> str:
    """Return a human-readable sentiment summary for a ticker.

    Args:
        ticker: Stock ticker symbol.
        features: Pre-computed sentiment features dict. If ``None``,
            calls :func:`get_sentiment_features` internally.

    Returns:
        Formatted string summarising social media sentiment.
    """
    if features is None:
        features = get_sentiment_features(ticker)
    polarity = features["sentiment_mean_polarity"]
    if polarity > 0.15:
        overall = "Strongly Positive"
    elif polarity > 0.05:
        overall = "Positive"
    elif polarity > -0.05:
        overall = "Neutral"
    elif polarity > -0.15:
        overall = "Negative"
    else:
        overall = "Strongly Negative"

    transcript_sent = features.get("transcript_sentiment")
    transcript_url = features.get("transcript_url")

    lines = [
        f"=== Sentiment for {ticker} ===",
        f"Overall Sentiment: {overall} (polarity={polarity:.3f})",
        f"Total mentions across sources: {features['sentiment_total_mentions']}",
        "",
        f"Reddit: {features['reddit_mention_count']} posts, "
        f"avg polarity={features['reddit_mean_polarity']:.3f}, "
        f"avg score={features['reddit_mean_score']:.1f}",
        f"Finviz News: {features['finviz_mention_count']} headlines, "
        f"avg polarity={features['finviz_mean_polarity']:.3f}",
        f"StockTwits: {features['stocktwits_mention_count']} messages, "
        f"bullish={features['stocktwits_bullish_count']}, "
        f"bearish={features['stocktwits_bearish_count']}, "
        f"bull/bear ratio={features['stocktwits_bull_bear_ratio']:.2f}",
        f"Earnings Call: sentiment={transcript_sent if transcript_sent is not None else 'N/A'}"
        + (f", url={transcript_url}" if transcript_url else ""),
    ]
    return "\n".join(lines)


def get_trending_tickers_from_social() -> list[str]:
    """Identify trending tickers from Reddit's popular investing subreddits.

    Scrapes hot posts from old.reddit.com (no API key required).

    Returns:
        List of ticker symbols mentioned most frequently.
    """
    import time
    from bs4 import BeautifulSoup

    ticker_pattern = re.compile(r"\b[A-Z]{2,5}\b")
    mention_counts: dict[str, int] = {}

    common_words = {
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
        "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS",
        "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID",
        "GET", "HIM", "LET", "SAY", "SHE", "TOO", "USE", "DAD", "MOM",
        "IMO", "CEO", "IPO", "ETF", "GDP", "USA", "NYSE", "SEC", "FDA",
        "FED", "ATH", "ATL", "DD", "SP", "PT", "EPS", "PE", "BS", "IV",
        "RSI", "TA", "WSB", "YOLO", "HODL", "FOMO", "FYI", "LOL", "OMG",
        "BUY", "PUT", "CALL", "HOLD", "SELL", "LONG", "HIGH", "LOW",
        "JUST", "LIKE", "THIS", "THAT", "WITH", "FROM", "THEY", "BEEN",
        "HAVE", "WILL", "MORE", "WHEN", "WHAT", "YOUR", "THAN", "THEM",
        "SOME", "VERY", "MOST", "MUCH", "EACH", "OVER", "ALSO", "BACK",
        "YEAR", "INTO", "GOOD", "MAKE", "TAKE", "EVEN", "ONLY", "COME",
        "MADE", "FIND", "HERE", "CASH", "EDIT", "LINK", "POST", "WEEK",
        "BEST", "NEXT", "LAST", "EVER",
    }
    valid_tickers = set(
        [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA",
            "AVGO", "COST", "NFLX", "AMD", "ADBE", "PEP", "CSCO",
            "TMUS", "INTC", "INTU", "CMCSA", "TXN", "QCOM", "AMGN",
            "AMAT", "ISRG", "HON", "BKNG", "LRCX", "SBUX", "VRTX",
            "MU", "ADI", "GILD", "MDLZ", "PANW", "REGN", "KLAC",
            "SNPS", "CDNS", "PYPL", "MELI", "CRWD", "MAR", "CTAS",
            "ABNB", "ORLY", "MRVL", "FTNT", "CEG", "DASH", "WDAY",
            "MNST", "GME", "AMC", "PLTR", "SOFI", "RIVN", "LCID",
            "MARA", "COIN", "HOOD", "ARM", "SMCI",
        ]
    )

    for subreddit in ["wallstreetbets", "stocks", "investing", "StockMarket"]:
        url = f"https://old.reddit.com/r/{subreddit}/hot"
        headers = {"User-Agent": _get_user_agent()}
        try:
            time.sleep(_REQUEST_DELAY)
            resp = requests.get(url, headers=headers, params={"limit": "50"}, timeout=15)
            if resp.status_code != 200:
                logger.warning(
                    "old.reddit.com returned %d for r/%s hot",
                    resp.status_code, subreddit,
                )
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            things = soup.find_all("div", class_="thing", attrs={"data-fullname": True})
            for thing in things:
                title_el = thing.find("a", class_="title")
                title = title_el.get_text(strip=True) if title_el else ""
                matches = ticker_pattern.findall(title)
                for match in matches:
                    if match in common_words:
                        continue
                    if match in valid_tickers:
                        mention_counts[match] = mention_counts.get(match, 0) + 1
        except Exception:
            logger.exception("Error scanning r/%s for trending tickers", subreddit)

    sorted_tickers = sorted(mention_counts.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_tickers[:20]]
