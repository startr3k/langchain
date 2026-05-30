"""Social media listener — surfaces the top-20 hottest stocks from Reddit,
StockTwits, and Finviz that are listed on Dow, S&P 500, or NASDAQ with
market cap >= $1B.

Updated daily; results include mention counts, sentiment, and source
breakdown.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from functools import lru_cache
from typing import Optional

import requests
from textblob import TextBlob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic index-ticker fetching (>= $1B market cap on Dow / S&P / NASDAQ)
# ---------------------------------------------------------------------------

# Fallback curated list in case dynamic fetch fails
_FALLBACK_TICKERS: set[str] = {
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
    "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
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
    "UBER", "LYFT", "RBLX", "PINS", "SNAP", "SPOT", "TTD",
    "DELL", "LULU", "NIO", "XPEV", "LI", "F", "GM",
    "LMT", "NOC", "GD", "GE", "DE", "EMR", "ROK", "ETN",
    "ACN", "ORCL", "SAP", "NOW", "HUBS", "VEEV", "TEAM",
    "OKTA", "CYBR", "COP", "EOG", "SLB", "HAL", "MPC", "VLO", "PSX",
    "BAC", "WFC", "MS", "BX", "KKR", "APO", "MA", "AFRM", "UPST",
    "T", "CHTR", "CL", "KMB", "EL",
    "HCA", "ELV", "HUM", "CVS",
}

MIN_MARKET_CAP = 1_000_000_000  # $1B


@lru_cache(maxsize=1)
def _fetch_index_tickers_cached() -> frozenset[str]:
    """Fetch tickers from S&P 500, Dow, and NASDAQ-100 via Wikipedia/yfinance.

    Uses Wikipedia tables as a reliable public source for index constituents,
    then filters to >= $1B market cap via yfinance.  Results are cached for
    the lifetime of the process (typically one Streamlit session).
    """
    import pandas as pd

    tickers: set[str] = set()

    # S&P 500 from Wikipedia
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            match="Symbol",
        )
        if tables:
            sp500 = tables[0]
            col = "Symbol" if "Symbol" in sp500.columns else sp500.columns[0]
            for sym in sp500[col]:
                t = str(sym).strip().replace(".", "-")
                if t:
                    tickers.add(t)
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
    except Exception:
        logger.warning("Could not fetch S&P 500 list from Wikipedia")

    # NASDAQ-100 from Wikipedia
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            match="Ticker",
        )
        if tables:
            ndx = tables[-1]
            col = "Ticker" if "Ticker" in ndx.columns else ndx.columns[0]
            before = len(tickers)
            for sym in ndx[col]:
                t = str(sym).strip().replace(".", "-")
                if t:
                    tickers.add(t)
            logger.info("Added %d NASDAQ-100 tickers", len(tickers) - before)
    except Exception:
        logger.warning("Could not fetch NASDAQ-100 list from Wikipedia")

    # Dow 30 — small enough to hardcode reliably
    dow30 = {
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
    }
    tickers.update(dow30)

    if not tickers:
        logger.warning("Dynamic fetch returned 0 tickers — using fallback list")
        return frozenset(_FALLBACK_TICKERS)

    # Filter by market cap >= $1B using yfinance (batch download for speed)
    logger.info("Filtering %d tickers by market cap >= $1B...", len(tickers))
    filtered: set[str] = set()
    try:
        import yfinance as yf
        # Process in batches to avoid timeout
        ticker_list = sorted(tickers)
        batch_size = 50
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]
            batch_str = " ".join(batch)
            try:
                data = yf.Tickers(batch_str)
                for sym in batch:
                    try:
                        info = data.tickers[sym].info
                        mcap = info.get("marketCap", 0)
                        if mcap and mcap >= MIN_MARKET_CAP:
                            filtered.add(sym)
                    except Exception:
                        # If we can't check, include it (better to include than exclude)
                        filtered.add(sym)
            except Exception:
                # On batch failure, include all tickers from this batch
                filtered.update(batch)
    except ImportError:
        logger.warning("yfinance not available — skipping market cap filter")
        filtered = tickers

    if not filtered:
        filtered = tickers  # Don't return empty set

    logger.info("Filtered to %d tickers with market cap >= $1B", len(filtered))
    return frozenset(filtered)


def get_eligible_tickers() -> set[str]:
    """Return the set of eligible tickers (Dow/S&P/NASDAQ, >= $1B market cap).

    Results are cached after the first call.
    """
    try:
        return set(_fetch_index_tickers_cached())
    except Exception:
        logger.warning("Falling back to curated ticker list")
        return set(_FALLBACK_TICKERS)


_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

_COMMON_WORDS = {
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
    "BEST", "NEXT", "LAST", "EVER", "FREE", "HELP",
}

_TICKER_RE = re.compile(r"\$?([A-Z]{2,5})\b")


def _get_user_agent() -> str:
    import random
    return random.choice(_USER_AGENTS)


def scan_reddit_hot(top_n: int = 20) -> list[dict]:
    """Scan Reddit finance subreddits for trending tickers.

    Returns a list of dicts: {ticker, mentions, avg_sentiment, sources}.
    Only includes tickers on Dow/S&P/NASDAQ with >= $1B market cap.
    """
    from bs4 import BeautifulSoup

    eligible = get_eligible_tickers()
    mention_data: dict[str, dict] = {}
    subreddits = ["wallstreetbets", "stocks", "investing", "StockMarket",
                  "options", "pennystocks"]

    for subreddit in subreddits:
        try:
            url = f"https://old.reddit.com/r/{subreddit}/hot"
            headers = {"User-Agent": _get_user_agent()}
            time.sleep(1.5)
            resp = requests.get(url, headers=headers, params={"limit": "100"}, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            things = soup.find_all("div", class_="thing", attrs={"data-fullname": True})

            for thing in things:
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
                    nums = re.findall(r"\d+", comments_el.get_text(strip=True))
                    if nums:
                        num_comments = int(nums[0])

                matches = _TICKER_RE.findall(title)
                for match in matches:
                    match = match.upper()
                    if match in _COMMON_WORDS:
                        continue
                    if match not in eligible:
                        continue

                    if match not in mention_data:
                        mention_data[match] = {
                            "ticker": match,
                            "mentions": 0,
                            "total_score": 0,
                            "total_comments": 0,
                            "polarities": [],
                            "sources": set(),
                        }

                    d = mention_data[match]
                    d["mentions"] += 1
                    d["total_score"] += score
                    d["total_comments"] += num_comments
                    d["sources"].add(f"r/{subreddit}")

                    polarity = TextBlob(title).sentiment.polarity
                    d["polarities"].append(polarity)

        except Exception:
            logger.exception("Error scanning r/%s", subreddit)

    # Build results sorted by mentions
    results = []
    for ticker, d in mention_data.items():
        avg_pol = sum(d["polarities"]) / len(d["polarities"]) if d["polarities"] else 0
        results.append({
            "ticker": ticker,
            "mentions": d["mentions"],
            "avg_sentiment": round(avg_pol, 3),
            "total_upvotes": d["total_score"],
            "total_comments": d["total_comments"],
            "engagement_score": d["mentions"] * 2 + d["total_score"] + d["total_comments"],
            "sources": sorted(d["sources"]),
            "source_type": "Reddit",
        })

    results.sort(key=lambda x: x["engagement_score"], reverse=True)
    return results[:top_n]


def scan_stocktwits_trending(top_n: int = 20) -> list[dict]:
    """Fetch StockTwits trending tickers filtered to eligible stocks."""
    eligible = get_eligible_tickers()
    results = []
    try:
        url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return results

        data = resp.json()
        symbols = data.get("symbols", [])
        for sym in symbols:
            ticker = sym.get("symbol", "").upper()
            if ticker not in eligible:
                continue
            results.append({
                "ticker": ticker,
                "mentions": sym.get("watchlist_count", 0),
                "avg_sentiment": 0.0,
                "total_upvotes": 0,
                "total_comments": 0,
                "engagement_score": sym.get("watchlist_count", 0),
                "sources": ["StockTwits Trending"],
                "source_type": "StockTwits",
            })
    except Exception:
        logger.exception("Error fetching StockTwits trending")

    return results[:top_n]


def get_social_hottest(top_n: int = 20) -> list[dict]:
    """Aggregate Reddit + StockTwits to find the top-N hottest stocks.

    Only includes tickers listed on Dow, S&P 500, or NASDAQ with market
    cap >= $1B.

    Returns a list of dicts sorted by combined engagement score:
        {ticker, mentions, avg_sentiment, engagement_score, sources}
    """
    reddit = scan_reddit_hot(top_n=50)
    stocktwits = scan_stocktwits_trending(top_n=50)

    # Merge by ticker
    merged: dict[str, dict] = {}
    for item in reddit + stocktwits:
        ticker = item["ticker"]
        if ticker not in merged:
            merged[ticker] = {
                "ticker": ticker,
                "mentions": 0,
                "polarities": [],
                "total_upvotes": 0,
                "total_comments": 0,
                "engagement_score": 0,
                "sources": [],
            }
        m = merged[ticker]
        m["mentions"] += item["mentions"]
        m["total_upvotes"] += item["total_upvotes"]
        m["total_comments"] += item["total_comments"]
        m["engagement_score"] += item["engagement_score"]
        m["sources"].extend(item["sources"])
        if item["avg_sentiment"] != 0:
            m["polarities"].append(item["avg_sentiment"])

    results = []
    for ticker, m in merged.items():
        avg_sent = sum(m["polarities"]) / len(m["polarities"]) if m["polarities"] else 0
        sentiment_label = "Bullish" if avg_sent > 0.1 else ("Bearish" if avg_sent < -0.1 else "Neutral")
        results.append({
            "ticker": ticker,
            "mentions": m["mentions"],
            "avg_sentiment": round(avg_sent, 3),
            "sentiment_label": sentiment_label,
            "total_upvotes": m["total_upvotes"],
            "total_comments": m["total_comments"],
            "engagement_score": m["engagement_score"],
            "sources": ", ".join(sorted(set(m["sources"]))),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    results.sort(key=lambda x: x["engagement_score"], reverse=True)
    return results[:top_n]
