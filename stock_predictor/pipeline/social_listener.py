"""Social media & market buzz listener — surfaces the top-20 hottest stocks
from Yahoo Finance, Finviz news, and GDELT that are listed on Dow, S&P 500,
or NASDAQ with market cap >= $1B.

Data sources:
  - Yahoo Finance Trending (real-time trending tickers)
  - Yahoo Finance Screeners (most active, day gainers)
  - Finviz news headlines (sentiment via TextBlob)
  - GDELT Global News (article volume + tone, with rate-limit fallback)

Updated on demand; results include mention counts, sentiment, and source
breakdown.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import requests
from textblob import TextBlob

logger = logging.getLogger(__name__)

# File-based cache for the eligible ticker universe.
_TICKER_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "eligible_tickers_cache.json"
_TICKER_CACHE_MAX_AGE_HOURS = 24  # refresh if older than this

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

    _wiki_headers = {"User-Agent": "StockPredictor/1.0 (python-requests)"}

    def _wiki_read_html(url: str, match: str) -> list[pd.DataFrame]:
        from io import StringIO

        resp = requests.get(url, headers=_wiki_headers, timeout=15)
        resp.raise_for_status()
        return pd.read_html(StringIO(resp.text), match=match)

    # S&P 500 from Wikipedia
    try:
        tables = _wiki_read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "Symbol",
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
        tables = _wiki_read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            "Ticker",
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
                        filtered.add(sym)
            except Exception:
                filtered.update(batch)
    except ImportError:
        logger.warning("yfinance not available — skipping market cap filter")
        filtered = tickers

    if not filtered:
        filtered = tickers

    logger.info("Filtered to %d tickers with market cap >= $1B", len(filtered))
    return frozenset(filtered)


def _load_ticker_cache() -> set[str] | None:
    """Load cached tickers from disk if fresh enough."""
    if not _TICKER_CACHE_PATH.exists():
        return None
    try:
        data = json.loads(_TICKER_CACHE_PATH.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600
        if age_hours > _TICKER_CACHE_MAX_AGE_HOURS:
            logger.info("Ticker cache is %.1f hours old — will refresh", age_hours)
            return None
        tickers = set(data["tickers"])
        logger.info(
            "Loaded %d cached eligible tickers (%.1fh old)",
            len(tickers), age_hours,
        )
        return tickers
    except Exception:
        logger.warning("Could not read ticker cache")
        return None


def _save_ticker_cache(tickers: set[str]) -> None:
    """Persist the eligible ticker set to disk."""
    try:
        data = {
            "cached_at": datetime.now().isoformat(),
            "count": len(tickers),
            "tickers": sorted(tickers),
        }
        _TICKER_CACHE_PATH.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d eligible tickers to cache", len(tickers))
    except Exception:
        logger.warning("Could not write ticker cache")


def get_eligible_tickers(*, force_refresh: bool = False) -> set[str]:
    """Return the set of eligible tickers (Dow/S&P/NASDAQ, >= $1B market cap).

    Uses a disk-based cache (JSON file) that refreshes every 24 hours.
    Pass ``force_refresh=True`` to bypass the cache.
    """
    if not force_refresh:
        cached = _load_ticker_cache()
        if cached:
            return cached

    try:
        # Clear the lru_cache so we re-fetch from Wikipedia
        _fetch_index_tickers_cached.cache_clear()
        tickers = set(_fetch_index_tickers_cached())
        _save_ticker_cache(tickers)
        return tickers
    except Exception:
        logger.warning("Falling back to curated ticker list")
        return set(_FALLBACK_TICKERS)


def get_ticker_cache_info() -> dict:
    """Return metadata about the ticker cache (for UI display)."""
    if not _TICKER_CACHE_PATH.exists():
        return {"cached": False, "count": 0, "age_hours": None}
    try:
        data = json.loads(_TICKER_CACHE_PATH.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600
        return {
            "cached": True,
            "count": data["count"],
            "age_hours": round(age_hours, 1),
            "cached_at": data["cached_at"],
        }
    except Exception:
        return {"cached": False, "count": 0, "age_hours": None}


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

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

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
]


def _get_user_agent() -> str:
    import random
    return random.choice(_USER_AGENTS)


# ---------------------------------------------------------------------------
# Source 1: Reddit (graceful fallback if blocked)
# ---------------------------------------------------------------------------

def scan_reddit_hot(top_n: int = 20) -> list[dict]:
    """Scan Reddit finance subreddits for trending tickers.

    Returns a list of dicts: {ticker, score, avg_sentiment, source}.
    Only includes tickers on Dow/S&P/NASDAQ with >= $1B market cap.
    Gracefully returns empty list if Reddit blocks the request.
    """
    from bs4 import BeautifulSoup

    eligible = get_eligible_tickers()
    mention_data: dict[str, dict] = {}
    subreddits = ["wallstreetbets", "stocks", "investing", "StockMarket"]

    for subreddit in subreddits:
        try:
            url = f"https://old.reddit.com/r/{subreddit}/hot"
            headers = {"User-Agent": _get_user_agent()}
            time.sleep(1.5)
            resp = requests.get(url, headers=headers, params={"limit": "100"}, timeout=15)
            if resp.status_code != 200:
                logger.debug("Reddit r/%s returned %d — skipping", subreddit, resp.status_code)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            things = soup.find_all("div", class_="thing", attrs={"data-fullname": True})

            for thing in things:
                title_el = thing.find("a", class_="title")
                title = title_el.get_text(strip=True) if title_el else ""

                score_el = thing.find("div", class_="score unvoted")
                score_text = score_el.get("title", "0") if score_el else "0"
                try:
                    post_score = int(score_text)
                except (ValueError, TypeError):
                    post_score = 0

                comments_el = thing.find("a", class_="comments")
                num_comments = 0
                if comments_el:
                    nums = re.findall(r"\d+", comments_el.get_text(strip=True))
                    if nums:
                        num_comments = int(nums[0])

                matches = _TICKER_RE.findall(title)
                for match in matches:
                    match = match.upper()
                    if match in _COMMON_WORDS or match not in eligible:
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
                    d["total_score"] += post_score
                    d["total_comments"] += num_comments
                    d["sources"].add(f"r/{subreddit}")

                    polarity = TextBlob(title).sentiment.polarity
                    d["polarities"].append(polarity)

        except Exception:
            logger.debug("Error scanning r/%s — Reddit may be blocked", subreddit)

    results = []
    for ticker, d in mention_data.items():
        avg_pol = sum(d["polarities"]) / len(d["polarities"]) if d["polarities"] else 0
        engagement = d["mentions"] * 2 + d["total_score"] + d["total_comments"]
        results.append({
            "ticker": ticker,
            "score": engagement,
            "avg_sentiment": round(avg_pol, 3),
            "source": "Reddit",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        logger.info("Reddit: found %d trending tickers", len(results))
    else:
        logger.info("Reddit: no results (may be blocked from this environment)")
    return results[:top_n]


# ---------------------------------------------------------------------------
# Source 2: Yahoo Finance Trending + Screeners
# ---------------------------------------------------------------------------

_YF_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _fetch_yahoo_trending() -> list[dict]:
    """Fetch trending tickers from Yahoo Finance."""
    results: list[dict] = []
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/trending/US",
            headers=_YF_HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("Yahoo trending returned %d", resp.status_code)
            return results

        data = resp.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        for rank, q in enumerate(quotes, 1):
            sym = q.get("symbol", "").upper()
            if sym:
                results.append({
                    "ticker": sym,
                    "score": max(1, 25 - rank),  # higher rank = higher score
                    "source": "Yahoo Trending",
                })
    except Exception:
        logger.exception("Error fetching Yahoo trending")
    return results


def _fetch_yahoo_screener(screener_id: str, label: str) -> list[dict]:
    """Fetch a Yahoo Finance pre-defined screener (most_actives, day_gainers, etc.)."""
    results: list[dict] = []
    try:
        resp = requests.get(
            f"https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved",
            params={"scrIds": screener_id, "count": "25"},
            headers=_YF_HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            return results

        data = resp.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        for rank, q in enumerate(quotes, 1):
            sym = q.get("symbol", "").upper()
            mcap = q.get("marketCap", 0)
            if sym:
                results.append({
                    "ticker": sym,
                    "score": max(1, 30 - rank),
                    "source": f"Yahoo {label}",
                    "market_cap": mcap,
                    "volume": q.get("regularMarketVolume", 0),
                    "change_pct": q.get("regularMarketChangePercent", 0),
                })
    except Exception:
        logger.exception("Error fetching Yahoo screener %s", screener_id)
    return results


def scan_yahoo_finance() -> list[dict]:
    """Aggregate Yahoo Finance sources: trending + most active + day gainers."""
    all_items: list[dict] = []
    all_items.extend(_fetch_yahoo_trending())
    all_items.extend(_fetch_yahoo_screener("most_actives", "Most Active"))
    all_items.extend(_fetch_yahoo_screener("day_gainers", "Day Gainers"))
    all_items.extend(_fetch_yahoo_screener("day_losers", "Day Losers"))
    return all_items


# ---------------------------------------------------------------------------
# Source 2: Finviz News Headlines
# ---------------------------------------------------------------------------

def scan_finviz_news(tickers: list[str], max_tickers: int = 30) -> list[dict]:
    """Scrape Finviz news headlines for a list of tickers, compute sentiment.

    Only fetches for the first ``max_tickers`` to stay within rate limits.
    """
    results: list[dict] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    for ticker in tickers[:max_tickers]:
        try:
            from bs4 import BeautifulSoup

            url = f"https://finviz.com/quote.ashx?t={ticker}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            news_table = soup.find("table", id="news-table")
            if not news_table:
                continue

            rows = news_table.find_all("tr")
            headline_count = 0
            total_polarity = 0.0

            for row in rows[:20]:  # last 20 headlines
                link = row.find("a")
                if not link:
                    continue
                headline = link.get_text(strip=True)
                polarity = TextBlob(headline).sentiment.polarity
                total_polarity += polarity
                headline_count += 1

            if headline_count > 0:
                results.append({
                    "ticker": ticker,
                    "score": headline_count,
                    "avg_sentiment": round(total_polarity / headline_count, 3),
                    "headline_count": headline_count,
                    "source": "Finviz News",
                })

            time.sleep(0.3)  # gentle rate limiting

        except Exception:
            logger.debug("Finviz fetch failed for %s", ticker)
    return results


# ---------------------------------------------------------------------------
# Source 3: GDELT Global News API
# ---------------------------------------------------------------------------

def scan_gdelt_news(tickers: list[str], max_tickers: int = 20) -> list[dict]:
    """Query GDELT DOC API for news article counts and tone per ticker.

    GDELT rate-limits to 1 request per 5 seconds.  If rate-limited,
    returns whatever was collected before the limit was hit.
    """
    results: list[dict] = []

    for ticker in tickers[:max_tickers]:
        try:
            time.sleep(6)  # GDELT requires >= 5s between requests
            resp = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": f"{ticker} stock",
                    "mode": "artlist",
                    "maxrecords": "10",
                    "format": "json",
                    "timespan": "24h",
                },
                timeout=20,
            )
            if resp.status_code == 429:
                logger.warning("GDELT rate-limited — stopping after %d tickers", len(results))
                break
            if resp.status_code != 200:
                continue

            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                continue

            # Compute average tone from GDELT (range roughly -10 to +10)
            tones = []
            for a in articles:
                tone_str = a.get("tone", "")
                if tone_str:
                    try:
                        tone_val = float(tone_str.split(",")[0])
                        tones.append(tone_val)
                    except (ValueError, IndexError):
                        pass

            avg_tone = sum(tones) / len(tones) if tones else 0
            # Normalize GDELT tone to -1..+1 scale
            normalized_sentiment = max(-1.0, min(1.0, avg_tone / 10.0))

            results.append({
                "ticker": ticker,
                "score": len(articles),
                "avg_sentiment": round(normalized_sentiment, 3),
                "article_count": len(articles),
                "source": "GDELT News",
            })

        except Exception:
            logger.debug("GDELT fetch failed for %s", ticker)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def get_social_hottest(top_n: int = 20) -> list[dict]:
    """Aggregate Reddit, Yahoo Finance, Finviz, and GDELT to find the top-N
    hottest stocks from Dow, S&P 500, and NASDAQ with >= $1B market cap.

    Returns a list of dicts sorted by combined engagement score:
        {ticker, mentions, avg_sentiment, engagement_score, sources, ...}
    """
    eligible = get_eligible_tickers()
    logger.info("Scanning %d eligible tickers for buzz...", len(eligible))

    # Reddit (graceful — returns empty list if blocked)
    reddit_items = scan_reddit_hot(top_n=50)
    logger.info("Reddit: %d trending tickers", len(reddit_items))

    # Yahoo Finance
    yahoo_items = scan_yahoo_finance()
    logger.info("Yahoo Finance: %d raw items", len(yahoo_items))

    # Identify which tickers from all sources are eligible (for Finviz deep-dive)
    seen_tickers: list[str] = []
    for item in reddit_items + yahoo_items:
        if item["ticker"] in eligible and item["ticker"] not in seen_tickers:
            seen_tickers.append(item["ticker"])

    # Finviz news for top candidates
    finviz_candidates = seen_tickers[:30]
    finviz_items = scan_finviz_news(finviz_candidates)
    logger.info("Finviz: %d tickers with news", len(finviz_items))

    # GDELT (slow — only fetch for top candidates, skip if rate-limited)
    gdelt_items = scan_gdelt_news(finviz_candidates[:10])
    logger.info("GDELT: %d tickers with news", len(gdelt_items))

    # Merge all sources by ticker
    merged: dict[str, dict] = {}

    for item in reddit_items + yahoo_items + finviz_items + gdelt_items:
        ticker = item["ticker"]
        if ticker not in eligible:
            continue

        if ticker not in merged:
            merged[ticker] = {
                "ticker": ticker,
                "mentions": 0,
                "total_score": 0,
                "polarities": [],
                "sources": [],
                "volume": 0,
                "change_pct": 0.0,
                "market_cap": 0,
            }

        m = merged[ticker]
        m["mentions"] += 1
        m["total_score"] += item.get("score", 1)
        m["sources"].append(item.get("source", "Unknown"))

        sent = item.get("avg_sentiment")
        if sent is not None and sent != 0:
            m["polarities"].append(sent)

        if item.get("volume"):
            m["volume"] = max(m["volume"], item["volume"])
        if item.get("change_pct"):
            m["change_pct"] = item["change_pct"]
        if item.get("market_cap"):
            m["market_cap"] = max(m["market_cap"], item["market_cap"])

    # Build final results
    results: list[dict] = []
    for ticker, m in merged.items():
        avg_sent = sum(m["polarities"]) / len(m["polarities"]) if m["polarities"] else 0
        sentiment_label = (
            "Bullish" if avg_sent > 0.1
            else ("Bearish" if avg_sent < -0.1 else "Neutral")
        )
        engagement = m["total_score"] + m["mentions"] * 3

        results.append({
            "ticker": ticker,
            "mentions": m["mentions"],
            "avg_sentiment": round(avg_sent, 3),
            "sentiment_label": sentiment_label,
            "total_upvotes": m["volume"],
            "total_comments": 0,
            "engagement_score": engagement,
            "change_pct": round(m["change_pct"], 2) if m["change_pct"] else 0,
            "market_cap": m["market_cap"],
            "sources": ", ".join(sorted(set(m["sources"]))),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    results.sort(key=lambda x: x["engagement_score"], reverse=True)
    top = results[:top_n]

    # Persist to CSV for cross-referencing in other pages
    if top:
        _save_social_buzz(top)

    return top


# ---------------------------------------------------------------------------
# Social buzz persistence
# ---------------------------------------------------------------------------

_SOCIAL_BUZZ_CSV = Path(__file__).resolve().parent.parent.parent / "social_buzz.csv"


def _save_social_buzz(items: list[dict]) -> None:
    """Save social buzz results to CSV (overwrites with latest snapshot)."""
    try:
        import pandas as pd

        df = pd.DataFrame(items)
        df.to_csv(_SOCIAL_BUZZ_CSV, index=False)
        logger.info("Saved %d social buzz items to %s", len(items), _SOCIAL_BUZZ_CSV)
    except Exception:
        logger.warning("Could not save social buzz CSV")


def get_hot_tickers() -> set[str]:
    """Return the set of tickers currently flagged as 'hot' on social media.

    Reads from the persisted social_buzz.csv.  Returns an empty set if the
    file doesn't exist or is stale (> 24 hours old).
    """
    if not _SOCIAL_BUZZ_CSV.exists():
        return set()
    try:
        import pandas as pd

        # Check age
        mtime = datetime.fromtimestamp(_SOCIAL_BUZZ_CSV.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours > 24:
            return set()

        df = pd.read_csv(_SOCIAL_BUZZ_CSV)
        return set(df["ticker"].tolist())
    except Exception:
        return set()


def get_social_buzz_data() -> dict[str, dict]:
    """Return a dict of ticker -> buzz info from the persisted CSV.

    Includes sentiment, mentions, engagement, and sources for each ticker.
    Returns empty dict if no data or stale.
    """
    if not _SOCIAL_BUZZ_CSV.exists():
        return {}
    try:
        import pandas as pd

        mtime = datetime.fromtimestamp(_SOCIAL_BUZZ_CSV.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours > 24:
            return {}

        df = pd.read_csv(_SOCIAL_BUZZ_CSV)
        result = {}
        for _, row in df.iterrows():
            result[row["ticker"]] = {
                "mentions": int(row.get("mentions", 0)),
                "avg_sentiment": float(row.get("avg_sentiment", 0)),
                "sentiment_label": row.get("sentiment_label", "Neutral"),
                "engagement_score": int(row.get("engagement_score", 0)),
                "sources": row.get("sources", ""),
                "last_updated": row.get("last_updated", ""),
            }
        return result
    except Exception:
        return {}
