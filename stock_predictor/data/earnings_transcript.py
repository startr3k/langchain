"""Earnings call transcript sentiment via DuckDuckGo + web scraping.

Portable module — works on any machine with Python. No API key required.
Searches DuckDuckGo for the latest earnings call transcript, scrapes
the content from Motley Fool or other sources, and computes
Loughran-McDonald financial sentiment.

Features:
- transcript_sentiment: positive/negative word ratio from latest call
- transcript_polarity: TextBlob polarity of key excerpts
- transcript_date: date of the earnings call
"""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache

import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

logger = logging.getLogger(__name__)

TRANSCRIPT_FEATURES = [
    "transcript_sentiment",
    "transcript_polarity",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Loughran-McDonald financial sentiment word lists
_LM_POSITIVE = {
    "achieve", "attain", "benefit", "better", "boost", "breakthrough",
    "creative", "deliver", "earn", "enhance", "exceed", "excellent",
    "favorable", "gain", "good", "great", "grew", "grow", "growth",
    "highest", "improve", "improvement", "increase", "innovation",
    "leader", "leading", "opportunity", "optimal", "outperform",
    "positive", "profitability", "profitable", "progress", "record",
    "recover", "recovery", "strong", "stronger", "succeed", "success",
    "successful", "superior", "surpass", "upturn", "winner",
}

_LM_NEGATIVE = {
    "abandon", "adverse", "against", "breach", "burden", "catastrophe",
    "cease", "challenge", "claim", "close", "closure", "concern",
    "critical", "decline", "decreased", "default", "deficit", "delay",
    "deteriorate", "difficult", "difficulty", "diminish", "disappointing",
    "discontinue", "doubt", "downgrade", "downturn", "drop", "fail",
    "failure", "force", "fraud", "hinder", "idle", "impair", "impairment",
    "inability", "inadequate", "ineffective", "investigation", "lawsuit",
    "layoff", "liabilities", "liquidate", "litigation", "lose", "loss",
    "losses", "negative", "penalty", "problem", "recall", "recession",
    "restructuring", "risk", "severe", "shortage", "shrink", "shutdown",
    "slowdown", "struggle", "sue", "suffer", "terminate", "threat",
    "uncertain", "uncertainty", "unfavorable", "unprofitable", "volatile",
    "volatility", "weak", "weakness", "worsen", "writedown", "writeoff",
}


def _search_transcript_url(ticker: str) -> str | None:
    """Search DuckDuckGo for the latest earnings call transcript URL."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning(
                "Neither 'ddgs' nor 'duckduckgo_search' installed. "
                "Install with: pip install ddgs"
            )
            return None

    try:
        results = DDGS().text(
            f"{ticker} earnings call transcript site:fool.com",
            max_results=3,
        )
        for r in results:
            url = r.get("href", "")
            if "fool.com" in url and "earnings/call-transcripts" in url:
                return url

        # Fallback: broader search
        results = DDGS().text(
            f"{ticker} latest earnings call transcript",
            max_results=5,
        )
        for r in results:
            url = r.get("href", "")
            if any(
                domain in url
                for domain in ["fool.com", "seekingalpha.com", "finance.yahoo.com"]
            ):
                return url
    except Exception:
        logger.debug("DuckDuckGo search failed for %s transcript", ticker)

    return None


def _scrape_motley_fool(url: str) -> str:
    """Scrape transcript text from a Motley Fool page."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        article = (
            soup.find("div", class_="article-body")
            or soup.find("article")
            or soup.find("div", class_="tailwind-article-body")
        )
        if article:
            return article.get_text(separator="\n", strip=True)
    except Exception:
        logger.debug("Failed to scrape Motley Fool: %s", url)
    return ""


def _scrape_generic(url: str) -> str:
    """Scrape transcript text from a generic page."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
            return text[:200_000]
    except Exception:
        logger.debug("Failed to scrape: %s", url)
    return ""


def _compute_lm_sentiment(text: str) -> float:
    """Compute Loughran-McDonald sentiment score.

    Returns a ratio: (positive - negative) / (positive + negative + 1).
    Range roughly [-1, 1]. Positive = optimistic tone.
    """
    words = re.findall(r"[a-z]+", text.lower())
    pos_count = sum(1 for w in words if w in _LM_POSITIVE)
    neg_count = sum(1 for w in words if w in _LM_NEGATIVE)
    return (pos_count - neg_count) / (pos_count + neg_count + 1)


def _compute_textblob_polarity(text: str) -> float:
    """Compute TextBlob polarity on key excerpts (management remarks).

    Focuses on the first ~5000 chars which typically contain the
    CEO/CFO prepared remarks — the most sentiment-rich portion.
    """
    excerpt = text[:5000]
    try:
        return TextBlob(excerpt).sentiment.polarity
    except Exception:
        return 0.0


@lru_cache(maxsize=256)
def fetch_earnings_transcript(ticker: str) -> dict:
    """Fetch and analyze the latest earnings call transcript for a ticker.

    Returns a dict with:
    - transcript_sentiment: Loughran-McDonald sentiment score
    - transcript_polarity: TextBlob polarity of key excerpts
    - transcript_url: source URL
    - transcript_date: date from the URL (if available)
    - transcript_text_preview: first 500 chars of transcript
    - transcript_source_texts: list of (source, text_excerpt, polarity) tuples
    """
    url = _search_transcript_url(ticker)
    if not url:
        return {
            "transcript_sentiment": None,
            "transcript_polarity": None,
            "transcript_url": None,
            "transcript_date": None,
            "transcript_text_preview": None,
            "transcript_source_texts": [],
        }

    time.sleep(0.5)  # Be polite

    if "fool.com" in url:
        text = _scrape_motley_fool(url)
    else:
        text = _scrape_generic(url)

    if not text or len(text) < 500:
        return {
            "transcript_sentiment": None,
            "transcript_polarity": None,
            "transcript_url": url,
            "transcript_date": None,
            "transcript_text_preview": None,
            "transcript_source_texts": [],
        }

    lm_score = _compute_lm_sentiment(text)
    tb_polarity = _compute_textblob_polarity(text)

    # Extract date from Motley Fool URL pattern
    # e.g. /2026/04/30/apple-aapl-q2-2026-...
    date_match = re.search(r"/(\d{4}/\d{2}/\d{2})/", url)
    transcript_date = date_match.group(1).replace("/", "-") if date_match else None

    # Build source texts for display
    # Split transcript into sections and get sentiment per section
    source_texts = []
    # Get prepared remarks (first section)
    sections = text.split("\n\n")
    prepared = "\n".join(sections[:3])[:2000]
    if prepared:
        source_texts.append((
            "Earnings Call (Prepared Remarks)",
            prepared[:500],
            round(TextBlob(prepared).sentiment.polarity, 3),
        ))

    # Get Q&A section (latter half)
    qa_start = text.find("Questions and Answers") or text.find("Question-and-Answer")
    if qa_start and qa_start > 0:
        qa_text = text[qa_start:qa_start + 2000]
        source_texts.append((
            "Earnings Call (Q&A)",
            qa_text[:500],
            round(TextBlob(qa_text).sentiment.polarity, 3),
        ))

    return {
        "transcript_sentiment": round(lm_score, 4),
        "transcript_polarity": round(tb_polarity, 4),
        "transcript_url": url,
        "transcript_date": transcript_date,
        "transcript_text_preview": text[:500],
        "transcript_source_texts": source_texts,
    }
