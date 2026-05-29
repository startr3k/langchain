"""SEC filing text sentiment analysis using Loughran-McDonald dictionary.

Extracts sentiment features from 10-K/10-Q filing text via EDGAR.
All features are time-aligned — only filings dated before each row date
are used (no look-ahead bias).

Features:
- filing_sentiment_score: positive/negative word ratio
- filing_risk_change: change in risk-related words vs prior filing
- filing_readability: Fog index (complexity measure)
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

SEC_SENTIMENT_FEATURES = [
    "filing_sentiment_score",
    "filing_risk_change",
    "filing_readability",
]

SEC_HEADERS = {
    "User-Agent": "StockPredictor Research research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Loughran-McDonald financial sentiment word lists (core subsets).
# Full dictionary has ~4,000 words; these are the most impactful.
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

_LM_RISK = {
    "risk", "risks", "risky", "uncertain", "uncertainty", "litigation",
    "lawsuit", "regulatory", "compliance", "violation", "penalty",
    "investigation", "contingent", "contingency", "exposure", "threat",
    "vulnerability", "adverse", "material", "impairment", "default",
    "fraud", "cybersecurity", "pandemic", "epidemic", "geopolitical",
}

_CIK_CACHE: dict[str, str] = {}


def _get_cik(ticker: str) -> str | None:
    """Look up SEC CIK for a ticker."""
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code == 200:
            for entry in resp.json().values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    _CIK_CACHE[ticker] = cik
                    return cik
    except Exception:
        pass
    return None


def _fetch_filing_text(cik: str, accession: str) -> str:
    """Fetch the full text of a filing from EDGAR."""
    acc_no_dash = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
        f"/{acc_no_dash}/{accession}.txt"
    )
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        time.sleep(0.15)
        if resp.status_code == 200:
            # Strip HTML tags for text analysis
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text)
            return text[:500_000]  # Cap at 500K chars to avoid memory issues
    except Exception:
        pass
    return ""


def _count_words(text: str, word_set: set[str]) -> int:
    """Count occurrences of words from a set in the text."""
    words = text.lower().split()
    return sum(1 for w in words if w.strip(".,;:!?()\"'") in word_set)


def _fog_index(text: str) -> float:
    """Compute Gunning Fog readability index."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return np.nan

    words = text.split()
    n_words = len(words)
    n_sentences = len(sentences)

    if n_sentences == 0 or n_words == 0:
        return np.nan

    # Count complex words (3+ syllables)
    def syllable_count(word: str) -> int:
        word = word.lower().strip(".,;:!?()\"'")
        if len(word) <= 3:
            return 1
        vowels = "aeiou"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    complex_words = sum(1 for w in words if syllable_count(w) >= 3)
    fog = 0.4 * (n_words / n_sentences + 100 * complex_words / n_words)
    return fog


def get_filing_sentiments(ticker: str) -> pd.DataFrame:
    """Fetch and analyze sentiment of all 10-K/10-Q filings for a ticker.

    Returns DataFrame with columns:
    - _filing_date: date the filing was made
    - filing_sentiment_score: positive/negative word ratio
    - filing_risk_count: number of risk-related words
    - filing_readability: Fog index
    """
    cik = _get_cik(ticker)
    if cik is None:
        return pd.DataFrame()

    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        time.sleep(0.15)

        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])

        # Filter to 10-K and 10-Q filings
        filings = []
        for form, date, acc in zip(forms, dates, accessions):
            if form in ("10-K", "10-K/A", "10-Q", "10-Q/A"):
                filings.append({
                    "form": form,
                    "date": date,
                    "accession": acc,
                })

        if not filings:
            return pd.DataFrame()

        # Limit to most recent 20 filings (5 years of quarterly)
        filings = filings[:20]

        records = []
        for f in filings:
            text = _fetch_filing_text(cik, f["accession"])
            if not text or len(text) < 1000:
                continue

            n_pos = _count_words(text, _LM_POSITIVE)
            n_neg = _count_words(text, _LM_NEGATIVE)
            n_risk = _count_words(text, _LM_RISK)

            sentiment = (n_pos - n_neg) / max(n_pos + n_neg, 1)
            fog = _fog_index(text[:50_000])  # Fog on first 50K chars

            records.append({
                "_filing_date": pd.Timestamp(f["date"]),
                "filing_sentiment_score": sentiment,
                "filing_risk_count": float(n_risk),
                "filing_readability": fog,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("_filing_date").reset_index(drop=True)

        # Compute risk change (vs prior filing)
        df["filing_risk_change"] = df["filing_risk_count"].diff()
        df.loc[df.index[0], "filing_risk_change"] = 0.0

        return df

    except Exception:
        logger.debug("Error fetching SEC sentiment for %s", ticker)
        return pd.DataFrame()


def align_sec_sentiment_to_dates(
    sentiment_df: pd.DataFrame,
    dates: pd.DatetimeIndex | pd.Index,
) -> pd.DataFrame:
    """Align SEC sentiment to training dates (most recent filing before each date)."""
    if sentiment_df.empty:
        return pd.DataFrame(
            {col: np.nan for col in SEC_SENTIMENT_FEATURES},
            index=range(len(dates)),
        )

    filing_dates = sentiment_df["_filing_date"].values

    aligned_rows: list[dict] = []
    for d in dates:
        ts = pd.Timestamp(d)
        mask = filing_dates <= ts
        if mask.any():
            idx = int(np.where(mask)[0][-1])
            row = sentiment_df.iloc[idx]
            rec = {col: row.get(col, np.nan) for col in SEC_SENTIMENT_FEATURES}
        else:
            rec = {col: np.nan for col in SEC_SENTIMENT_FEATURES}
        aligned_rows.append(rec)

    return pd.DataFrame(aligned_rows)
