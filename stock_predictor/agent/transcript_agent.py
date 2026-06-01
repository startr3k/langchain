"""Earnings call transcript agent — fetches and analyzes forward guidance.

Uses SEC EDGAR to find the latest earnings call transcript (8-K exhibits)
for a given ticker, then uses OpenAI to extract forward guidance and
forward-looking initiatives.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "StockPredictor Research research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Cache: ticker -> CIK (shared with sec_edgar.py)
_CIK_CACHE: dict[str, str] = {}


# ------------------------------------------------------------------
# SEC EDGAR helpers
# ------------------------------------------------------------------

def _get_cik(ticker: str) -> str | None:
    """Look up the SEC CIK number for a ticker symbol."""
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    _CIK_CACHE[ticker] = cik
                    return cik
    except Exception:
        logger.debug("Could not look up CIK for %s", ticker)
    return None


def _get_recent_8k_filings(cik: str, max_filings: int = 20) -> list[dict]:
    """Return recent 8-K filings for a CIK from EDGAR submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        time.sleep(0.12)
        if resp.status_code != 200:
            return []
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for i, form in enumerate(forms):
            if form == "8-K" and i < len(accessions):
                filings.append({
                    "accession": accessions[i],
                    "date": dates[i] if i < len(dates) else "",
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                })
                if len(filings) >= max_filings:
                    break
        return filings
    except Exception as e:
        logger.debug("Error fetching 8-K filings for CIK %s: %s", cik, e)
        return []


def _get_filing_exhibits(cik: str, accession: str) -> list[dict]:
    """Get the list of documents/exhibits in a filing via HTML directory listing."""
    acc_clean = accession.replace("-", "")
    cik_num = cik.lstrip("0")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_clean}/"
    )
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        time.sleep(0.12)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        items = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            name = href.rsplit("/", 1)[-1] if "/" in href else href
            if name.endswith((".htm", ".html", ".txt")) and name != "":
                # Skip non-exhibit files
                if name.endswith(("-index.html", "-index-headers.html")):
                    continue
                items.append({"name": name})
        return items
    except Exception as e:
        logger.debug("Error fetching exhibits for %s: %s", accession, e)
        return []


def _download_exhibit(cik: str, accession: str, filename: str) -> str | None:
    """Download the text content of an exhibit."""
    acc_clean = accession.replace("-", "")
    cik_num = cik.lstrip("0")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_num}/"
        f"{acc_clean}/{filename}"
    )
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        time.sleep(0.12)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type or filename.endswith(".htm"):
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = resp.text
        return text
    except Exception as e:
        logger.debug("Error downloading exhibit %s: %s", filename, e)
        return None


def _is_transcript_exhibit(item: dict) -> bool:
    """Heuristic to determine if an exhibit is likely a call transcript."""
    name = item.get("name", "").lower()
    if name.endswith((".xml", ".xsd", ".json")):
        return False
    # Exhibits named ex-99.*, or files with "transcript", "commentary",
    # "pr" (press release), or "earnings" in the name
    transcript_patterns = [
        r"ex-?99", r"exhibit.*99", r"transcript", r"commentary",
        r"earnings", r"call", r"pr\b",
    ]
    return any(re.search(p, name) for p in transcript_patterns)


def _text_looks_like_transcript(text: str) -> bool:
    """Check if the text content looks like an earnings call transcript or press release with guidance."""
    text_lower = text[:8000].lower()
    transcript_keywords = [
        "earnings call", "conference call", "q&a", "question-and-answer",
        "operator", "good morning", "good afternoon", "thank you for joining",
        "opening remarks", "prepared remarks", "forward-looking statements",
        "revenue", "earnings per share", "guidance", "outlook", "fiscal",
        "quarter", "results", "commentary", "expects", "financial results",
    ]
    matches = sum(1 for kw in transcript_keywords if kw in text_lower)
    return matches >= 2


# ------------------------------------------------------------------
# Main transcript fetcher
# ------------------------------------------------------------------

def fetch_latest_transcript(ticker: str) -> dict:
    """Fetch the latest earnings call transcript for a ticker from SEC EDGAR.

    Searches through recent 8-K filings to find exhibits that look like
    earnings call transcripts.

    Returns:
        Dict with keys: ticker, found (bool), date, text, source_url, error.
    """
    cik = _get_cik(ticker)
    if cik is None:
        return {
            "ticker": ticker,
            "found": False,
            "error": f"Could not find SEC CIK for {ticker}",
        }

    filings = _get_recent_8k_filings(cik, max_filings=15)
    if not filings:
        return {
            "ticker": ticker,
            "found": False,
            "error": f"No recent 8-K filings found for {ticker}",
        }

    for filing in filings:
        exhibits = _get_filing_exhibits(cik, filing["accession"])

        # Look for transcript-like exhibits
        transcript_candidates = [e for e in exhibits if _is_transcript_exhibit(e)]
        if not transcript_candidates:
            continue

        for exhibit in transcript_candidates:
            text = _download_exhibit(cik, filing["accession"], exhibit["name"])
            if text and _text_looks_like_transcript(text):
                # Truncate very long transcripts (keep first ~15k chars
                # which covers the prepared remarks + start of Q&A)
                if len(text) > 15000:
                    text = text[:15000] + "\n\n[... transcript truncated ...]"

                acc_clean = filing["accession"].replace("-", "")
                cik_num = cik.lstrip("0")
                source_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_num}/"
                    f"{acc_clean}/{exhibit['name']}"
                )
                return {
                    "ticker": ticker,
                    "found": True,
                    "date": filing["date"],
                    "text": text,
                    "source_url": source_url,
                }

    # No transcript found — try EFTS full-text search as fallback
    return _efts_fallback(ticker, cik)


def _efts_fallback(ticker: str, cik: str) -> dict:
    """Use SEC EDGAR full-text search to find earnings call transcripts."""
    try:
        search_url = (
            "https://efts.sec.gov/LATEST/search-index"
            f"?q=%22earnings+call+transcript%22&forms=8-K"
            f"&dateRange=custom"
            f"&startdt=2024-01-01"
            f"&entities={cik}"
            f"&size=5"
        )
        resp = requests.get(search_url, headers=SEC_HEADERS, timeout=15)
        time.sleep(0.12)
        if resp.status_code != 200:
            return {
                "ticker": ticker,
                "found": False,
                "error": f"EFTS search failed (HTTP {resp.status_code})",
            }

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return {
                "ticker": ticker,
                "found": False,
                "error": (
                    f"No earnings call transcript found for {ticker} "
                    "in recent SEC filings"
                ),
            }

        # Try the first hit
        hit = hits[0]
        file_url = hit.get("_source", {}).get("file_url", "")
        filing_date = hit.get("_source", {}).get("file_date", "")

        if file_url:
            full_url = f"https://www.sec.gov{file_url}" if file_url.startswith("/") else file_url
            try:
                resp2 = requests.get(full_url, headers=SEC_HEADERS, timeout=30)
                time.sleep(0.12)
                if resp2.status_code == 200:
                    soup = BeautifulSoup(resp2.text, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    if len(text) > 15000:
                        text = text[:15000] + "\n\n[... transcript truncated ...]"
                    return {
                        "ticker": ticker,
                        "found": True,
                        "date": filing_date,
                        "text": text,
                        "source_url": full_url,
                    }
            except Exception:
                pass

        return {
            "ticker": ticker,
            "found": False,
            "error": f"Found EFTS hit but could not download transcript for {ticker}",
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "found": False,
            "error": f"EFTS search error: {e}",
        }


# ------------------------------------------------------------------
# Forward guidance extraction via LLM
# ------------------------------------------------------------------

def extract_forward_guidance(
    ticker: str,
    transcript_text: str,
    api_key: str | None = None,
    model: str = "gpt-4o",
) -> str:
    """Use OpenAI to extract forward guidance from a transcript.

    Args:
        ticker: Stock ticker symbol.
        transcript_text: The earnings call transcript text.
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        model: OpenAI model to use.

    Returns:
        Structured summary of forward guidance and initiatives.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: OpenAI API key required for transcript analysis."

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOpenAI(model=model, temperature=0.1, api_key=api_key)

    system_msg = SystemMessage(content=(
        "You are an expert financial analyst specializing in earnings call "
        "transcript analysis. Your task is to extract forward guidance and "
        "forward-looking initiatives from the transcript. Be specific and "
        "cite numbers when available."
    ))

    user_msg = HumanMessage(content=f"""Analyze this earnings call transcript for {ticker} and answer:

**What are the forward guidance and any forward-related initiatives mentioned in the call transcript?**

Structure your response as:

## Forward Guidance
- Revenue guidance (if mentioned)
- Earnings/EPS guidance (if mentioned)
- Margin guidance (if mentioned)
- Any other quantitative guidance

## Forward-Looking Initiatives
- New products/services/markets being launched or expanded
- Strategic partnerships or acquisitions planned
- R&D investments or technology initiatives
- Cost reduction or restructuring plans
- Capital allocation plans (buybacks, dividends, capex)

## Key Quotes
- Include 2-3 direct quotes from management that best capture the forward outlook

## Risk Factors Mentioned
- Any risks or headwinds management highlighted for the forward period

If any section has no relevant information, state "Not mentioned in transcript."

TRANSCRIPT:
{transcript_text}""")

    try:
        response = llm.invoke([system_msg, user_msg])
        return response.content
    except Exception as e:
        logger.error("LLM extraction failed for %s: %s", ticker, e)
        return f"Error analyzing transcript: {e}"


# ------------------------------------------------------------------
# Combined: fetch + analyze
# ------------------------------------------------------------------

def analyze_ticker_forward_guidance(
    ticker: str,
    api_key: str | None = None,
    model: str = "gpt-4o",
) -> dict:
    """Fetch the latest earnings call transcript and extract forward guidance.

    This is the main entry point for the transcript agent.

    Returns:
        Dict with ticker, found, date, source_url, forward_guidance, error.
    """
    result = fetch_latest_transcript(ticker)

    if not result.get("found"):
        return {
            "ticker": ticker,
            "found": False,
            "error": result.get("error", "Transcript not found"),
            "forward_guidance": None,
        }

    guidance = extract_forward_guidance(
        ticker,
        result["text"],
        api_key=api_key,
        model=model,
    )

    return {
        "ticker": ticker,
        "found": True,
        "date": result.get("date"),
        "source_url": result.get("source_url"),
        "forward_guidance": guidance,
    }


def analyze_batch_forward_guidance(
    tickers: list[str],
    api_key: str | None = None,
    model: str = "gpt-4o",
) -> list[dict]:
    """Analyze forward guidance for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        api_key: OpenAI API key.
        model: OpenAI model to use.

    Returns:
        List of analysis results, one per ticker.
    """
    results = []
    for ticker in tickers:
        logger.info("Analyzing forward guidance for %s...", ticker)
        result = analyze_ticker_forward_guidance(ticker, api_key=api_key, model=model)
        results.append(result)
    return results
