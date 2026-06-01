"""Centralised configuration helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = Path(__file__).resolve().parent / "ticker_universe.yaml"


def load_ticker_config() -> dict[str, Any]:
    """Load ticker universe configuration from YAML."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_min_market_cap() -> int:
    """Return the configured minimum market cap in USD."""
    cfg = load_ticker_config()
    return int(cfg["min_market_cap"])


def get_eligible_tickers() -> list[str]:
    """Return sorted list of tickers from market_cap_cache.json that pass the
    minimum market cap filter defined in ``ticker_universe.yaml``.

    Falls back to ``fetch_all_nasdaq_tickers()`` only if the cache file is
    missing.
    """
    cfg = load_ticker_config()
    min_mcap: int = int(cfg["min_market_cap"])
    cache_rel: str = cfg["market_cap_cache"]
    cache_path = _REPO_ROOT / cache_rel

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                mcap: dict[str, int | None] = json.load(f)
            tickers = sorted(
                sym for sym, val in mcap.items()
                if (val or 0) >= min_mcap
            )
            logger.info(
                "Loaded %d tickers (>= $%dM) from %s",
                len(tickers), min_mcap // 1_000_000, cache_rel,
            )
            return tickers
        except Exception:
            logger.warning("Failed to read %s", cache_path)

    # Fallback: fetch from NASDAQ API (slow, requires network)
    from stock_predictor.data.yfinance_client import fetch_all_nasdaq_tickers

    logger.info("market_cap_cache.json not found — fetching from NASDAQ API")
    return fetch_all_nasdaq_tickers(
        min_market_cap=min_mcap,
        cache_path=str(cache_path),
    )
