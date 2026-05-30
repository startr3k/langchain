# Test Plan v2 — Reddit Scraping + Code Review Fixes

## What Changed (Latest Commit)
1. **Reddit scraping**: Replaced JSON API with old.reddit.com HTML scraping (rotating user agents, request delays)
2. **Stale predictor singleton fix**: `_get_predictor()` now reloads if not trained
3. **Duplicate sentiment API calls fix**: `social_media_listener_tool` calls `get_sentiment_features` once, passes result to `get_sentiment_summary`
4. **UnboundLocalError fix**: `run_agent` handles `max_iterations=0`
5. **Train/serve skew fix**: Predictions use saved training medians instead of 0.0
6. **pyproject.toml fix**: Uses `setuptools.build_meta` instead of private API

## Test Strategy
Mix of shell-based verification (unit tests, Python checks) and GUI testing (Streamlit app). GUI tests cover the user-visible Reddit scraping change and end-to-end sentiment flow.

---

## Test 1: Unit Tests Pass (Shell)
**Command**: `python -m pytest stock_predictor/tests/ -v --tb=short`
**Pass criteria**: All 45 tests pass, exit code 0
**Why adversarial**: Tests include HTML-mocked Reddit scraping, median persistence, and sentiment summary with pre-computed features — a broken change would fail specific tests.

## Test 2: Reddit Scraping Returns Data (Shell)
**Command**: Python script calling `fetch_reddit_sentiment("NVDA", limit=5)` directly
**Pass criteria**:
- Returns a non-empty list (at least 1 post) OR logs `old.reddit.com returned` warnings (rate-limited is acceptable)
- Each post has keys: `source="reddit"`, `subreddit`, `title`, `score`, `polarity`, `subjectivity`
- No `json()` call errors (old JSON API signature)
**Why adversarial**: If scraping is broken (wrong selectors, wrong URL), the list will be empty and no posts will have valid structure. If the old JSON code was accidentally left, the response parsing would fail differently.

## Test 3: Sentiment Summary Uses Single API Call (Shell)
**Command**: Python script that patches `fetch_reddit_sentiment` with a counter, calls `social_media_listener_tool.invoke({"ticker": "AAPL"})`, checks counter
**Pass criteria**: `fetch_reddit_sentiment` is called exactly ONCE (via `get_sentiment_features`), not twice
**Why adversarial**: Before the fix, it was called twice — this test would catch a regression.

## Test 4: Saved Medians Used in Prediction (Shell)
**Command**: Python script that loads the saved model, checks `predictor.feature_medians is not None`, and verifies predictions don't use 0.0 fill
**Pass criteria**:
- `predictor.feature_medians` is a pandas Series with length > 0
- `MEDIANS_PATH` file exists on disk
**Why adversarial**: Before the fix, `feature_medians` was not saved. If the save/load is broken, the attribute will be None.

## Test 5: Streamlit Social Sentiment Page (GUI)
**Steps**:
1. Navigate to "Social Sentiment" page via sidebar
2. Enter "AAPL" in ticker input
3. Click "Get Sentiment"
4. Wait for results
**Pass criteria**:
- Output contains "Social Media Sentiment for AAPL"
- Output shows "Reddit: N posts" where N >= 0 (with polarity value)
- Output shows "Finviz News: N headlines" where N > 0
- Output shows "StockTwits: N messages"
- No Python error/traceback displayed
**Why adversarial**: If old.reddit.com scraping is broken, Reddit would show 0 posts. Finviz should still work. The formatted output proves the full pipeline works.

## Test 6: LLM Agent Chat (GUI)
**Steps**:
1. Navigate to "AI Stock Advisor" page
2. Enter query: "What is the current price and sentiment for NVDA?"
3. Wait for agent response (30-60s)
**Pass criteria**:
- Agent responds with structured analysis (not an error)
- Response mentions price data, sentiment, and/or technical indicators
- No "Error" or traceback in the response
**Why adversarial**: The agent internally uses `social_media_listener_tool` which exercises the duplicate-call fix. If `get_sentiment_summary` signature change broke the tool, the agent would fail.

## Test 7: Trending Tickers Scraping (GUI)
**Steps**:
1. Navigate to "Social Sentiment" page
2. Click "Trending Tickers" tab
3. Click "Find Trending Tickers"
**Pass criteria**:
- Either shows trending tickers (numbered list) OR "No trending tickers found" (both valid — depends on Reddit availability)
- No Python error/traceback
**Why adversarial**: This exercises the `get_trending_tickers_from_social` function which was rewritten to use old.reddit.com scraping. If the HTML parsing is broken, it would crash rather than return empty.
