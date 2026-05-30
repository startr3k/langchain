# Test Plan v5: Scheduler + Optimized Pipeline + Social Media Fix

## What Changed (latest commits)
1. **Optimized pipeline**: Batch scoring from cached training CSV (617 tickers in ~5s vs 25+ min)
2. **Market cap filtering**: Uses yfinance `fast_info` for all 617 tickers, then filters to >=1B (309 pass)
3. **Scheduler**: APScheduler-based background scheduler with Streamlit UI (daily/weekly at configurable time)
4. **Wikipedia fix**: Added User-Agent header — eligible tickers now 611 (was 29 due to 403 errors)

## Pre-conditions
- Streamlit running on http://localhost:8501
- daily_picks.csv already has 10 picks from earlier test run
- Trained model files exist on disk

---

## Test 1: Navigation — All 10 pages in sidebar
**Steps:** Open Streamlit app, inspect sidebar
**Pass:** Sidebar has exactly 10 items including "Social Media Listener", "Daily Picks Pipeline", "Daily Picks History"
**Fail:** Missing pages or wrong count

## Test 2: Social Media Listener — Eligible ticker count > 100
**Steps:** Click "Social Media Listener", expand "Eligible Ticker Universe"
**Pass:** Shows count >= 400 eligible tickers (S&P 500 + NASDAQ-100 + Dow, filtered by $1B). Previously was 29 (only Dow).
**Fail:** Count <= 100, or only showing ~29 tickers (Wikipedia fetch still broken)

## Test 3: Social Media Listener — Refresh trending stocks
**Steps:** Click "Refresh Social Media Data"
**Pass:** Either shows trending stocks table OR "No trending tickers found" info (acceptable if APIs rate-limited)
**Fail:** Crash, unhandled exception, or page hangs

## Test 4: Daily Picks Pipeline — Page loads with scheduler section
**Steps:** Click "Daily Picks Pipeline"
**Pass:** Page shows (1) "Generate Daily Picks" section, (2) "Evaluate Ground Truth" section, (3) "Schedule Pipeline" section with frequency/hour/minute controls, (4) "Precision Over Time" section
**Fail:** Missing scheduler section, crash, or missing UI controls

## Test 5: Daily Picks Pipeline — Scheduler configuration
**Steps:** In the "Schedule Pipeline" section, expand "Configure Schedule", set frequency=daily, hour=14, minute=30, click "Save & Start Schedule"
**Pass:** Success message appears showing next run time. Status changes to show "Pipeline is scheduled daily at 14:30 UTC" with a "Stop Schedule" button.
**Fail:** Error on save, no confirmation, or schedule status doesn't update

## Test 6: Daily Picks Pipeline — Stop scheduler
**Steps:** Click "Stop Schedule" button
**Pass:** Schedule status changes to "No schedule active" info message
**Fail:** Error, or schedule still shows as active

## Test 7: Daily Picks History — Shows existing picks
**Steps:** Click "Daily Picks History"
**Pass:** Date picker shows today's date (2026-05-30), table shows 10 picks with columns including ticker, probability, close_price, sector, market_cap. SHAP explanations visible.
**Fail:** "No picks recorded" (shouldn't happen — CSV has data), or missing columns

## Test 8: Daily Picks Pipeline — CSV download
**Steps:** Go to "Daily Picks Pipeline", expand "Raw CSV Data"
**Pass:** Shows dataframe with 21 columns (date, rank, ticker, close_price, probability, signal, ensemble_score, ltr_score, classification_score, volume_surge_3d, regime_confidence, ticker_calibration, volatility_20d, sentiment_score, sentiment_mentions, shap_top_features, market_cap, sector, max_upside_pct, hit_20pct, ground_truth_date). "Download CSV" button visible.
**Fail:** Missing columns, empty table, or no download button
