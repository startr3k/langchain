# Test Plan: HPO + Social Media Listener + MLOps Pipeline

## What Changed
1. HPO-optimized params (LTR_ENSEMBLE_WEIGHT=0.6, VOLATILITY_SCORE_ALPHA=0.25)
2. New "Social Media Listener" page — top 20 hottest stocks from Reddit/StockTwits, filtered to Dow/S&P/NASDAQ ≥$1B
3. New "Daily Picks Pipeline" page — run daily top-10 picks, evaluate ground truth, precision chart
4. New "Daily Picks History" page — browse picks by date with SHAP explanations

## Pre-conditions
- Trained model exists on disk (ltr_model.json, stock_predictor_model.pkl)
- No daily_picks.csv exists yet (tests empty state → creation)
- Streamlit running on http://localhost:8501

---

## Test 1: Navigation — All 10 pages appear in sidebar
**Steps:** Open Streamlit, inspect sidebar radio buttons
**Pass criteria:** Sidebar shows exactly 10 items: "Top Recommendations", "Stock Chart", "AI Stock Advisor", "Stock Analysis", "Social Sentiment", "Social Media Listener", "Model Training", "Batch Predictions", "Daily Picks Pipeline", "Daily Picks History"
**Fail signal:** Missing any of the 3 new pages, or old pages missing

## Test 2: Social Media Listener — Eligible ticker universe loads
**Steps:** Click "Social Media Listener" in sidebar. Expand "Eligible Ticker Universe" expander.
**Pass criteria:** Shows a count of eligible tickers (should be >100 for S&P 500 + NASDAQ-100 + Dow), with ticker symbols listed. The description should mention "≥$1B market cap".
**Fail signal:** Shows 0 tickers, or fallback list only (~130 tickers), or expander doesn't render

## Test 3: Social Media Listener — Refresh shows trending stocks
**Steps:** Click "🔄 Refresh Social Media Data" button
**Pass criteria:** Either (a) shows a table with columns Rank, Ticker, Mentions, Sentiment, Upvotes, Comments, Engagement, Sources — with at least 1 row, OR (b) shows "No trending tickers found" info message (acceptable if Reddit/StockTwits are rate-limited)
**Fail signal:** Error/crash, no response, or page hangs indefinitely

## Test 4: Daily Picks Pipeline — Empty state
**Steps:** Click "Daily Picks Pipeline" in sidebar
**Pass criteria:** Page loads with title "📋 Daily Picks Pipeline (MLOps)". Left column shows "Generate Daily Picks" with top_k selector (default 10) and min market cap selector. Right column shows "Evaluate Ground Truth". Bottom shows "No evaluated picks yet" info message.
**Fail signal:** Page crash, missing columns, or components don't render

## Test 5: Daily Picks Pipeline — Run pipeline generates CSV
**Steps:** Click "▶️ Run Daily Pipeline" button (with default top_k=10, min_mcap=$100M)
**Pass criteria:** Spinner shows "Running daily picks pipeline...", then success message "Recorded N picks!" with N=10. A dataframe appears showing the picks with columns including ticker, date, probability, close_price.
**Fail signal:** Error message, 0 picks generated, crash, or missing columns

## Test 6: Daily Picks Pipeline — Ground truth evaluation
**Steps:** After Test 5, click "🔍 Evaluate Ground Truth" button
**Pass criteria:** Either (a) shows "No picks old enough to evaluate yet" (if picks were just created today — expected), or (b) shows precision metrics. Should not crash.
**Fail signal:** Error/crash

## Test 7: Daily Picks History — Shows today's picks
**Steps:** Click "Daily Picks History" in sidebar
**Pass criteria:** Page loads, date dropdown shows today's date, table displays the 10 picks from Test 5 with columns: rank, ticker, probability, signal, close_price, volume_surge_3d, sentiment_score, etc. SHAP explanations section shows per-ticker SHAP feature text.
**Fail signal:** "No picks recorded yet" message (shouldn't happen after Test 5), crash, or missing data columns

## Test 8: Daily Picks History — Precision chart empty state
**Steps:** Scroll down to "Precision Over Time (All Dates)" section
**Pass criteria:** Shows "No ground-truth evaluations available yet" info message (since picks are too new)
**Fail signal:** Chart renders with incorrect data, or crash

## Test 9: Daily Picks Pipeline — CSV download
**Steps:** Go back to "Daily Picks Pipeline", expand "Raw CSV Data" expander
**Pass criteria:** Shows dataframe with all CSV columns (date, rank, ticker, close_price, probability, signal, ensemble_score, ltr_score, classification_score, volume_surge_3d, regime_confidence, ticker_calibration, volatility_20d, sentiment_score, sentiment_mentions, shap_top_features, market_cap, sector, max_upside_pct, hit_20pct, ground_truth_date). "Download CSV" button is visible.
**Fail signal:** Missing columns, empty dataframe, or no download button
