# E2E Test Plan — Stock Predictor App

## What Changed (PRs #7, #8, #9 merged to main)
- LTR ensemble + HPO optimization
- Social Media Listener page (multi-source: Yahoo/Finviz/GDELT/Reddit)
- Daily Picks Pipeline page (scheduler controls)
- Daily Picks History page (date filter, SHAP, precision chart)
- Top Recommendations: loads from daily picks CSV if available, Regenerate button
- Close price/sector fix (batch yf.download + retry)
- GPT model dropdown expanded (gpt-4.1, o3, o4-mini, etc.)
- Ticker universe caching + Refresh button
- Social buzz persistence + hot stock indicator

## Test Environment
- Streamlit on localhost:8501
- Model trained and saved (LTR + classification + calibration)
- Daily picks CSV empty for today (tests Regenerate flow)
- Eligible tickers cache exists
- OPENAI_API_KEY available

---

## Test 1: Sidebar Navigation — All 10 Pages Render
**Steps:** Click each of the 10 sidebar nav items in order.
**Pass criteria:** Each page renders its title without errors:
1. "Top Stock Recommendations"
2. "Stock Chart Dashboard"
3. "AI Stock Advisor"
4. "Stock Analysis"
5. "Social Sentiment"
6. "Social Media Listener" (with fire emoji)
7. "Model Training"
8. "Batch Predictions"
9. "Daily Picks Pipeline (MLOps)"
10. "Daily Picks History"

**Fail if:** Any page shows a Streamlit error traceback or is missing from sidebar.

## Test 2: GPT Model Dropdown
**Steps:** Click the OpenAI Model dropdown in the sidebar.
**Pass criteria:** Dropdown shows at minimum: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o3`, `o3-mini`, `o4-mini`. Default selection is `gpt-4.1`.
**Fail if:** Only old models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo) are shown, or gpt-4.1 is missing.

## Test 3: Top Recommendations — Empty State + Regenerate
**Steps:**
1. Navigate to "Top Recommendations"
2. Verify empty state message: "No picks available for today"
3. Click "Regenerate Recommendations" button
4. Wait for pipeline to complete (~1-2 min)
5. Verify results table appears with 10 rows

**Pass criteria:**
- Empty state shows info message with "No picks available"
- After Regenerate: table shows 10 rows with columns including Ticker, Model P(>=20%), Signal
- **Close Price column has actual dollar values (not NaN/empty)** — this is the PR #9 fix
- **Sector column has actual sector names (e.g., "Technology", not NaN/empty)** — PR #9 fix
- Prediction Details expanders show SHAP feature contributions

**Fail if:** Close price or sector are empty/NaN for all picks (the original bug).

## Test 4: Social Media Listener — Eligible Tickers + Refresh
**Steps:**
1. Navigate to "Social Media Listener"
2. Expand "Eligible Ticker Universe" expander
3. Verify ticker count is displayed (should be > 100)
4. Click "Refresh Market Buzz Data" button
5. Wait for scan to complete

**Pass criteria:**
- Eligible ticker count shown as a number > 100
- "Refresh Tickers" button is visible in the expander
- After clicking Refresh Market Buzz: either shows trending stocks table OR "No trending tickers found" info message (both are valid — depends on API availability)
- No Streamlit error tracebacks

**Fail if:** Page crashes, eligible ticker count is 0, or an unhandled exception appears.

## Test 5: Daily Picks Pipeline — Scheduler Controls
**Steps:**
1. Navigate to "Daily Picks Pipeline"
2. Verify scheduler UI elements are present
3. Verify "Save & Start Schedule" button exists

**Pass criteria:**
- Title shows "Daily Picks Pipeline (MLOps)"
- "Schedule Pipeline" subheader visible
- Frequency dropdown with "daily"/"weekly" options
- Hour slider (0-23) and Minute slider (0-59)
- Day of week dropdown
- Min market cap dropdown with $100M/$500M/$1B options
- "Save & Start Schedule" button present
- "Evaluate Ground Truth" section present below

**Fail if:** Scheduler controls are missing or page shows error.

## Test 6: Daily Picks History — Shows Today's Picks
**Steps:**
1. Navigate to "Daily Picks History"
2. Verify today's picks from the Regenerate in Test 3 appear
3. Check summary metrics and SHAP explanations

**Pass criteria:**
- Date selector shows today's date
- Summary metrics: Picks count = 10, Avg Probability shows a percentage
- Picks table shows ticker, probability, signal, close_price columns
- SHAP explanations section shows feature contributions (not "nan")
- Sector Breakdown expander available

**Fail if:** No picks shown, SHAP values are all NaN, or close_price column is empty.

## Test 7: Stock Chart — Basic Rendering
**Steps:**
1. Navigate to "Stock Chart"
2. Enter "AAPL" and select period
3. Verify chart renders

**Pass criteria:** Candlestick chart renders with AAPL price data and volume bars below.
**Fail if:** Chart fails to render or shows error.
