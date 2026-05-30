# Test Plan v3 — Training Data Preview/Download + Code Review Fixes

## What Changed
1. **Training data preview + CSV download** (new feature): "Generate & Preview Training Data" button, summary metrics, histogram, data table, CSV download
2. **Reddit scraping**: Replaced JSON API with old.reddit.com HTML scraping
3. **5 code review bug fixes**: stale singleton, duplicate API calls, UnboundLocalError, train/serve skew, pyproject.toml

## Test Strategy
GUI testing via Streamlit for the primary new feature (training data), plus shell verification for unit tests and Reddit scraping.

---

## Test 1: Unit Tests (Shell)
**Command**: `python -m pytest stock_predictor/tests/ -v --tb=short`
**Pass**: All 45 tests pass, exit code 0

## Test 2: Training Data Preview + Download (GUI — PRIMARY TEST)
**Steps**:
1. Start Streamlit app, navigate to "Model Training" via sidebar
2. Set ticker slider to 5 (minimum, for speed)
3. Uncheck "Include social media sentiment features" (faster)
4. Click "Generate & Preview Training Data" button
5. Wait for data generation to complete
**Pass criteria**:
- Success message appears with format: "Generated N training samples across M tickers with C columns"
- N > 0 (training samples generated)
- 3 metric cards visible: "Total Samples" (>0), "Tickers" (5), "Features" (>30)
- Histogram titled "Distribution of Forward Returns" renders with bars
- Data table renders with scrollable rows
- "Download Training Data as CSV" button is visible and clickable
6. Click the download button
**Pass**: CSV file downloads successfully

## Test 3: Social Sentiment with Reddit Scraping (GUI)
**Steps**:
1. Navigate to "Social Sentiment" via sidebar
2. Enter "AAPL" in ticker input
3. Click "Get Sentiment"
**Pass criteria**:
- Output contains "Social Media Sentiment for AAPL"
- Shows "Reddit: N posts" line (N >= 0)
- Shows "Finviz News: N headlines" where N > 0
- No Python traceback/error

## Test 4: LLM Agent Chat (GUI)
**Steps**:
1. Navigate to "AI Stock Advisor" via sidebar
2. Type: "What is the current price and sentiment for NVDA?"
3. Wait for response
**Pass criteria**:
- Agent responds with structured analysis (not an error message)
- Response mentions at least one of: price, sentiment, technical indicators
