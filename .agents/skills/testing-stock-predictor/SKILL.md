---
name: testing-stock-predictor
description: Test the AI Stock Predictor system end-to-end. Use when verifying stock predictor UI, model training, sentiment analysis, or LLM agent changes.
---

# Testing the Stock Predictor System

## Prerequisites

- Python 3.12+ with dependencies from `requirements.txt`
- Streamlit must be running: `streamlit run stock_predictor/app/streamlit_app.py --server.port 8501 --server.headless true`
- For LLM agent testing: `OPENAI_API_KEY` environment variable must be set

## Devin Secrets Needed

- `OPENAI_API_KEY` — Required for LLM agent chat (Test 6). All other tests work without it.

## Test Procedure

### 1. Unit Tests (Shell)
```bash
cd /home/ubuntu/repos/langchain
python -m pytest stock_predictor/tests/ -v --tb=short
```
All tests use mocked external APIs. Expect 45+ tests to pass.

### 2. CLI Model Training (Shell)
```python
from stock_predictor.models.automl_model import StockReturnPredictor
predictor = StockReturnPredictor()
metrics = predictor.train(tickers=["AAPL", "MSFT", "NVDA"], time_budget=10, include_sentiment=False)
# Verify: metrics['best_estimator'] is one of xgboost/lgbm/rf/extra_tree
result = predictor.predict_ticker("TSLA")
# Verify: result['predicted_return_6m'] is a finite float
```
Training with `include_sentiment=False` avoids slow/rate-limited Reddit/StockTwits APIs.

### 3. Streamlit App (GUI — record this)
Start the app, then test each page:

1. **AI Stock Advisor** — Verify title, chat input, 3 quick prompt buttons render
2. **Stock Analysis** — Enter "AAPL", click Analyze. Verify: candlestick chart, company info ("Apple Inc."), model prediction (if trained), sentiment text
3. **Social Sentiment** — Click "Get Sentiment" for default ticker. Verify formatted output with polarity, mention counts, source breakdowns
4. **Model Training** — Verify sliders (tickers: 5-50 default 20, budget: 30-600 default 120), sentiment checkbox, "Start Training" button. If model was trained via CLI, shows "Trained model found" with feature importances
5. **Training Data Preview** — Set ticker slider to ~7, uncheck sentiment (faster), click "Generate & Preview Training Data". Verify: success message with sample/ticker/column counts, 3 metric cards (Total Samples, Tickers, Features), plotly histogram of Forward_Return_6M distribution, scrollable data table with feature columns, "Download Training Data as CSV" button. Click download and verify CSV file appears in browser downloads.
6. **Batch Predictions** — Verify ticker textarea pre-filled with NASDAQ top 20, "Run Predictions" button

### 4. LLM Agent Chat (GUI — requires OPENAI_API_KEY)
1. Enter API key in sidebar
2. Type query: "What is the current price and sentiment for NVDA?"
3. Wait 30-60s for agent to call tools and respond
4. Verify response contains: price data, technical indicators, sentiment analysis, fundamentals, recommendation

## Known Issues & Workarounds

- **Reddit API returns 403**: Rate-limited from cloud environments. Sentiment falls back to Finviz news headlines. Not a bug — code handles gracefully.
- **StockTwits API returns 403**: Same as Reddit. Code handles gracefully with logging.
- **Finviz news**: Works reliably as primary sentiment source. Typically returns 20-30 headlines per ticker.
- **Model training with sentiment**: Use `include_sentiment=False` for faster training. Sentiment features add ~17 features but require API calls that might be rate-limited.
- **Clipboard for API key**: If pasting API key into Streamlit sidebar, install `xclip` first: `sudo apt-get install -y xclip`, then `echo -n "$OPENAI_API_KEY" | xclip -selection clipboard` and Ctrl+V in the browser.

## Tips

- Train the model via CLI before GUI testing so the Model Training page shows feature importances and Stock Analysis page shows predictions.
- The Streamlit app reads `OPENAI_API_KEY` from environment on startup as the default sidebar value, but you can also paste it manually.
- YFinance is the most reliable external API — it consistently works from cloud environments.
- When testing model predictions, values are typically in the -50% to +100% range for 6-month returns.
