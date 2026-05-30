# E2E Test Plan: Earnings Transcript Sentiment + Source Text Transparency

## Constraint
Computer-use engine not available (no Chrome browser running). All testing is shell-based via Python scripts and HTTP requests. No recording needed.

## What Changed (User-Visible)
1. Earnings call transcript sentiment integrated into live composite score
2. Source texts with individual sentiments displayed per ticker
3. Expanded EDGAR XBRL tags reduced NaN rates
4. Stock Chart page should still work after code changes

---

## Test 1: Earnings transcript scraper returns real data for AAPL

**Command:**
```python
from stock_predictor.data.earnings_transcript import fetch_earnings_transcript
result = fetch_earnings_transcript("AAPL")
```

**Pass/Fail Criteria:**
- `result["transcript_sentiment"]` is a float (Loughran-McDonald score)
- `result["transcript_polarity"]` is a float (TextBlob polarity)
- `result["transcript_url"]` is a string containing "fool.com" (Motley Fool source)
- `result["transcript_source_texts"]` is a non-empty list of tuples
- Each tuple has 3 elements: (source_name: str, text_excerpt: str, polarity: float)
- `text_excerpt` contains at least 50 characters of actual transcript content (not empty/placeholder)

**Why adversarial:** If the scraper is broken (DDGS search fails, Motley Fool page structure changed, sentiment function errors), `transcript_sentiment` would be None and `transcript_source_texts` would be empty.

---

## Test 2: get_sentiment_features includes transcript data alongside social sentiment

**Command:**
```python
from stock_predictor.data.sentiment import get_sentiment_features
features = get_sentiment_features("AAPL")
```

**Pass/Fail Criteria:**
- `features["transcript_sentiment"]` is not None (proves transcript was fetched)
- `features["transcript_polarity"]` is a float
- `features["transcript_url"]` is a string URL
- `features["source_texts"]` is a list containing at least one tuple where source name contains "Earnings"
- `features["sentiment_mean_polarity"]` is a float (proves social sources also aggregated)
- `features["sentiment_total_mentions"]` >= 1 (at minimum, transcript counts)

**Why adversarial:** If the import/integration in sentiment.py is broken, `transcript_sentiment` would be missing from the dict entirely (KeyError) or always None. If `source_texts` doesn't include earnings data, the integration failed silently.

---

## Test 3: Model predictions work with the retrained model

**Command:**
```python
from stock_predictor.models.automl_model import StockReturnPredictor
predictor = StockReturnPredictor()
predictor.load()
result = predictor.predict_ticker("AAPL")
```

**Pass/Fail Criteria:**
- `result["probability_gain"]` is a float between 0.0 and 1.0
- `result["signal"]` is one of "BUY", "HOLD", or "SELL"
- `result["predicted_return_6m"]` or equivalent return field exists and is finite
- No exceptions thrown during prediction

**Why adversarial:** If the model was saved with wrong feature names or the scaler/threshold files are mismatched, prediction would crash with KeyError or produce NaN.

---

## Test 4: Streamlit app serves pages without errors

**Command:**
```bash
# Check Top Recommendations page loads
curl -s http://localhost:8501 | grep -o "Top Stock Recommendations"
# Check Stock Chart page is in the HTML
curl -s http://localhost:8501 | grep -o "Stock Chart"
```

**Pass/Fail Criteria:**
- HTTP 200 response
- HTML contains "Top Stock Recommendations" text
- HTML contains "Stock Chart" text (proving the page exists in navigation)

---

## Test 5: Internal fields (_source_texts, _transcript_url) are filtered from display

**Command:**
```python
# Simulate what the Streamlit app does
results = [{"Ticker": "AAPL", "Model P(≥20%)": 0.75, "_source_texts": [...], "_transcript_url": "..."}]
display_top = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
```

**Pass/Fail Criteria:**
- `display_top[0]` does NOT contain "_source_texts" key
- `display_top[0]` does NOT contain "_transcript_url" key
- `display_top[0]` DOES contain "Ticker" and "Model P(≥20%)" keys

---

## Test 6: Unit tests all pass (45 tests)

**Command:**
```bash
python -m pytest stock_predictor/tests/ -v --tb=short
```

**Pass/Fail Criteria:**
- All 45 tests pass
- 0 failures, 0 errors
