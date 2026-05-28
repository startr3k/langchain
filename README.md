# Stock Predictor & AI Investment Advisor

An AI-powered stock prediction and recommendation system that combines AutoML with an OpenAI LLM agent to identify high-return NASDAQ stocks.

## Architecture

```
stock_predictor/
├── data/
│   ├── yfinance_client.py    # YFinance data fetching & technical indicators
│   ├── sentiment.py          # Social media sentiment (Reddit, Finviz, StockTwits)
│   └── feature_engineering.py # Feature pipeline combining all data sources
├── models/
│   ├── automl_model.py       # FLAML AutoML model for 6-month return prediction
│   └── saved/                # Persisted trained models
├── agent/
│   ├── agent.py              # OpenAI LLM agent with ReAct loop
│   └── tools.py              # LangChain tools (YFinance, Sentiment, Predictor)
├── app/
│   └── streamlit_app.py      # Streamlit frontend
└── tests/                    # Unit and integration tests
```

## Features

### 1. AutoML Prediction Model
- Trains on historical YFinance data + social media sentiment features
- Uses FLAML to automatically select the best model (XGBoost, LightGBM, Random Forest)
- 40+ features including technical indicators, fundamentals, and sentiment scores
- Predicts 6-month forward returns for any NASDAQ stock

### 2. Social Media Sentiment Analysis
- **Reddit**: Scans r/wallstreetbets, r/stocks, r/investing, r/StockMarket, etc.
- **Finviz**: Scrapes news headlines and computes sentiment
- **StockTwits**: Fetches messages with bullish/bearish labels
- Aggregates sentiment into model-ready features (polarity, mention counts, bull/bear ratio)

### 3. OpenAI LLM Agent
- Expert investment analyst powered by GPT-4o
- Four tools: YFinance, Social Media Listener, Stock Predictor, Trending Scanner
- ReAct-style reasoning loop for multi-step analysis
- Structured recommendations with risk assessment

### 4. Streamlit Frontend
- Chat interface for AI advisor interaction
- Individual stock analysis with candlestick charts
- Social sentiment dashboard
- Model training interface with feature importance visualization
- Batch prediction scanner

## Setup

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

Or enter it directly in the Streamlit sidebar.

## Usage

### Run the Streamlit App
```bash
streamlit run stock_predictor/app/streamlit_app.py
```

### Train the Model (CLI)
```python
from stock_predictor.models.automl_model import StockReturnPredictor

predictor = StockReturnPredictor()
metrics = predictor.train(time_budget=120)
print(metrics)
```

### Run the Agent (CLI)
```python
from stock_predictor.agent.agent import run_agent

response = run_agent("Which NASDAQ stocks could return 100% in 6 months?")
print(response)
```

## Testing

```bash
# Run all tests
pytest stock_predictor/tests/ -v

# Run with coverage
pytest stock_predictor/tests/ -v --cov=stock_predictor --cov-report=term-missing
```

## Disclaimer

This tool is for educational and research purposes only. Predictions are model-based estimates, not financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.
