"""Streamlit frontend for the Stock Predictor & Recommendation Agent."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_predictor.agent.agent import run_agent
from stock_predictor.data.feature_engineering import (
    ALL_FEATURE_NAMES,
    TARGET_COLUMN,
    build_training_dataset,
)
from stock_predictor.data.sentiment import get_sentiment_summary, get_trending_tickers_from_social
from stock_predictor.data.yfinance_client import get_stock_data, get_stock_info, NASDAQ_TOP_TICKERS
from stock_predictor.models.automl_model import StockReturnPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Predictor & AI Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Stock Predictor & AI Advisor")
st.sidebar.markdown("---")

# API Key
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", ""),
    help="Required for the AI agent. Set OPENAI_API_KEY env var or enter here.",
)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

model_choice = st.sidebar.selectbox(
    "OpenAI Model",
    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    index=0,
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "AI Stock Advisor",
        "Stock Analysis",
        "Social Sentiment",
        "Model Training",
        "Batch Predictions",
    ],
)

# ---------------------------------------------------------------------------
# Page: AI Stock Advisor (Chat)
# ---------------------------------------------------------------------------
if page == "AI Stock Advisor":
    st.title("AI Stock Investment Advisor")
    st.markdown(
        "Ask the AI agent for stock recommendations. It uses YFinance data, "
        "social media sentiment, and a trained prediction model to provide analysis."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Ask about stocks (e.g., 'Which NASDAQ stocks could return 100% in 6 months?')"
    )

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if not api_key:
            with st.chat_message("assistant"):
                st.error("Please enter your OpenAI API key in the sidebar.")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Error: OpenAI API key required."}
            )
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing... (this may take a minute as the agent queries multiple data sources)"):
                    try:
                        response = run_agent(
                            query=user_input,
                            model=model_choice,
                            api_key=api_key,
                        )
                        st.markdown(response)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": error_msg}
                        )

    # Quick prompts
    st.markdown("---")
    st.subheader("Quick Prompts")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Top NASDAQ picks for 100% return"):
            st.session_state["_quick_prompt"] = (
                "Which NASDAQ stocks have the highest potential for 100% return "
                "in the next 6 months? Scan trending stocks and give me your top 5 picks."
            )
            st.rerun()
    with col2:
        if st.button("Analyze trending meme stocks"):
            st.session_state["_quick_prompt"] = (
                "Analyze the currently trending meme stocks on social media. "
                "Which ones have genuine upside potential based on fundamentals?"
            )
            st.rerun()
    with col3:
        if st.button("AI/Tech sector analysis"):
            st.session_state["_quick_prompt"] = (
                "Analyze AI and tech stocks in NASDAQ. Which ones are most likely "
                "to outperform in the next 6 months based on sentiment and technicals?"
            )
            st.rerun()

    # Handle quick prompt
    if "_quick_prompt" in st.session_state:
        prompt = st.session_state.pop("_quick_prompt")
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Stock Analysis
# ---------------------------------------------------------------------------
elif page == "Stock Analysis":
    st.title("Individual Stock Analysis")

    ticker = st.text_input("Enter Ticker Symbol", value="NVDA").upper()

    if st.button("Analyze"):
        with st.spinner(f"Fetching data for {ticker}..."):
            col1, col2 = st.columns(2)

            # Price chart
            with col1:
                st.subheader(f"{ticker} Price History")
                df = get_stock_data(ticker, period="1y")
                if not df.empty:
                    import plotly.graph_objects as go

                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=df["Date"],
                                open=df["Open"],
                                high=df["High"],
                                low=df["Low"],
                                close=df["Close"],
                            )
                        ]
                    )
                    fig.update_layout(
                        title=f"{ticker} — 1 Year",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No price data available.")

            # Company info
            with col2:
                st.subheader("Company Info")
                info = get_stock_info(ticker)
                if info:
                    for key, value in info.items():
                        st.metric(key, str(value))
                else:
                    st.warning("No company info available.")

            # Prediction
            st.subheader("Model Prediction")
            try:
                predictor = StockReturnPredictor()
                predictor.load()
                result = predictor.predict_ticker(ticker)
                if result.get("predicted_return_6m") is not None:
                    ret = result["predicted_return_6m"]
                    st.metric(
                        "Predicted 6-Month Return",
                        f"{ret * 100:.2f}%",
                        delta=f"{ret * 100:.2f}%",
                    )
                else:
                    st.info(result.get("error", "Prediction unavailable."))
            except FileNotFoundError:
                st.info("Model not trained yet. Go to 'Model Training' to train.")

            # Sentiment
            st.subheader("Social Media Sentiment")
            sentiment = get_sentiment_summary(ticker)
            st.text(sentiment)


# ---------------------------------------------------------------------------
# Page: Social Sentiment
# ---------------------------------------------------------------------------
elif page == "Social Sentiment":
    st.title("Social Media Sentiment Dashboard")

    tab1, tab2 = st.tabs(["Ticker Sentiment", "Trending Tickers"])

    with tab1:
        ticker = st.text_input("Ticker for sentiment analysis", value="TSLA").upper()
        if st.button("Get Sentiment"):
            with st.spinner("Fetching sentiment data..."):
                summary = get_sentiment_summary(ticker)
                st.text(summary)

    with tab2:
        if st.button("Find Trending Tickers"):
            with st.spinner("Scanning Reddit for trending stocks..."):
                trending = get_trending_tickers_from_social()
                if trending:
                    st.success(f"Found {len(trending)} trending tickers")
                    for i, t in enumerate(trending, 1):
                        st.write(f"{i}. **{t}**")
                else:
                    st.info("No trending tickers found at the moment.")


# ---------------------------------------------------------------------------
# Page: Model Training
# ---------------------------------------------------------------------------
elif page == "Model Training":
    st.title("AutoML Model Training")
    st.markdown(
        "Train the stock return prediction model using historical data from "
        "YFinance combined with social media sentiment features."
    )

    col1, col2 = st.columns(2)
    with col1:
        num_tickers = st.slider("Number of training tickers", 5, 50, 20)
    with col2:
        time_budget = st.slider("AutoML time budget (seconds)", 30, 600, 120)

    include_sentiment = st.checkbox("Include social media sentiment features", value=True)

    tickers = NASDAQ_TOP_TICKERS[:num_tickers]
    st.write(f"Training tickers: {', '.join(tickers)}")

    train_col, data_col = st.columns(2)

    with train_col:
        if st.button("Start Training", type="primary"):
            predictor = StockReturnPredictor()
            progress_bar = st.progress(0)
            status = st.empty()

            status.info("Building training dataset (this may take several minutes)...")
            progress_bar.progress(10)

            try:
                metrics = predictor.train(
                    tickers=tickers,
                    time_budget=time_budget,
                    include_sentiment=include_sentiment,
                )
                progress_bar.progress(100)
                status.success("Training complete!")

                st.subheader("Training Results")
                st.json(metrics)

                # Feature importance
                importance = predictor.get_feature_importance(top_n=15)
                if importance:
                    st.subheader("Top Feature Importances")
                    import plotly.express as px
                    import pandas as pd

                    imp_df = pd.DataFrame(importance, columns=["Feature", "Importance"])
                    fig = px.bar(
                        imp_df, x="Importance", y="Feature",
                        orientation="h", title="Feature Importance",
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                progress_bar.progress(0)
                status.error(f"Training failed: {e}")

    with data_col:
        if st.button("Generate & Preview Training Data"):
            with st.spinner("Building training dataset (fetching data from YFinance)..."):
                try:
                    training_df = build_training_dataset(
                        tickers, include_sentiment=include_sentiment,
                    )
                    if training_df.empty:
                        st.warning("No training data could be generated.")
                    else:
                        st.session_state["training_data"] = training_df
                        st.success(
                            f"Generated {len(training_df)} training samples "
                            f"across {training_df['Ticker'].nunique()} tickers "
                            f"with {len(training_df.columns)} columns."
                        )
                except Exception as e:
                    st.error(f"Error generating training data: {e}")

    # Display training data if available
    if "training_data" in st.session_state:
        training_df = st.session_state["training_data"]
        st.markdown("---")
        st.subheader("Training Data Preview")

        # Summary stats
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Samples", len(training_df))
        with col_b:
            st.metric("Tickers", training_df["Ticker"].nunique())
        with col_c:
            st.metric("Features", len(training_df.columns) - 2)  # exclude Ticker & target

        # Target distribution
        if TARGET_COLUMN in training_df.columns:
            target_vals = training_df[TARGET_COLUMN].dropna()
            if not target_vals.empty:
                import plotly.express as px

                st.subheader("Target Distribution (6-Month Forward Return)")
                fig = px.histogram(
                    target_vals, nbins=50,
                    labels={"value": "6-Month Return", "count": "Count"},
                    title="Distribution of Forward Returns",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("Data Table")
        st.dataframe(training_df, use_container_width=True, height=400)

        # Download button
        csv = training_df.to_csv(index=False)
        st.download_button(
            label="Download Training Data as CSV",
            data=csv,
            file_name="stock_predictor_training_data.csv",
            mime="text/csv",
            type="primary",
        )

    # Model status
    st.markdown("---")
    st.subheader("Model Status")
    try:
        predictor = StockReturnPredictor()
        predictor.load()
        st.success("Trained model found and loaded successfully.")
        importance = predictor.get_feature_importance(top_n=10)
        if importance:
            st.write("Top features:", [f"{n} ({v:.3f})" for n, v in importance])
    except FileNotFoundError:
        st.warning("No trained model found. Train the model above.")


# ---------------------------------------------------------------------------
# Page: Batch Predictions
# ---------------------------------------------------------------------------
elif page == "Batch Predictions":
    st.title("Batch Stock Predictions")
    st.markdown("Run the prediction model on multiple stocks to find the best candidates.")

    tickers_input = st.text_area(
        "Enter tickers (comma-separated)",
        value=", ".join(NASDAQ_TOP_TICKERS[:20]),
    )

    if st.button("Run Predictions", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        try:
            predictor = StockReturnPredictor()
            predictor.load()

            results = []
            progress = st.progress(0)
            for i, ticker in enumerate(tickers):
                progress.progress((i + 1) / len(tickers))
                result = predictor.predict_ticker(ticker)
                if result.get("predicted_return_6m") is not None:
                    results.append(result)

            results.sort(key=lambda x: x["predicted_return_6m"], reverse=True)

            st.subheader(f"Results ({len(results)} stocks)")

            import pandas as pd

            df = pd.DataFrame(results)
            df = df.rename(columns={
                "ticker": "Ticker",
                "predicted_return_6m": "Predicted Return (6M)",
                "predicted_return_6m_pct": "Predicted Return %",
            })
            st.dataframe(df, use_container_width=True)

            # Highlight stocks with >100% predicted return
            high_return = [r for r in results if r["predicted_return_6m"] >= 1.0]
            if high_return:
                st.success(
                    f"Found {len(high_return)} stocks with predicted 100%+ return!"
                )
                for r in high_return:
                    st.write(
                        f"**{r['ticker']}**: {r['predicted_return_6m_pct']} predicted return"
                    )
            else:
                st.info("No stocks with predicted 100%+ return found in this batch.")

        except FileNotFoundError:
            st.error("Model not trained yet. Go to 'Model Training' first.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    "Disclaimer: This tool is for educational purposes only. "
    "Predictions are model-based estimates, not financial advice. "
    "Past performance does not guarantee future results."
)
