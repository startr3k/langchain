"""Streamlit frontend for the Stock Predictor & Recommendation Agent."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
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
from stock_predictor.data.sentiment import (
    get_sentiment_features,
    get_sentiment_summary,
    get_trending_tickers_from_social,
)
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
        "Top Recommendations",
        "AI Stock Advisor",
        "Stock Analysis",
        "Social Sentiment",
        "Model Training",
        "Batch Predictions",
    ],
)

# ---------------------------------------------------------------------------
# Page: Top Recommendations
# ---------------------------------------------------------------------------
if page == "Top Recommendations":
    st.title("Top Stock Recommendations")
    st.markdown(
        "Combines **model probability** (P(≥30% peak gain in 3 months)) with "
        "**live sentiment** from Reddit, Finviz, and StockTwits to produce a "
        "composite score ranking."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        top_x = st.number_input(
            "Show top X results",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
        )
    with col_cfg2:
        sentiment_weight = st.slider(
            "Sentiment weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Weight for sentiment in composite score. "
                 "Model probability weight = 1 - sentiment weight.",
        )
    with col_cfg3:
        ticker_source = st.selectbox(
            "Ticker universe",
            ["NASDAQ Top 50", "NASDAQ Top 100", "Custom list"],
        )

    if ticker_source == "Custom list":
        custom_tickers = st.text_area(
            "Enter tickers (comma-separated)",
            value=", ".join(NASDAQ_TOP_TICKERS[:20]),
        )
        scan_tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    elif ticker_source == "NASDAQ Top 100":
        scan_tickers = NASDAQ_TOP_TICKERS[:100]
    else:
        scan_tickers = NASDAQ_TOP_TICKERS[:50]

    model_weight = 1.0 - sentiment_weight

    if st.button("Generate Recommendations", type="primary"):
        try:
            predictor = StockReturnPredictor()
            predictor.load()
        except FileNotFoundError:
            st.error("Model not trained yet. Go to 'Model Training' first.")
            st.stop()

        results = []
        progress_bar = st.progress(0, text="Running predictions...")
        status_text = st.empty()

        for i, ticker in enumerate(scan_tickers):
            progress_bar.progress(
                (i + 1) / len(scan_tickers),
                text=f"Processing {ticker} ({i+1}/{len(scan_tickers)})...",
            )

            # Model prediction
            pred = predictor.predict_ticker(ticker)
            prob = pred.get("probability_30pct_gain")
            if prob is None:
                continue

            # Live sentiment
            status_text.text(f"Fetching sentiment for {ticker}...")
            try:
                sent_feats = get_sentiment_features(ticker)
            except Exception:
                sent_feats = {}

            mean_polarity = sent_feats.get("sentiment_mean_polarity", 0.0)
            total_mentions = sent_feats.get("sentiment_total_mentions", 0)
            reddit_count = sent_feats.get("reddit_mention_count", 0)
            stocktwits_bull = sent_feats.get("stocktwits_bullish_count", 0)
            stocktwits_bear = sent_feats.get("stocktwits_bearish_count", 0)
            bull_bear_ratio = sent_feats.get("stocktwits_bull_bear_ratio", 1.0)

            # Normalize sentiment polarity from [-1, 1] to [0, 1]
            sentiment_score = (mean_polarity + 1.0) / 2.0

            # Composite score: weighted combination
            composite = model_weight * prob + sentiment_weight * sentiment_score

            results.append({
                "Ticker": ticker,
                "Model P(≥30%)": round(prob, 4),
                "Sentiment Score": round(sentiment_score, 4),
                "Composite Score": round(composite, 4),
                "Signal": pred.get("signal", "HOLD"),
                "Sentiment Polarity": round(mean_polarity, 3),
                "Total Mentions": total_mentions,
                "Reddit Mentions": reddit_count,
                "StockTwits Bull/Bear": f"{stocktwits_bull}/{stocktwits_bear}",
            })

        progress_bar.empty()
        status_text.empty()

        if not results:
            st.warning("No results — model could not generate predictions.")
            st.stop()

        # Sort by composite score and take top X
        results.sort(key=lambda x: x["Composite Score"], reverse=True)
        top_results = results[:top_x]

        st.subheader(f"Top {len(top_results)} Recommendations")
        st.caption(
            f"Score = {model_weight:.0%} × Model Probability + "
            f"{sentiment_weight:.0%} × Sentiment Score"
        )

        df = pd.DataFrame(top_results)
        st.dataframe(
            df.style.format({
                "Model P(≥30%)": "{:.1%}",
                "Sentiment Score": "{:.1%}",
                "Composite Score": "{:.1%}",
                "Sentiment Polarity": "{:+.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Highlight top picks
        buy_picks = [r for r in top_results if r["Signal"] == "BUY"]
        if buy_picks:
            st.success(f"**{len(buy_picks)} BUY signals** in top {len(top_results)}:")
            for r in buy_picks:
                model_p = r["Model P(≥30%)"]
                sent_p = r["Sentiment Polarity"]
                mentions = r["Total Mentions"]
                comp = r["Composite Score"]
                st.write(
                    f"**{r['Ticker']}** — "
                    f"Model: {model_p:.1%}, "
                    f"Sentiment: {sent_p:+.3f} "
                    f"({mentions} mentions), "
                    f"Composite: {comp:.1%}"
                )

        # Show all results table
        with st.expander(f"All {len(results)} scanned stocks"):
            all_df = pd.DataFrame(results)
            st.dataframe(
                all_df.style.format({
                    "Model P(≥30%)": "{:.1%}",
                    "Sentiment Score": "{:.1%}",
                    "Composite Score": "{:.1%}",
                    "Sentiment Polarity": "{:+.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Page: AI Stock Advisor (Chat)
# ---------------------------------------------------------------------------
elif page == "AI Stock Advisor":
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
        "Ask about stocks (e.g., 'Which NASDAQ stocks have the best 3-month outlook?')"
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
                "Which NASDAQ stocks have the highest potential for strong returns "
                "in the next 3 months? Scan trending stocks and give me your top 5 picks."
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
                "to outperform in the next 3 months based on sentiment and technicals?"
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
                if result.get("probability_30pct_gain") is not None:
                    prob = result["probability_30pct_gain"]
                    signal = result.get("signal", "HOLD")
                    col_a, col_b = st.columns(2)
                    col_a.metric(
                        "P(≥30% gain in 3M)",
                        f"{prob * 100:.1f}%",
                    )
                    col_b.metric("Signal", signal)
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
        time_budget = st.slider("AutoML time budget (seconds)", 30, 600, 300)

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

                st.session_state["training_metrics"] = metrics
                st.session_state["trained_predictor"] = predictor

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

                st.subheader("Target Distribution (3-Month Forward Return)")
                fig = px.histogram(
                    target_vals, nbins=50,
                    labels={"value": "3-Month Return", "count": "Count"},
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

    # ---- Model Evaluation Metrics ----
    if "training_metrics" in st.session_state:
        metrics = st.session_state["training_metrics"]
        st.markdown("---")
        st.subheader("Model Evaluation Metrics")

        # Ranking metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "AUC-ROC (Test)",
                f"{metrics.get('auc_roc', 0):.4f}",
                help="Area Under ROC Curve on held-out test set (0.5 = random, 1.0 = perfect)",
            )
        with m2:
            st.metric(
                "Avg Precision (Test)",
                f"{metrics.get('avg_precision', 0):.4f}",
                help="Area under the Precision-Recall curve (higher = better at ranking positives)",
            )
        with m3:
            st.metric("Best Model", metrics.get("best_estimator", "N/A"))
        with m4:
            st.metric("Training Samples", metrics.get("training_samples", 0))

        # Metrics at default threshold (0.5)
        st.markdown("**At default threshold (0.50):**")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with d2:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with d3:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        with d4:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")

        # Metrics at optimal threshold
        opt_thresh = metrics.get("optimal_threshold", 0.5)
        st.markdown(f"**At precision-optimized threshold ({opt_thresh:.2f}):**")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            st.metric("Precision", f"{metrics.get('precision_optimal', 0):.4f}")
        with o2:
            st.metric("Recall", f"{metrics.get('recall_optimal', 0):.4f}")
        with o3:
            st.metric("F1 Score", f"{metrics.get('f1_optimal', 0):.4f}")
        with o4:
            st.metric("Accuracy", f"{metrics.get('accuracy_optimal', 0):.4f}")

        # Overfitting check
        m5, m6 = st.columns(2)
        with m5:
            st.metric(
                "AUC (Train)",
                f"{metrics.get('auc_train', 0):.4f}",
                help="Training AUC — compare with Test AUC to detect overfitting",
            )
        with m6:
            st.metric(
                "AP (Train)",
                f"{metrics.get('ap_train', 0):.4f}",
                help="Training Average Precision — compare with Test AP",
            )

        with st.expander("Full Training Configuration"):
            st.json(metrics)

    # ---- Feature Importances ----
    show_importance = False
    predictor_for_importance = None
    if "trained_predictor" in st.session_state:
        predictor_for_importance = st.session_state["trained_predictor"]
        show_importance = True
    else:
        try:
            predictor_for_importance = StockReturnPredictor()
            predictor_for_importance.load()
            show_importance = True
        except FileNotFoundError:
            pass

    if show_importance and predictor_for_importance is not None:
        importance = predictor_for_importance.get_feature_importance(top_n=30)
        if importance:
            st.markdown("---")
            st.subheader("Feature Importances")
            import plotly.express as px

            imp_df = pd.DataFrame(importance, columns=["Feature", "Importance"])
            imp_df = imp_df.sort_values("Importance", ascending=True)
            fig = px.bar(
                imp_df, x="Importance", y="Feature",
                orientation="h",
                title="Top Feature Importances (Higher = More Predictive)",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                height=max(400, len(importance) * 22),
                yaxis_title="",
                xaxis_title="Importance Score",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Gain Chart ----
    if "training_metrics" in st.session_state:
        gain_data = st.session_state["training_metrics"].get("gain_chart")
        if gain_data and gain_data.get("percentages"):
            import plotly.graph_objects as go  # noqa: F811

            st.markdown("---")
            st.subheader("Cumulative Gain Chart")
            st.markdown(
                "Shows the percentage of actual 30%+ gainers captured when "
                "scoring the population from highest to lowest predicted "
                "probability. The further the model curve is above the "
                "diagonal (random), the better it is at ranking stocks."
            )

            fig_gain = go.Figure()
            fig_gain.add_trace(go.Scatter(
                x=gain_data["percentages"],
                y=gain_data["gains"],
                mode="lines+markers",
                name="Model",
                line=dict(color="#636EFA", width=2),
                marker=dict(size=5),
            ))
            fig_gain.add_trace(go.Scatter(
                x=gain_data["percentages"],
                y=gain_data["random"],
                mode="lines",
                name="Random (baseline)",
                line=dict(color="gray", width=1, dash="dash"),
            ))
            fig_gain.update_layout(
                xaxis_title="% of Population (ranked by model score)",
                yaxis_title="% of Actual Positives Captured",
                height=450,
                legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
                hovermode="x unified",
            )
            st.plotly_chart(fig_gain, use_container_width=True)

    # Model status (if no metrics in session)
    if "training_metrics" not in st.session_state:
        st.markdown("---")
        st.subheader("Model Status")
        try:
            predictor = StockReturnPredictor()
            predictor.load()
            st.success("Trained model found and loaded successfully.")
            st.info("Train a new model above to see detailed evaluation metrics (R², MAE, RMSE, MAPE).")
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
                if result.get("probability_30pct_gain") is not None:
                    results.append(result)

            results.sort(key=lambda x: x["probability_30pct_gain"], reverse=True)

            st.subheader(f"Results ({len(results)} stocks)")

            df = pd.DataFrame(results)
            df = df.rename(columns={
                "ticker": "Ticker",
                "probability_30pct_gain": "P(≥30% gain)",
                "probability_pct": "Probability %",
                "signal": "Signal",
            })
            st.dataframe(df, use_container_width=True)

            buy_signals = [r for r in results if r.get("signal") == "BUY"]
            if buy_signals:
                st.success(
                    f"Found {len(buy_signals)} stocks with BUY signal (≥50% probability of 30%+ gain)!"
                )
                for r in buy_signals:
                    st.write(
                        f"**{r['ticker']}**: {r['probability_pct']} probability of ≥30% gain"
                    )
            else:
                st.info("No stocks with BUY signal found in this batch.")

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
