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
from stock_predictor.pipeline.daily_picks import (
    run_daily_picks,
    evaluate_ground_truth,
    get_precision_over_time,
    DEFAULT_CSV_PATH,
)
from stock_predictor.pipeline.scheduler import (
    get_schedule_config,
    schedule_pipeline,
    stop_schedule,
    is_scheduled,
    get_next_run,
    get_run_log,
    restore_schedule,
)
from stock_predictor.pipeline.email_notifier import (
    get_smtp_config,
    save_smtp_config,
    is_email_configured,
    send_test_email,
)
from stock_predictor.pipeline.social_listener import (
    get_social_hottest,
    get_eligible_tickers,
    get_ticker_cache_info,
    get_hot_tickers,
    get_social_buzz_data,
)

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

# Restore any saved pipeline schedule on app startup
restore_schedule()

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
    ["gpt-5.5", "gpt-5.5-mini", "gpt-5.5-pro", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o4-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    index=0,
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "Top Recommendations",
        "Stock Chart",
        "AI Stock Advisor",
        "Stock Analysis",
        "Social Sentiment",
        "Social Media Listener",
        "Model Training",
        "Batch Predictions",
        "Daily Picks Pipeline",
        "Daily Picks History",
    ],
)

# ---------------------------------------------------------------------------
# Page: Top Recommendations
# ---------------------------------------------------------------------------
if page == "Top Recommendations":
    st.title("Top Stock Recommendations")
    st.markdown(
        "Shows the **top daily stock picks** from the NASDAQ ticker universe.  "
        "Loads instantly from the daily picks pipeline if it has already run today.  "
        "🔥 = also trending on social media."
    )

    def _fmt_mcap(val) -> str:
        """Format market cap as human-readable string."""
        try:
            v = float(val)
        except (TypeError, ValueError):
            return "N/A"
        if v <= 0:
            return "N/A"
        if v >= 1e12:
            return f"${v / 1e12:.1f}T"
        if v >= 1e9:
            return f"${v / 1e9:.1f}B"
        if v >= 1e6:
            return f"${v / 1e6:.0f}M"
        return f"${v:,.0f}"

    # ── Helper: load today's picks from the daily_picks CSV ──────────
    def _load_todays_picks() -> pd.DataFrame | None:
        """Return today's picks from the CSV, or None if not available."""
        from datetime import date as _date

        csv_path = Path(DEFAULT_CSV_PATH)
        if not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path)
            today_str = _date.today().isoformat()
            today_df = df[df["date"] == today_str]
            if today_df.empty:
                return None
            return today_df
        except Exception:
            return None

    # ── Helper: convert CSV rows → display-friendly dicts ────────────
    def _csv_rows_to_results(df: pd.DataFrame, top_x: int) -> list[dict]:
        """Convert daily_picks CSV rows to the display format."""
        rows = df.head(top_x)
        results = []
        for _, row in rows.iterrows():
            prob = float(row.get("probability", row.get("ensemble_score", 0)))
            sent_score = float(row.get("sentiment_score", 0))
            # Sentiment score in CSV is raw polarity; normalize to [0,1]
            sentiment_normalized = (sent_score + 1.0) / 2.0 if abs(sent_score) <= 1 else sent_score
            vol_surge = row.get("volume_surge_3d")
            vol_surge_str = f"{vol_surge:.2f}x" if pd.notna(vol_surge) and vol_surge else "N/A"

            shap_str = row.get("shap_top_features", "")

            results.append({
                "Ticker": row["ticker"],
                "Model P(≥20%)": round(prob, 4),
                "Classifier P": float(row.get("cls_proba", 0)) if pd.notna(row.get("cls_proba")) else 0.0,
                "Pred MFD": float(row.get("pred_mfd", 0)) if pd.notna(row.get("pred_mfd")) else 0.0,
                "Z_cls": float(row.get("z_cls", 0)) if pd.notna(row.get("z_cls")) else 0.0,
                "Z_ltr": float(row.get("z_ltr", 0)) if pd.notna(row.get("z_ltr")) else 0.0,
                "Score": float(row.get("ensemble_score", 0)) if pd.notna(row.get("ensemble_score")) else 0.0,
                "Elite Pool Size": int(row.get("elite_pool_size", 0)) if pd.notna(row.get("elite_pool_size")) else 0,
                "Sentiment Score": round(sentiment_normalized, 4),
                "Composite Score": round(prob, 4),  # pipeline already ranked by ensemble
                "Signal": row.get("signal", "BUY"),
                "Vol Surge 3d": vol_surge_str,
                "Regime Confidence": float(row.get("regime_confidence", 0.5)),
                "Ticker Calibration": float(row.get("ticker_calibration", 1.0)),
                "Sentiment Polarity": round(sent_score, 3),
                "Total Mentions": int(row.get("sentiment_mentions", 0)),
                "Market Cap": _fmt_mcap(row.get("market_cap")),
                "Sector": row.get("sector", "N/A"),
                "Close Price": row.get("close_price"),
                "SHAP Explanation": shap_str if pd.notna(shap_str) else "",
                "_explanation_str": shap_str if pd.notna(shap_str) else "",
            })
        return results

    # ── Check for existing pipeline picks ────────────────────────────
    # Priority: session state cache > CSV file.
    # This ensures picks survive page navigation even when pool < 75
    # (not saved to CSV but still in session state).
    if "today_picks_cache" in st.session_state and st.session_state["today_picks_cache"] is not None:
        today_picks = st.session_state["today_picks_cache"]
        source = "session cache"
    else:
        today_picks = _load_todays_picks()
        source = "CSV"
        if today_picks is not None:
            st.session_state["today_picks_cache"] = today_picks

    if today_picks is not None:
        st.success(
            f"Loaded **{len(today_picks)} picks** from {source}. "
            "Click Regenerate to re-run."
        )

    col_cfg1, col_cfg2 = st.columns([1, 2])
    with col_cfg1:
        top_x = st.number_input(
            "Show top X results",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
        )
    with col_cfg2:
        if today_picks is not None:
            st.caption("Pipeline results loaded. Click Regenerate to re-run.")

    # ── Display picks (from CSV or after regeneration) ───────────────
    regenerate = st.button("🔄 Regenerate Recommendations", type="primary")

    results: list[dict] = []
    from_pipeline = False

    if regenerate:
        from datetime import date as _date
        from stock_predictor.models.automl_model import MIN_ELITE_POOL

        csv_path = Path(DEFAULT_CSV_PATH)
        today_str = _date.today().isoformat()

        with st.spinner(
            "Re-running daily picks pipeline on NASDAQ universe..."
        ):
            try:
                # Always generate picks for display (save_to_csv=False
                # bypasses the pool-size gate and CSV caching).
                new_picks = run_daily_picks(
                    top_k=max(top_x, 10),
                    save_to_csv=False,
                )
                if new_picks is not None and not new_picks.empty:
                    today_picks = new_picks
                    from_pipeline = True

                    # Always cache in session state for page navigation
                    st.session_state["today_picks_cache"] = new_picks

                    pool_size = int(new_picks["elite_pool_size"].iloc[0]) if "elite_pool_size" in new_picks.columns else 0

                    # Only update the scheduler CSV when pool >= 75
                    if pool_size >= MIN_ELITE_POOL:
                        # Remove today's old rows, write new ones
                        if csv_path.exists():
                            try:
                                existing = pd.read_csv(csv_path)
                                cleaned = existing[existing["date"] != today_str]
                                cleaned.to_csv(csv_path, index=False)
                            except Exception:
                                pass
                        new_picks.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
                        st.success(f"Regenerated {len(today_picks)} picks! (pool={pool_size}, saved to CSV)")
                    else:
                        st.success(
                            f"Regenerated {len(today_picks)} picks for display. "
                            f"Pool size {pool_size} < {MIN_ELITE_POOL} — not saved to scheduler CSV."
                        )
                else:
                    st.warning("Pipeline produced no picks.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")

    if today_picks is not None and not results:
        results = _csv_rows_to_results(today_picks, top_x)
    elif not results:
        st.info(
            "No picks available for today. Click **Regenerate Recommendations** "
            "to run the pipeline, or schedule it via the Daily Picks Pipeline page."
        )
        st.stop()

    if not results:
        st.warning("No results available.")
        st.stop()

    top_results = results[:top_x]

    # Load social buzz data for 🔥 indicator
    hot_tickers = get_hot_tickers()
    buzz_data = get_social_buzz_data()

    st.subheader(f"Top {len(top_results)} Recommendations")
    if hot_tickers:
        st.caption(
            f"🔥 = trending on social media ({len(hot_tickers)} hot stocks)"
        )

    display_top = []
    for r in top_results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        ticker = r["Ticker"]
        if ticker in hot_tickers:
            row["Ticker"] = f"🔥 {ticker}"
            buzz = buzz_data.get(ticker, {})
            row["Social Buzz"] = buzz.get("sentiment_label", "Hot")
        else:
            row["Social Buzz"] = "—"
        display_top.append(row)

    df = pd.DataFrame(display_top)
    # Reorder columns: Ticker, Social Buzz, then the rest
    cols = df.columns.tolist()
    if "Social Buzz" in cols:
        cols.remove("Social Buzz")
        ticker_idx = cols.index("Ticker") + 1
        cols.insert(ticker_idx, "Social Buzz")
        df = df[cols]

    # Show elite pool size banner if available
    pool_sizes = [r.get("Elite Pool Size", 0) for r in top_results]
    avg_pool = max(pool_sizes) if pool_sizes else 0
    if avg_pool > 0:
        pool_color = "green" if avg_pool >= 75 else "orange" if avg_pool >= 25 else "red"
        st.markdown(
            f"**Elite Pool Size: :{pool_color}[{avg_pool}]** "
            f"({'Strong signal — pool ≥ 75' if avg_pool >= 75 else 'Moderate signal' if avg_pool >= 25 else 'Weak signal — consider sitting out'})"
        )

    st.dataframe(
        df.style.format({
            "Model P(≥20%)": "{:.1%}",
            "Classifier P": "{:.1%}",
            "Pred MFD": "{:.1%}",
            "Z_cls": "{:+.2f}",
            "Z_ltr": "{:+.2f}",
            "Score": "{:.3f}",
            "Sentiment Score": "{:.1%}",
            "Composite Score": "{:.1%}",
            "Sentiment Polarity": "{:+.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Detailed view for each top pick
    st.subheader("Prediction Details")
    for r in top_results:
        ticker_name = r["Ticker"]
        model_p = r["Model P(≥20%)"]
        signal = r.get("Signal", "HOLD")
        shap_str = r.get("_explanation_str", "") or r.get("SHAP Explanation", "")

        cls_p = r.get("Classifier P", 0)
        pred_mfd = r.get("Pred MFD", 0)
        z_cls = r.get("Z_cls", 0)
        z_ltr = r.get("Z_ltr", 0)
        score = r.get("Score", 0)
        pool = r.get("Elite Pool Size", 0)

        mcap_str = r.get("Market Cap", "N/A")
        with st.expander(
            f"**{ticker_name}** — P: {cls_p:.1%} | "
            f"MFD: {pred_mfd:.1%} | "
            f"Score: {score:.3f} | "
            f"MCap: {mcap_str} | "
            f"Sector: {r.get('Sector', 'N/A')}"
        ):
            # 4-stage pipeline scores
            st.markdown("**4-Stage Pipeline Scores**")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric(
                "Classifier P", f"{cls_p:.1%}",
                help="Stage 1: P(MFD ≥ 20%). Gate: P ≥ 0.50",
            )
            col2.metric(
                "Pred MFD", f"{pred_mfd:.1%}",
                help="Stage 2: Predicted max forward drawdown. Gate: ≥ 25%",
            )
            col3.metric(
                "Z_cls", f"{z_cls:+.2f}",
                help="Z-score of classifier probability within elite pool",
            )
            col4.metric(
                "Z_ltr", f"{z_ltr:+.2f}",
                help="Z-score of LTR ranking within elite pool",
            )
            col5.metric(
                "Final Score", f"{score:.3f}",
                help="max(Z_cls, 0) × max(Z_ltr, 0) — both must be above avg",
            )

            # Pool size, market cap, and signal strength
            col6, col7, col8, col9, col_mc = st.columns(5)
            col6.metric(
                "Elite Pool", pool,
                help="Stocks passing both gates today. ≥75 = strong signal",
            )
            col7.metric("Signal", signal)
            vol_surge = r.get("Vol Surge 3d", "N/A")
            col8.metric(
                "Volume Surge (3d)", vol_surge,
                help="3-day volume relative to 20-day average",
            )
            if r.get("Close Price"):
                col9.metric("Last Close", f"${r['Close Price']:.2f}")
            col_mc.metric(
                "Market Cap", mcap_str,
                help="Current market cap (informational, not used for filtering)",
            )

            # Additional context
            regime_conf = r.get("Regime Confidence", "N/A")
            ticker_cal = r.get("Ticker Calibration", 1.0)
            if regime_conf != "N/A" or (ticker_cal != 1.0 and ticker_cal is not None):
                col10, col11, col12, col13 = st.columns(4)
                if regime_conf != "N/A":
                    col10.metric(
                        "Regime Confidence",
                        f"{regime_conf:.0%}" if isinstance(regime_conf, (int, float)) else str(regime_conf),
                        help="Market regime model's predicted daily hit rate",
                    )
                if ticker_cal != 1.0 and ticker_cal is not None:
                    col11.metric(
                        "Ticker Calibration", f"{ticker_cal:.2f}",
                        help="Calibration factor (<1.0 = historically underperforms)",
                    )
                col12.metric("Sentiment", f"{r['Sentiment Polarity']:+.3f}")
                col13.metric("Mentions", r["Total Mentions"])

            # SHAP-based prediction explanation
            if shap_str:
                st.markdown("---")
                st.markdown("**Why This Prediction** (top SHAP features)")
                for part in str(shap_str).split("; "):
                    if "=" in part:
                        feat_name, val_str = part.split("=", 1)
                        try:
                            val = float(val_str)
                            color = ":green[▲]" if val > 0 else ":red[▼]"
                            st.markdown(f"{color} **{feat_name}**: {val:+.4f}")
                        except ValueError:
                            st.markdown(f"- {part}")
                    else:
                        st.markdown(f"- {part}")

            # Forward guidance from earnings call transcript
            st.markdown("---")
            guidance_key = f"guidance_{ticker_name}"
            if st.button(
                f"📞 Fetch Forward Guidance ({ticker_name})",
                key=f"btn_guidance_{ticker_name}",
            ):
                if not api_key:
                    st.error("OpenAI API key required for transcript analysis.")
                else:
                    with st.spinner(f"Fetching earnings call transcript for {ticker_name}..."):
                        try:
                            from stock_predictor.agent.transcript_agent import (
                                analyze_ticker_forward_guidance,
                            )
                            analysis = analyze_ticker_forward_guidance(
                                ticker_name, api_key=api_key,
                                model=model_choice,
                            )
                            st.session_state[guidance_key] = analysis
                        except Exception as e:
                            import traceback
                            st.error(f"Error fetching transcript: {e}")
                            st.code(traceback.format_exc())

            if guidance_key in st.session_state:
                analysis = st.session_state[guidance_key]
                if analysis.get("found"):
                    st.markdown(f"**📞 Forward Guidance** (transcript from {analysis.get('date', 'N/A')})")
                    st.markdown(analysis.get("forward_guidance", "No guidance extracted."))
                    if analysis.get("source_url"):
                        st.caption(f"Source: [{analysis['source_url']}]({analysis['source_url']})")
                else:
                    st.warning(
                        f"No earnings call transcript found for {ticker_name}. "
                        f"{analysis.get('error', '')}"
                    )

    # Highlight BUY signals
    buy_picks = [r for r in top_results if r.get("Signal") == "BUY"]
    if buy_picks:
        st.success(f"**{len(buy_picks)} BUY signals** in top {len(top_results)}:")
        for r in buy_picks:
            model_p = r["Model P(≥20%)"]
            sent_p = r["Sentiment Polarity"]
            mentions = r["Total Mentions"]
            st.write(
                f"**{r['Ticker']}** — "
                f"Model: {model_p:.1%}, "
                f"Sentiment: {sent_p:+.3f} "
                f"({mentions} mentions)"
            )


# ---------------------------------------------------------------------------
# Page: Stock Chart
# ---------------------------------------------------------------------------
elif page == "Stock Chart":
    st.title("Stock Chart Dashboard")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime, timedelta
    import numpy as _np

    col_ticker, col_period = st.columns([1, 2])
    with col_ticker:
        chart_ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    with col_period:
        period_options = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "Custom"]
        selected_period = st.radio("Time Period", period_options, horizontal=True, index=3)

    custom_start = custom_end = None
    if selected_period == "Custom":
        col_s, col_e = st.columns(2)
        with col_s:
            custom_start = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col_e:
            custom_end = st.date_input("End Date", value=datetime.now())

    if st.button("Load Chart", type="primary") or chart_ticker:
        # Map period to yfinance period string
        period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}

        with st.spinner(f"Loading {chart_ticker} data..."):
            if selected_period == "Custom" and custom_start and custom_end:
                import yfinance as yf
                tk = yf.Ticker(chart_ticker)
                df = tk.history(start=str(custom_start), end=str(custom_end))
            else:
                yf_period = period_map.get(selected_period, "1y")
                df = get_stock_data(chart_ticker, period=yf_period)

        if df.empty:
            st.error(f"No data found for {chart_ticker}")
        else:
            # Compute moving averages
            df["SMA_20"] = df["Close"].rolling(20).mean()
            df["SMA_50"] = df["Close"].rolling(50).mean()
            df["SMA_200"] = df["Close"].rolling(200).mean()

            # Compute RSI
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["RSI_14"] = 100 - (100 / (1 + rs))

            # Use index as dates
            dates = df.index

            # Create subplots: price, volume, RSI
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f"{chart_ticker} Price", "Volume", "RSI (14)"),
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=dates,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Price",
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                ),
                row=1, col=1,
            )

            # Moving averages
            fig.add_trace(
                go.Scatter(x=dates, y=df["SMA_20"], name="SMA 20",
                           line=dict(color="#2196F3", width=1)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=dates, y=df["SMA_50"], name="SMA 50",
                           line=dict(color="#FF9800", width=1)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=dates, y=df["SMA_200"], name="SMA 200",
                           line=dict(color="#9C27B0", width=1.5)),
                row=1, col=1,
            )

            # Volume bars (colored by direction)
            colors = [
                "#26a69a" if c >= o else "#ef5350"
                for c, o in zip(df["Close"], df["Open"])
            ]
            fig.add_trace(
                go.Bar(x=dates, y=df["Volume"], name="Volume",
                       marker_color=colors, opacity=0.7),
                row=2, col=1,
            )

            # RSI
            fig.add_trace(
                go.Scatter(x=dates, y=df["RSI_14"], name="RSI 14",
                           line=dict(color="#7C4DFF", width=1.5)),
                row=3, col=1,
            )
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray",
                          opacity=0.3, row=3, col=1)

            fig.update_layout(
                height=800,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=50, r=20, t=60, b=20),
            )

            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            latest = df.iloc[-1]
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Close", f"${latest['Close']:.2f}",
                          delta=f"{((latest['Close'] / df.iloc[-2]['Close']) - 1) * 100:.2f}%" if len(df) > 1 else None)
            with col2:
                st.metric("SMA 20", f"${latest['SMA_20']:.2f}" if pd.notna(latest.get('SMA_20')) else "N/A")
            with col3:
                st.metric("SMA 50", f"${latest['SMA_50']:.2f}" if pd.notna(latest.get('SMA_50')) else "N/A")
            with col4:
                st.metric("SMA 200", f"${latest['SMA_200']:.2f}" if pd.notna(latest.get('SMA_200')) else "N/A")
            with col5:
                rsi_val = latest.get("RSI_14")
                rsi_label = ""
                if pd.notna(rsi_val):
                    if rsi_val >= 70:
                        rsi_label = " (Overbought)"
                    elif rsi_val <= 30:
                        rsi_label = " (Oversold)"
                st.metric("RSI 14", f"{rsi_val:.1f}{rsi_label}" if pd.notna(rsi_val) else "N/A")


# ---------------------------------------------------------------------------
# Page: AI Stock Advisor (Chat)
# ---------------------------------------------------------------------------
elif page == "AI Stock Advisor":
    st.title("AI Stock Investment Advisor")
    st.markdown(
        "Ask the AI agent for stock recommendations. It uses YFinance data, "
        "social media sentiment, a trained prediction model, and **earnings call "
        "transcript analysis** (forward guidance) to provide analysis."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Handle quick prompt (injected by buttons below)
    if "_quick_prompt" in st.session_state:
        _pending_prompt = st.session_state.pop("_quick_prompt")
    else:
        _pending_prompt = None

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Ask about stocks (e.g., 'Which NASDAQ stocks have the best 3-month outlook?')"
    )

    # Quick prompt takes priority when no manual input
    if not user_input and _pending_prompt:
        user_input = _pending_prompt

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
                if result.get("probability_gain") is not None:
                    prob = result["probability_gain"]
                    signal = result.get("signal", "HOLD")
                    vol_surge = result.get("volume_surge_3d")
                    regime = result.get("regime_confidence", 0.5)
                    cal = result.get("ticker_calibration", 1.0)
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric(
                        "P(≥20% gain in 3M)",
                        f"{prob * 100:.1f}%",
                    )
                    col_b.metric("Signal", signal)
                    col_c.metric(
                        "Volume Surge (3d)",
                        f"{vol_surge:.2f}x" if vol_surge is not None else "N/A",
                        help="3-day volume relative to 20-day average",
                    )
                    col_d, col_e = st.columns(2)
                    col_d.metric(
                        "Regime Confidence",
                        f"{regime:.0%}",
                        help="Market regime model's predicted daily hit rate",
                    )
                    if cal < 1.0:
                        col_e.metric(
                            "Ticker Calibration",
                            f"{cal:.2f}",
                            help="Historical accuracy factor (<1.0 = underperforms in top picks)",
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
        time_budget = st.slider("AutoML time budget (seconds)", 30, 600, 300)

    include_sentiment = st.checkbox("Include social media sentiment features", value=True)

    tickers = NASDAQ_TOP_TICKERS[:num_tickers]
    st.write(f"Training tickers: {', '.join(tickers)}")

    # Walk-forward options
    st.markdown("---")
    st.subheader("Training Mode")
    training_mode = st.radio(
        "Select training mode:",
        ["Standard (single split)", "Walk-Forward Cross-Validation"],
        help="Walk-forward trains on expanding windows for more robust evaluation.",
    )

    if training_mode == "Walk-Forward Cross-Validation":
        wf_col1, wf_col2 = st.columns(2)
        with wf_col1:
            wf_folds = st.slider("Number of folds", 3, 10, 5)
        with wf_col2:
            wf_min_years = st.slider("Minimum training years", 2, 5, 3)

    train_col, data_col = st.columns(2)

    with train_col:
        if st.button("Start Training", type="primary"):
            predictor = StockReturnPredictor()
            progress_bar = st.progress(0)
            status = st.empty()

            status.info("Building training dataset (this may take several minutes)...")
            progress_bar.progress(10)

            try:
                if training_mode == "Walk-Forward Cross-Validation":
                    import os
                    csv_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "training_data_10y_full.csv",
                    )
                    if os.path.exists(csv_path):
                        import pandas as _pd
                        status.info(
                            f"Loading full training dataset and running "
                            f"{wf_folds}-fold walk-forward CV..."
                        )
                        progress_bar.progress(20)
                        _df = _pd.read_csv(csv_path)
                        metrics = predictor.train_walk_forward(
                            df=_df,
                            time_budget=time_budget,
                            n_folds=wf_folds,
                            min_train_years=wf_min_years,
                        )
                    else:
                        status.warning(
                            "Full training CSV not found. "
                            "Falling back to standard training."
                        )
                        metrics = predictor.train(
                            tickers=tickers,
                            time_budget=time_budget,
                            include_sentiment=include_sentiment,
                        )
                else:
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

        # --- Walk-Forward Results ---
        if "folds" in metrics:
            st.subheader("Walk-Forward Cross-Validation Results")
            st.info(
                f"**{metrics.get('n_folds', 0)} folds** | "
                f"Aggregate Top-10: **{metrics.get('aggregate_top10_hits', 0)}"
                f"/{metrics.get('aggregate_top10_total', 0)}** "
                f"({metrics.get('aggregate_top10_hit_rate', 0)*100:.1f}%) | "
                f"Mean AUC: {metrics.get('mean_auc_test', 0):.4f} ± "
                f"{metrics.get('std_auc_test', 0):.4f} | "
                f"Mean AP: {metrics.get('mean_ap_test', 0):.4f} ± "
                f"{metrics.get('std_ap_test', 0):.4f}"
            )

            # Per-fold table
            fold_data = []
            for f in metrics["folds"]:
                fold_data.append({
                    "Fold": f["fold"],
                    "Test Period": f["test_period"],
                    "Train Rows": f["train_rows"],
                    "Test Rows": f["test_rows"],
                    "AUC (Train)": f["auc_train"],
                    "AUC (Test)": f["auc_test"],
                    "AUC Gap": f["auc_gap"],
                    "AP (Train)": f["ap_train"],
                    "AP (Test)": f["ap_test"],
                    "AP Gap": f["ap_gap"],
                    "Top-10 Hits": f"{f['top10_hits']}/10",
                    "Hit Rate": f"{f['top10_hit_rate']*100:.0f}%",
                })
            import pandas as _pd_wf
            st.dataframe(
                _pd_wf.DataFrame(fold_data),
                use_container_width=True,
                hide_index=True,
            )

            # Per-fold top-10 picks
            for f in metrics["folds"]:
                if f.get("top10_picks"):
                    with st.expander(
                        f"Fold {f['fold']} Top-10 Picks ({f['test_period']})"
                    ):
                        picks_df = _pd_wf.DataFrame(f["top10_picks"])
                        st.dataframe(picks_df, use_container_width=True, hide_index=True)
        else:
            st.subheader("Model Evaluation Metrics")

        # --- Architecture summary ---
        ltr = metrics.get("ltr", {})
        if ltr.get("status") == "trained":
            st.info(
                "**Ensemble Architecture:** Classification (FLAML) + "
                "Learning-to-Rank (LambdaMART) — 50/50 weighted. "
                "The LTR model directly optimizes NDCG@10 to rank "
                "stocks within each trading day."
            )

        # --- LTR Ranking Quality (primary) ---
        if ltr.get("status") == "trained":
            st.markdown("#### Ranking Quality (LTR)")
            st.caption(
                "The LTR model ranks stocks within each day so the "
                "top-10 picks have the highest precision."
            )
            l1, l2, l3 = st.columns(3)
            with l1:
                st.metric(
                    "NDCG@10 (Test)",
                    f"{ltr.get('ndcg10_test', 0):.4f}",
                    help="Ranking quality of top-10 on held-out test dates",
                )
            with l2:
                st.metric(
                    "NDCG@10 (Train)",
                    f"{ltr.get('ndcg10_train', 0):.4f}",
                )
            with l3:
                st.metric(
                    "NDCG@10 Gap",
                    f"{ltr.get('ndcg10_gap', 0):.4f}",
                    help="Train − Test gap (lower = less overfitting)",
                )
            l4, l5 = st.columns(2)
            with l4:
                st.metric("LTR Best Iteration", ltr.get("best_iteration", 0))
            with l5:
                st.metric(
                    "Date Groups (Train/Test)",
                    f"{ltr.get('train_groups', 0)} / {ltr.get('test_groups', 0)}",
                )

        # --- Top-N Precision (primary metrics) ---
        top_n = metrics.get("top_n", {})
        top_10 = top_n.get("top_10", {})
        top_20 = top_n.get("top_20", {})
        top_50 = top_n.get("top_50", {})

        if top_10:
            st.markdown("#### Top Pick Precision (Test Set)")
            st.caption(
                "How many of the ensemble's highest-confidence picks actually "
                "achieved ≥20% peak return within 3 months."
            )
            p1, p2, p3 = st.columns(3)
            with p1:
                hit_rate_10 = top_10.get("hit_rate", 0)
                hits_10 = top_10.get("hits", 0)
                st.metric(
                    "Top-10 Hit Rate",
                    f"{hits_10}/{top_10.get('total', 10)} ({hit_rate_10:.0%})",
                    help="Fraction of top 10 predictions that hit ≥20% peak return",
                )
            with p2:
                if top_20:
                    hit_rate_20 = top_20.get("hit_rate", 0)
                    hits_20 = top_20.get("hits", 0)
                    st.metric(
                        "Top-20 Hit Rate",
                        f"{hits_20}/{top_20.get('total', 20)} ({hit_rate_20:.0%})",
                    )
            with p3:
                if top_50:
                    hit_rate_50 = top_50.get("hit_rate", 0)
                    hits_50 = top_50.get("hits", 0)
                    st.metric(
                        "Top-50 Hit Rate",
                        f"{hits_50}/{top_50.get('total', 50)} ({hit_rate_50:.0%})",
                    )

            r1, r2, r3 = st.columns(3)
            with r1:
                avg_ret_10 = top_10.get("avg_peak_return", 0)
                st.metric(
                    "Top-10 Avg Peak Return",
                    f"{avg_ret_10:.1%}",
                    help="Average peak return across the top 10 picks",
                )
            with r2:
                if top_20:
                    st.metric(
                        "Top-20 Avg Peak Return",
                        f"{top_20.get('avg_peak_return', 0):.1%}",
                    )
            with r3:
                if top_50:
                    st.metric(
                        "Top-50 Avg Peak Return",
                        f"{top_50.get('avg_peak_return', 0):.1%}",
                    )

            # Top 10 individual picks table
            picks = top_10.get("picks", [])
            if picks:
                st.markdown("**Top 10 Picks Detail**")
                picks_df = pd.DataFrame(picks)
                picks_df = picks_df.rename(columns={
                    "rank": "Rank",
                    "ticker": "Ticker",
                    "probability": "Ensemble Score",
                    "actual_return": "Peak Return",
                    "hit": "Hit (≥20%)",
                })
                picks_df["Peak Return"] = picks_df["Peak Return"].apply(
                    lambda x: f"{x:.1%}" if x is not None else "N/A"
                )
                picks_df["Ensemble Score"] = picks_df["Ensemble Score"].apply(
                    lambda x: f"{x:.1%}"
                )
                picks_df["Hit (≥20%)"] = picks_df["Hit (≥20%)"].apply(
                    lambda x: "✓" if x else "✗"
                )
                st.dataframe(picks_df, use_container_width=True, hide_index=True)

        # --- Market Regime & Ticker Calibration ---
        regime = metrics.get("regime", {})
        ticker_cal = metrics.get("ticker_calibration", {})

        if regime.get("status") == "trained" or ticker_cal.get("status") == "trained":
            st.markdown("#### Precision Enhancement Models")

        if regime.get("status") == "trained":
            with st.expander("Market Regime Detection", expanded=False):
                st.caption(
                    "Predicts daily hit rate from macro conditions "
                    "(VIX, market returns, yield curve). Used to flag "
                    "low-confidence days."
                )
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("R² (Test)", f"{regime.get('r2_test', 0):.4f}")
                with rc2:
                    st.metric("R² (Train)", f"{regime.get('r2_train', 0):.4f}")
                with rc3:
                    st.metric(
                        "R² Gap (Overfitting)",
                        f"{regime.get('r2_gap', 0):.4f}",
                        help="Train − Test gap (lower = less overfitting)",
                    )
                st.caption(
                    f"Training days: {regime.get('n_train_days', 0)} | "
                    f"Test days: {regime.get('n_test_days', 0)} | "
                    f"Features: {', '.join(regime.get('features', []))}"
                )

        if ticker_cal.get("status") == "trained":
            with st.expander("Per-Ticker Calibration", expanded=False):
                st.caption(
                    "Tracks each ticker's historical hit rate in top-10 "
                    "picks and penalizes repeat false positives (hit rate <30%)."
                )
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.metric(
                        "Tickers Tracked",
                        ticker_cal.get("tickers_tracked", 0),
                    )
                with tc2:
                    st.metric(
                        "Tickers Penalized",
                        ticker_cal.get("tickers_penalized", 0),
                        help="Tickers with <30% hit rate in top-10 appearances",
                    )
                penalized = ticker_cal.get("penalized_tickers", {})
                if penalized:
                    st.markdown("**Penalized Tickers (lowest calibration factors):**")
                    pen_rows = [
                        {"Ticker": t, "Calibration Factor": f}
                        for t, f in penalized.items()
                    ]
                    st.dataframe(
                        pd.DataFrame(pen_rows),
                        use_container_width=True, hide_index=True,
                    )

        # --- Classification metrics (collapsible) ---
        with st.expander("Classification Model Metrics (FLAML)"):
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    "AUC-ROC (Test)",
                    f"{metrics.get('auc_roc', 0):.4f}",
                    help="Area Under ROC Curve on held-out test set",
                )
            with m2:
                st.metric(
                    "Avg Precision (Test)",
                    f"{metrics.get('avg_precision', 0):.4f}",
                    help="Area under the Precision-Recall curve",
                )
            with m3:
                st.metric("Best Model", metrics.get("best_estimator", "N/A"))
            with m4:
                st.metric("Training Samples", metrics.get("training_samples", 0))

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
            st.markdown("**Overfitting Check:**")
            m5, m6 = st.columns(2)
            with m5:
                st.metric(
                    "AUC (Train)",
                    f"{metrics.get('auc_train', 0):.4f}",
                    help="Compare with Test AUC to detect overfitting",
                )
            with m6:
                st.metric(
                    "AP (Train)",
                    f"{metrics.get('ap_train', 0):.4f}",
                    help="Compare with Test AP",
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
        grouped = predictor_for_importance.get_grouped_feature_importance(top_n=25)
        if grouped:
            st.markdown("---")
            st.subheader("Feature Importances (Grouped by Correlation)")
            st.caption(
                "Classification model feature importances. Correlated features "
                "(Spearman |r| > 0.70) are summed into concept-level groups "
                "so importance isn't diluted across redundant features. "
                "These features feed both the classification and LTR models."
            )
            import plotly.express as px

            labels = []
            values = []
            colors = []
            for name, imp, members in grouped:
                if len(members) > 1:
                    label = f"⬛ {name} ({len(members)} features)"
                else:
                    label = name
                labels.append(label)
                values.append(imp)
                colors.append("group" if len(members) > 1 else "single")

            imp_df = pd.DataFrame({
                "Concept": labels,
                "Importance": values,
                "Type": colors,
            })
            imp_df = imp_df.sort_values("Importance", ascending=True)
            fig = px.bar(
                imp_df, x="Importance", y="Concept",
                orientation="h",
                title="Grouped Feature Importances (Higher = More Predictive)",
                color="Type",
                color_discrete_map={"group": "#636EFA", "single": "#00CC96"},
            )
            fig.update_layout(
                height=max(400, len(grouped) * 28),
                yaxis_title="",
                xaxis_title="Importance Score (sum of split-based importances)",
                showlegend=True,
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Group Details"):
                for name, imp, members in grouped:
                    if len(members) > 1:
                        st.markdown(f"**{name}** (importance: {imp})")
                        st.markdown("  " + ", ".join(f"`{m}`" for m in members))
                    else:
                        st.markdown(f"`{name}` (importance: {imp})")

    # ---- Gain Chart ----
    if "training_metrics" in st.session_state:
        gain_data = st.session_state["training_metrics"].get("gain_chart")
        if gain_data and gain_data.get("percentages"):
            import plotly.graph_objects as go  # noqa: F811

            st.markdown("---")
            st.subheader("Cumulative Gain Chart")
            st.markdown(
                "Shows the percentage of actual 20%+ gainers captured when "
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
            ltr_status = "with LTR" if predictor.ltr_model is not None else "classification only"
            st.info(f"Model type: ensemble ({ltr_status}). Train a new model above to see detailed metrics.")
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

    top_k = st.selectbox("Top picks to highlight", [5, 10, 20], index=0)

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
                if result.get("probability_gain") is not None:
                    results.append(result)

            results.sort(key=lambda x: x["probability_gain"], reverse=True)

            # --- Top-K picks ---
            st.subheader(f"Top {top_k} Picks")
            top_picks = results[:top_k]
            top_rows = []
            for rank, r in enumerate(top_picks, 1):
                vol_surge = r.get("volume_surge_3d")
                regime = r.get("regime_confidence", 0.5)
                cal = r.get("ticker_calibration", 1.0)
                top_rows.append({
                    "Rank": rank,
                    "Ticker": r["ticker"],
                    "P(≥20% gain)": r["probability_gain"],
                    "Probability %": r["probability_pct"],
                    "Signal": r.get("signal", "HOLD"),
                    "Vol Surge 3d": f"{vol_surge:.2f}x" if vol_surge is not None else "N/A",
                    "Regime Conf.": f"{regime:.0%}",
                    "Calibration": f"{cal:.2f}" if cal < 1.0 else "—",
                })
            st.dataframe(pd.DataFrame(top_rows), use_container_width=True)

            # Regime confidence indicator
            if top_picks:
                avg_regime = sum(r.get("regime_confidence", 0.5) for r in top_picks) / len(top_picks)
                if avg_regime >= 0.6:
                    st.success(f"Market Regime: Favorable ({avg_regime:.0%} confidence)")
                elif avg_regime >= 0.4:
                    st.info(f"Market Regime: Neutral ({avg_regime:.0%} confidence)")
                else:
                    st.warning(f"Market Regime: Unfavorable ({avg_regime:.0%} confidence) — picks may underperform")

            # --- Full results ---
            with st.expander(f"All Results ({len(results)} stocks)"):
                batch_rows = []
                for r in results:
                    vol_surge = r.get("volume_surge_3d")
                    cal = r.get("ticker_calibration", 1.0)
                    batch_rows.append({
                        "Ticker": r["ticker"],
                        "P(≥20% gain)": r["probability_gain"],
                        "Probability %": r["probability_pct"],
                        "Signal": r.get("signal", "HOLD"),
                        "Vol Surge 3d": f"{vol_surge:.2f}x" if vol_surge is not None else "N/A",
                        "Calibration": f"{cal:.2f}" if cal < 1.0 else "—",
                    })
                st.dataframe(pd.DataFrame(batch_rows), use_container_width=True)

            buy_signals = [r for r in results if r.get("signal") == "BUY"]
            if buy_signals:
                st.success(
                    f"Found {len(buy_signals)} stocks with BUY signal (≥50% probability of 20%+ gain)!"
                )
            else:
                st.info("No stocks with BUY signal found in this batch.")

        except FileNotFoundError:
            st.error("Model not trained yet. Go to 'Model Training' first.")

# ---------------------------------------------------------------------------
# Page: Social Media Listener
# ---------------------------------------------------------------------------
elif page == "Social Media Listener":
    st.title("🔥 Social Media Listener")
    st.markdown(
        "Top 20 hottest stocks — filtered to **NASDAQ-listed** stocks.  "
        "Data sourced from Reddit, Yahoo Finance (trending, most active, day movers), "
        "Finviz news headlines, and GDELT global news."
    )

    with st.expander("Eligible Ticker Universe"):
        cache_info = get_ticker_cache_info()
        col_u1, col_u2 = st.columns([3, 1])
        with col_u1:
            eligible = get_eligible_tickers()
            if cache_info["cached"]:
                st.write(
                    f"**{len(eligible)}** NASDAQ tickers "
                    f"(cached {cache_info['age_hours']}h ago)"
                )
            else:
                st.write(f"**{len(eligible)}** NASDAQ tickers")
        with col_u2:
            if st.button("🔄 Refresh Tickers", key="refresh_tickers"):
                with st.spinner("Fetching tickers from Wikipedia + yFinance..."):
                    eligible = get_eligible_tickers(force_refresh=True)
                st.success(f"Refreshed: {len(eligible)} eligible tickers")
                st.rerun()
        st.caption(", ".join(sorted(eligible)[:50]) + f"... ({len(eligible)} total)")

    if st.button("🔄 Refresh Market Buzz Data", type="primary"):
        with st.spinner("Scanning Yahoo Finance, Finviz & GDELT for trending stocks..."):
            hottest = get_social_hottest(top_n=20)

        if hottest:
            st.success(f"Found {len(hottest)} trending stocks on major exchanges")

            rows = []
            for rank, item in enumerate(hottest, 1):
                sent = item["avg_sentiment"]
                if sent > 0.1:
                    sent_icon = "🟢"
                elif sent < -0.1:
                    sent_icon = "🔴"
                else:
                    sent_icon = "⚪"

                rows.append({
                    "Rank": rank,
                    "Ticker": item["ticker"],
                    "Mentions": item["mentions"],
                    "Sentiment": f"{sent_icon} {item['sentiment_label']} ({sent:+.3f})",
                    "Volume": f"{item['total_upvotes']:,}" if item["total_upvotes"] else "—",
                    "Change %": f"{item['change_pct']:+.1f}%" if item["change_pct"] else "—",
                    "Engagement": item["engagement_score"],
                    "Sources": item["sources"],
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiments = [h["avg_sentiment"] for h in hottest]
            labels = [h["ticker"] for h in hottest]
            chart_df = pd.DataFrame({"Ticker": labels, "Sentiment": sentiments})
            st.bar_chart(chart_df.set_index("Ticker"))

            st.caption(f"Last updated: {hottest[0].get('last_updated', 'N/A')}")
        else:
            st.info("No trending tickers found on major exchanges at the moment.")


# ---------------------------------------------------------------------------
# Page: Daily Picks Pipeline
# ---------------------------------------------------------------------------
elif page == "Daily Picks Pipeline":
    st.title("📋 Daily Picks Pipeline (MLOps)")
    st.markdown(
        "Schedule automated daily stock pick generation.  The pipeline "
        "scores all tickers from the training universe and records the top "
        "picks with SHAP explanations, sentiment, and market cap "
        "to a CSV.  Use the ground-truth evaluator to track precision over time."
    )

    # -- Scheduler status banner -------------------------------------------
    st.subheader("⏰ Scheduler Status")

    sched_cfg = get_schedule_config()
    active = is_scheduled()
    run_log = get_run_log()

    # Status columns: schedule status | last run info
    stat_col1, stat_col2, stat_col3 = st.columns(3)

    with stat_col1:
        if active:
            next_run = get_next_run()
            st.metric(
                "Schedule",
                f"{sched_cfg.get('frequency', 'daily').title()}",
                f"{sched_cfg.get('hour', 6):02d}:{sched_cfg.get('minute', 0):02d} UTC",
            )
        else:
            st.metric("Schedule", "Inactive", "Not configured")

    with stat_col2:
        if run_log:
            last = run_log[-1]
            last_time = last.get("timestamp", "N/A")
            last_status = last.get("status", "unknown")
            st.metric(
                "Last Run",
                last_time[:16].replace("T", " "),
                f"{last_status} — {last.get('picks', 0)} picks",
            )
        else:
            st.metric("Last Run", "Never", "No runs yet")

    with stat_col3:
        if active:
            next_run = get_next_run()
            if next_run:
                st.metric("Next Run", next_run[:16].replace("T", " "), "Scheduled")
            else:
                st.metric("Next Run", "N/A", "")
        else:
            st.metric("Next Run", "—", "Schedule inactive")

    if active:
        email_status = "📧 Email ON" if is_email_configured() else "📧 Email OFF"
        st.success(
            f"Pipeline is scheduled **{sched_cfg.get('frequency', 'daily')}** "
            f"at **{sched_cfg.get('hour', 6):02d}:{sched_cfg.get('minute', 0):02d} UTC** "
            f"({email_status})"
        )
        if st.button("⏹️ Stop Schedule"):
            stop_schedule()
            st.rerun()
    else:
        st.info("No schedule active. Configure one below to auto-generate picks.")

    with st.expander("Configure Schedule", expanded=not active):
        sched_col1, sched_col2 = st.columns(2)
        with sched_col1:
            sched_freq = st.selectbox(
                "Frequency",
                ["daily", "weekly"],
                index=0 if sched_cfg.get("frequency", "daily") == "daily" else 1,
                key="sched_freq",
            )
            sched_hour = st.slider(
                "Hour (UTC)", 0, 23,
                value=sched_cfg.get("hour", 6),
                key="sched_hour",
            )
            sched_minute = st.slider(
                "Minute", 0, 59,
                value=sched_cfg.get("minute", 0),
                key="sched_minute",
            )
        with sched_col2:
            sched_dow = st.selectbox(
                "Day of week (weekly only)",
                ["mon", "tue", "wed", "thu", "fri", "sat", "sun", "mon-fri"],
                index=7,
                key="sched_dow",
            )
        if st.button("💾 Save & Start Schedule", type="primary"):
            result = schedule_pipeline(
                hour=sched_hour,
                minute=sched_minute,
                frequency=sched_freq,
                day_of_week=sched_dow,
            )
            st.success(
                f"Schedule saved! Next run: {result.get('next_run', 'N/A')}"
            )
            st.rerun()

    # -- Email notification settings --------------------------------------
    st.markdown("---")
    st.subheader("📧 Email Notifications")
    st.markdown(
        "Get an email with the daily top-10 picks after each scheduled run. "
        "For Gmail, use an **App Password** — create one at "
        "[myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)."
    )

    smtp_cfg = get_smtp_config()

    with st.expander(
        "Email Settings" + (" ✅" if is_email_configured() else ""),
        expanded=not is_email_configured(),
    ):
        email_col1, email_col2 = st.columns(2)
        with email_col1:
            smtp_server = st.text_input(
                "SMTP Server",
                value=smtp_cfg.get("smtp_server", "smtp.gmail.com"),
                key="smtp_server",
            )
            smtp_port = st.number_input(
                "SMTP Port",
                value=int(smtp_cfg.get("smtp_port", 587)),
                min_value=1, max_value=65535, step=1,
                key="smtp_port",
            )
            sender_email = st.text_input(
                "Sender Email (Gmail address)",
                value=smtp_cfg.get("sender_email", ""),
                key="sender_email",
            )
        with email_col2:
            sender_password = st.text_input(
                "App Password",
                value=smtp_cfg.get("sender_password", ""),
                type="password",
                key="sender_password",
                help="For Gmail, use an App Password (not your regular password).",
            )
            recipient_email = st.text_input(
                "Recipient Email",
                value=smtp_cfg.get("recipient_email", ""),
                key="recipient_email",
                help="Email address to receive daily picks.",
            )
            email_enabled = st.checkbox(
                "Enable email notifications",
                value=smtp_cfg.get("enabled", False),
                key="email_enabled",
            )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("💾 Save Email Settings", type="primary"):
                save_smtp_config({
                    "enabled": email_enabled,
                    "smtp_server": smtp_server,
                    "smtp_port": int(smtp_port),
                    "sender_email": sender_email,
                    "sender_password": sender_password,
                    "recipient_email": recipient_email,
                })
                st.success("Email settings saved!")
                st.rerun()
        with btn_col2:
            if st.button("📨 Send Test Email"):
                if not sender_email or not sender_password or not recipient_email:
                    st.error("Fill in all email fields first.")
                else:
                    # Save first so the test uses the latest settings
                    save_smtp_config({
                        "enabled": email_enabled,
                        "smtp_server": smtp_server,
                        "smtp_port": int(smtp_port),
                        "sender_email": sender_email,
                        "sender_password": sender_password,
                        "recipient_email": recipient_email,
                    })
                    with st.spinner("Sending test email..."):
                        ok, msg = send_test_email()
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)

    # -- Run log -----------------------------------------------------------
    if run_log:
        with st.expander(f"Scheduler Run History ({len(run_log)} runs)"):
            log_df = pd.DataFrame(run_log)
            st.dataframe(log_df, use_container_width=True)

    # -- Ground truth evaluation ------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Evaluate Ground Truth")
    st.markdown(
        "Check all historical picks for ≥20% upside from the recorded "
        "closing price to date.  Updates existing records only if the "
        "new upside is higher."
    )
    if st.button("🔍 Evaluate Ground Truth"):
        with st.spinner("Checking price history for all recorded picks..."):
            try:
                df_gt = evaluate_ground_truth()
                if df_gt.empty:
                    st.info("No picks recorded yet.")
                else:
                    evaluated = df_gt[df_gt["hit_20pct"].notna()]
                    if evaluated.empty:
                        st.info("No picks old enough to evaluate yet.")
                    else:
                        hits = int(evaluated["hit_20pct"].sum())
                        total = len(evaluated)
                        prec = hits / total if total > 0 else 0
                        st.metric("Overall Precision", f"{prec:.1%}", f"{hits}/{total} hits")
                        st.dataframe(evaluated[["date", "ticker", "probability", "close_price",
                                                 "max_upside_pct", "hit_20pct"]],
                                     use_container_width=True)
            except Exception as e:
                st.error(f"Evaluation error: {e}")

    # Show precision over time if data exists
    st.markdown("---")
    st.subheader("Precision Over Time")
    prec_df = get_precision_over_time()
    if not prec_df.empty:
        st.line_chart(prec_df.set_index("date")["precision"])

        col_a, col_b, col_c = st.columns(3)
        overall_prec = prec_df["hits"].sum() / prec_df["total_picks"].sum() if prec_df["total_picks"].sum() > 0 else 0
        col_a.metric("Overall Precision", f"{overall_prec:.1%}")
        col_b.metric("Total Picks", int(prec_df["total_picks"].sum()))
        col_c.metric("Days Tracked", len(prec_df))
    else:
        st.info("No evaluated picks yet. Run the pipeline and evaluate ground truth to see precision over time.")

    # Show raw CSV
    if DEFAULT_CSV_PATH.exists():
        with st.expander("Raw CSV Data"):
            raw = pd.read_csv(DEFAULT_CSV_PATH)
            st.dataframe(raw, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=raw.to_csv(index=False),
                file_name="daily_picks.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Page: Daily Picks History
# ---------------------------------------------------------------------------
elif page == "Daily Picks History":
    st.title("📊 Daily Picks History")
    st.markdown(
        "Browse historical daily stock picks with all supporting information "
        "including model probability, SHAP explanations, sentiment, and "
        "ground-truth performance."
    )

    if not DEFAULT_CSV_PATH.exists():
        st.info("No picks recorded yet. Go to 'Daily Picks Pipeline' to start recording picks.")
    else:
        df_all = pd.read_csv(DEFAULT_CSV_PATH)
        if df_all.empty:
            st.info("No picks recorded yet.")
        else:
            available_dates = sorted(df_all["date"].unique(), reverse=True)

            # Date filter
            selected_date = st.selectbox("Select Date", available_dates, index=0)

            df_day = df_all[df_all["date"] == selected_date].copy()

            if df_day.empty:
                st.warning(f"No picks for {selected_date}")
            else:
                st.subheader(f"Top {len(df_day)} Picks — {selected_date}")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Picks", len(df_day))
                avg_prob = df_day["probability"].mean()
                col2.metric("Avg Probability", f"{avg_prob:.1%}")

                if "hit_20pct" in df_day.columns and df_day["hit_20pct"].notna().any():
                    hits = int(df_day["hit_20pct"].sum())
                    total_eval = int(df_day["hit_20pct"].notna().sum())
                    prec = hits / total_eval if total_eval > 0 else 0
                    col3.metric("Precision", f"{prec:.0%}", f"{hits}/{total_eval}")
                else:
                    col3.metric("Precision", "Pending")

                avg_sent = df_day["sentiment_score"].mean() if "sentiment_score" in df_day.columns else 0
                sent_label = "Bullish" if avg_sent > 0.05 else ("Bearish" if avg_sent < -0.05 else "Neutral")
                col4.metric("Avg Sentiment", f"{avg_sent:+.3f} ({sent_label})")

                # Picks table
                display_cols = ["rank", "ticker", "probability", "signal", "close_price",
                                "volume_surge_3d", "sentiment_score", "sentiment_mentions",
                                "regime_confidence", "ticker_calibration"]
                if "max_upside_pct" in df_day.columns:
                    display_cols.extend(["max_upside_pct", "hit_20pct"])

                available_cols = [c for c in display_cols if c in df_day.columns]
                st.dataframe(
                    df_day[available_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                # SHAP explanations
                if "shap_top_features" in df_day.columns:
                    st.subheader("Prediction Explanations (SHAP)")
                    for _, row in df_day.iterrows():
                        shap_str = row.get("shap_top_features", "")
                        if shap_str:
                            st.markdown(f"**{row['ticker']}** (P={row['probability']:.1%}): `{shap_str}`")

                # Sector breakdown
                if "sector" in df_day.columns:
                    with st.expander("Sector Breakdown"):
                        sector_counts = df_day["sector"].value_counts()
                        st.bar_chart(sector_counts)

            # Overall precision chart across all dates
            st.markdown("---")
            st.subheader("Precision Over Time (All Dates)")
            prec_df = get_precision_over_time()
            if not prec_df.empty:
                st.line_chart(prec_df.set_index("date")["precision"])
            else:
                st.info("No ground-truth evaluations available yet.")

            # Date range filter for browsing
            with st.expander("Browse All Picks"):
                st.dataframe(df_all, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    "Disclaimer: This tool is for educational purposes only. "
    "Predictions are model-based estimates, not financial advice. "
    "Past performance does not guarantee future results."
)
