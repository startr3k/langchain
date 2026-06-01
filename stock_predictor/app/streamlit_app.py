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
    build_incremental_dataset,
    build_training_dataset,
)
from stock_predictor.data.sentiment import (
    get_sentiment_features,
    get_sentiment_summary,
    get_trending_tickers_from_social,
)
from stock_predictor.data.yfinance_client import get_stock_data, get_stock_info
from stock_predictor.models.automl_model import StockReturnPredictor
from stock_predictor.pipeline.daily_picks import (
    run_daily_picks,
    evaluate_ground_truth,
    get_precision_over_time,
    evaluate_folds_at_pool,
    DEFAULT_CSV_PATH,
    CSV_COLUMNS,
)

# Separate CSV for Top Recommendations (always written, regardless of pool size).
# The scheduler's daily_picks.csv is only written when pool >= 75.
TOP_RECS_CSV_PATH = Path(DEFAULT_CSV_PATH).parent / "top_recommendations.csv"

# Folder for persisting per-ticker social buzz and forward guidance text files.
TICKER_DATA_DIR = Path(DEFAULT_CSV_PATH).parent / "ticker_data"
TICKER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load data dictionary for inclusion in all LLM chat contexts.
_DATA_DICT_PATH = Path(DEFAULT_CSV_PATH).parent / "data_dictionary.md"
_DATA_DICTIONARY = _DATA_DICT_PATH.read_text() if _DATA_DICT_PATH.exists() else ""
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
        "Model Explanations",
        "Daily Picks Pipeline",
        "Daily Picks History",
    ],
)

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------
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

def _fmt_rsi(val) -> str:
    """Format RSI with Overbought/Oversold label."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(v):
        return "N/A"
    if v >= 70:
        return f"{v:.1f} (Overbought)"
    elif v <= 30:
        return f"{v:.1f} (Oversold)"
    else:
        return f"{v:.1f}"

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

    # ── Helper: load today's picks from the top_recommendations CSV ─
    def _load_todays_picks() -> pd.DataFrame | None:
        """Return today's picks from the top_recommendations CSV,
        falling back to the scheduler's daily_picks CSV."""
        from datetime import date as _date

        today_str = _date.today().isoformat()

        # Try top_recommendations.csv first (always written by this page)
        for csv_path in [TOP_RECS_CSV_PATH, Path(DEFAULT_CSV_PATH)]:
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
                today_df = df[df["date"] == today_str]
                if not today_df.empty:
                    return today_df
            except Exception:
                continue
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
                "Signal": row.get("signal", "BUY"),
                "Vol Surge 3d": vol_surge_str,
                "Regime Confidence": float(row.get("regime_confidence", 0.5)),
                "Ticker Calibration": float(row.get("ticker_calibration", 1.0)),
                "Sentiment Polarity": round(sent_score, 3),
                "Total Mentions": int(row.get("sentiment_mentions", 0)),
                "RSI (14)": _fmt_rsi(row.get("rsi_14")),
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
    from datetime import date as _date
    _today_str = _date.today().isoformat()

    if "today_picks_cache" in st.session_state and st.session_state["today_picks_cache"] is not None:
        cached = st.session_state["today_picks_cache"]
        # Validate cache is from today (stale if session spans midnight)
        if "date" in cached.columns and (cached["date"] == _today_str).any():
            today_picks = cached[cached["date"] == _today_str]
            source = "session cache"
        else:
            del st.session_state["today_picks_cache"]
            today_picks = _load_todays_picks()
            source = "CSV"
            if today_picks is not None:
                st.session_state["today_picks_cache"] = today_picks
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

        today_str = _date.today().isoformat()
        recs_csv = TOP_RECS_CSV_PATH

        with st.spinner(
            "Re-running daily picks pipeline on NASDAQ universe..."
        ):
            try:
                # Always generate picks for display (save_to_csv=False
                # bypasses the pool-size gate and scheduler CSV caching).
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

                    # Always save to top_recommendations.csv
                    # (regardless of pool size — this is separate from
                    # the scheduler's daily_picks.csv)
                    if recs_csv.exists():
                        try:
                            existing = pd.read_csv(recs_csv)
                            cleaned = existing[existing["date"] != today_str]
                            cleaned.to_csv(recs_csv, index=False)
                        except Exception:
                            pass
                    else:
                        recs_csv.parent.mkdir(parents=True, exist_ok=True)
                    new_picks.reindex(columns=CSV_COLUMNS).to_csv(
                        recs_csv, mode="a",
                        header=not recs_csv.exists(),
                        index=False,
                    )
                    st.success(
                        f"Regenerated {len(today_picks)} picks! "
                        f"(pool={pool_size}, saved to top_recommendations.csv)"
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
        display_top.append(row)

    df = pd.DataFrame(display_top)

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

            # ── Social media buzz ─────────────────────────────────
            st.markdown("---")
            buzz_key = f"buzz_{ticker_name}"
            from datetime import date as _d
            _today = _d.today().isoformat()
            buzz_file = TICKER_DATA_DIR / f"{_today}_{ticker_name}_social_buzz.txt"

            # Cache hierarchy: session state → file → generate
            if buzz_key not in st.session_state and buzz_file.exists():
                st.session_state[buzz_key] = {"_from_file": True, "text": buzz_file.read_text()}

            if st.button(
                f"📊 Fetch Social Buzz ({ticker_name})",
                key=f"btn_buzz_{ticker_name}",
            ):
                with st.spinner(f"Scanning social media for {ticker_name}..."):
                    try:
                        import requests as _req
                        from bs4 import BeautifulSoup
                        from textblob import TextBlob
                        from stock_predictor.data.sentiment import get_sentiment_summary

                        headlines = []
                        resp = _req.get(
                            f"https://finviz.com/quote.ashx?t={ticker_name}",
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            soup = BeautifulSoup(resp.text, "html.parser")
                            news_table = soup.find("table", id="news-table")
                            if news_table:
                                for row in news_table.find_all("tr")[:10]:
                                    link = row.find("a")
                                    if link:
                                        text = link.get_text(strip=True)
                                        polarity = TextBlob(text).sentiment.polarity
                                        headlines.append({"headline": text, "sentiment": polarity})

                        sentiment_text = get_sentiment_summary(ticker_name)

                        # Build display text and persist to file
                        lines = [f"Social Media Buzz for {ticker_name} ({_today})", "=" * 50]
                        for h in headlines:
                            s = h["sentiment"]
                            tag = "POS" if s > 0.1 else ("NEG" if s < -0.1 else "NEU")
                            lines.append(f"[{tag} {s:+.2f}] {h['headline']}")
                        if sentiment_text:
                            lines.extend(["", "Sentiment Summary:", sentiment_text])
                        full_text = "\n".join(lines)

                        buzz_file.write_text(full_text)
                        st.session_state[buzz_key] = {
                            "headlines": headlines,
                            "sentiment_summary": sentiment_text,
                            "text": full_text,
                        }
                    except Exception as e:
                        logger.exception("Social buzz fetch failed for %s", ticker_name)
                        st.error(f"Error fetching social buzz: {e}")

            if buzz_key in st.session_state:
                buzz_result = st.session_state[buzz_key]
                st.markdown(f"**📊 Social Media Buzz for {ticker_name}**")

                if buzz_result.get("_from_file"):
                    st.text(buzz_result["text"])
                else:
                    headlines = buzz_result.get("headlines", [])
                    if headlines:
                        for item in headlines:
                            sent = item["sentiment"]
                            icon = "🟢" if sent > 0.1 else ("🔴" if sent < -0.1 else "⚪")
                            st.markdown(f"{icon} {item['headline']} `({sent:+.2f})`")
                    else:
                        st.info("No recent Finviz headlines found.")

                    sent_text = buzz_result.get("sentiment_summary", "")
                    if sent_text:
                        st.markdown("**Sentiment Summary:**")
                        st.text(sent_text)

            # ── Forward guidance ──────────────────────────────────
            st.markdown("---")
            guidance_key = f"guidance_{ticker_name}"
            guidance_file = TICKER_DATA_DIR / f"{_today}_{ticker_name}_forward_guidance.txt"

            # Cache hierarchy: session state → file → generate
            if guidance_key not in st.session_state and guidance_file.exists():
                st.session_state[guidance_key] = {
                    "found": True,
                    "_from_file": True,
                    "text": guidance_file.read_text(),
                }

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
                            # Persist to file
                            if analysis.get("found"):
                                lines = [
                                    f"Forward Guidance for {ticker_name} ({_today})",
                                    "=" * 50,
                                    f"Transcript date: {analysis.get('date', 'N/A')}",
                                    f"Source: {analysis.get('source_url', 'N/A')}",
                                    "",
                                    analysis.get("forward_guidance", ""),
                                ]
                                guidance_file.write_text("\n".join(lines))
                            st.session_state[guidance_key] = analysis
                        except Exception as e:
                            logger.exception("Transcript fetch failed for %s", ticker_name)
                            st.error(f"Error fetching transcript: {e}")

            if guidance_key in st.session_state:
                analysis = st.session_state[guidance_key]
                if analysis.get("found"):
                    if analysis.get("_from_file"):
                        st.markdown(f"**📞 Forward Guidance** (loaded from file)")
                        st.text(analysis["text"])
                    else:
                        st.markdown(f"**📞 Forward Guidance** (transcript from {analysis.get('date', 'N/A')})")
                        st.markdown(analysis.get("forward_guidance", "No guidance extracted."))
                        if analysis.get("source_url"):
                            st.caption(f"Source: [{analysis['source_url']}]({analysis['source_url']})")
                else:
                    st.warning(
                        f"No earnings call transcript found for {ticker_name}. "
                        f"{analysis.get('error', '')}"
                    )

            # ── Chatbot for this ticker ───────────────────────────
            st.markdown("---")
            chat_key = f"chat_{ticker_name}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = []

            st.markdown(f"**💬 Ask about {ticker_name}**")

            # Build context from all available data for this ticker
            def _build_ticker_context(tk: str, result: dict) -> str:
                parts = [
                    f"Ticker: {tk}",
                    f"Sector: {result.get('Sector', 'N/A')}",
                    f"Market Cap: {result.get('Market Cap', 'N/A')}",
                    f"Close Price: {result.get('Close Price', 'N/A')}",
                    f"Signal: {result.get('Signal', 'N/A')}",
                    f"Classifier P: {result.get('Classifier P', 0):.1%}",
                    f"Pred MFD: {result.get('Pred MFD', 0):.1%}",
                    f"Z_cls: {result.get('Z_cls', 0):+.2f}",
                    f"Z_ltr: {result.get('Z_ltr', 0):+.2f}",
                    f"Final Score: {result.get('Score', 0):.3f}",
                    f"Elite Pool Size: {result.get('Elite Pool Size', 0)}",
                    f"RSI (14): {result.get('RSI (14)', 'N/A')}",
                    f"Vol Surge 3d: {result.get('Vol Surge 3d', 'N/A')}",
                    f"Sentiment Polarity: {result.get('Sentiment Polarity', 0):+.3f}",
                    f"Total Mentions: {result.get('Total Mentions', 0)}",
                    f"SHAP Explanation: {result.get('SHAP Explanation', '')}",
                ]
                # Add social buzz if available
                bk = f"buzz_{tk}"
                if bk in st.session_state:
                    bd = st.session_state[bk]
                    parts.append(f"\nSocial Media Buzz:\n{bd.get('text', '')}")
                # Add forward guidance if available
                gk = f"guidance_{tk}"
                if gk in st.session_state:
                    gd = st.session_state[gk]
                    if gd.get("found"):
                        if gd.get("_from_file"):
                            parts.append(f"\nForward Guidance:\n{gd.get('text', '')}")
                        else:
                            parts.append(f"\nForward Guidance:\n{gd.get('forward_guidance', '')}")
                return "\n".join(parts)

            # Display chat history
            for msg in st.session_state[chat_key]:
                role_icon = "🧑" if msg["role"] == "user" else "🤖"
                st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")

            chat_col1, chat_col2 = st.columns([5, 1])
            with chat_col1:
                user_input = st.text_input(
                    "Ask anything...",
                    key=f"chat_input_{ticker_name}",
                    label_visibility="collapsed",
                    placeholder=f"Ask about {ticker_name}...",
                )
            with chat_col2:
                send_clicked = st.button("Send", key=f"chat_send_{ticker_name}")

            if send_clicked and user_input:
                st.session_state[chat_key].append({"role": "user", "content": user_input})

                if not api_key:
                    st.error("OpenAI API key required for chat.")
                else:
                    context = _build_ticker_context(ticker_name, r)
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a stock analysis assistant. You have detailed data about "
                                f"the stock {ticker_name} from an AI stock predictor pipeline. "
                                "Answer the user's questions based on this data. Be concise and "
                                "data-driven.\n\n"
                                f"Available data:\n{context}\n\n"
                                "## Data Dictionary Reference\n"
                                "Use this to understand every column and metric:\n\n"
                                f"{_DATA_DICTIONARY}"
                            ),
                        },
                    ]
                    for msg in st.session_state[chat_key]:
                        messages.append({"role": msg["role"], "content": msg["content"]})

                    with st.spinner("Thinking..."):
                        try:
                            from openai import OpenAI
                            client = OpenAI(api_key=api_key)
                            resp = client.chat.completions.create(
                                model=model_choice,
                                messages=messages,
                                max_completion_tokens=1000,
                            )
                            reply = resp.choices[0].message.content
                            st.session_state[chat_key].append(
                                {"role": "assistant", "content": reply}
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Chat error: {e}")



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

    sa_ticker = st.text_input("Enter Ticker Symbol", value="NVDA").upper()
    sa_cache_key = f"sa_cache_{sa_ticker}"

    # ── Analyze button: fetches everything and caches ──────────────
    if st.button("Analyze", type="primary"):
        with st.spinner(f"Fetching all data for {sa_ticker}..."):
            from datetime import date as _sa_date
            _sa_today = _sa_date.today().isoformat()
            cache = {"ticker": sa_ticker, "date": _sa_today}

            # 1. Price data
            price_df = get_stock_data(sa_ticker, period="1y")
            cache["price_df"] = price_df

            # 2. Company info
            cache["info"] = get_stock_info(sa_ticker)

            # 3. Model prediction (with SHAP explanation)
            try:
                predictor = StockReturnPredictor()
                predictor.load()
                cache["prediction"] = predictor.predict_ticker(sa_ticker, include_explanation=True)
            except FileNotFoundError:
                cache["prediction"] = {"error": "Model not trained yet."}

            # 4. Social buzz (auto-fetch)
            try:
                import requests as _req
                from bs4 import BeautifulSoup
                from textblob import TextBlob

                headlines = []
                resp = _req.get(
                    f"https://finviz.com/quote.ashx?t={sa_ticker}",
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    news_table = soup.find("table", id="news-table")
                    if news_table:
                        for row in news_table.find_all("tr")[:10]:
                            link = row.find("a")
                            if link:
                                text = link.get_text(strip=True)
                                polarity = TextBlob(text).sentiment.polarity
                                headlines.append({"headline": text, "sentiment": polarity})

                sentiment_text = get_sentiment_summary(sa_ticker)

                lines = [f"Social Media Buzz for {sa_ticker} ({_sa_today})", "=" * 50]
                for h in headlines:
                    s = h["sentiment"]
                    tag = "POS" if s > 0.1 else ("NEG" if s < -0.1 else "NEU")
                    lines.append(f"[{tag} {s:+.2f}] {h['headline']}")
                if sentiment_text:
                    lines.extend(["", "Sentiment Summary:", sentiment_text])
                full_text = "\n".join(lines)

                # Persist to file
                buzz_file = TICKER_DATA_DIR / f"{_sa_today}_{sa_ticker}_social_buzz.txt"
                buzz_file.write_text(full_text)

                cache["buzz"] = {"headlines": headlines, "sentiment_summary": sentiment_text, "text": full_text}
            except Exception as e:
                logger.exception("Social buzz fetch failed for %s", sa_ticker)
                cache["buzz"] = {"error": str(e)}

            # 5. Forward guidance (auto-fetch)
            if api_key:
                try:
                    from stock_predictor.agent.transcript_agent import analyze_ticker_forward_guidance
                    analysis = analyze_ticker_forward_guidance(
                        sa_ticker, api_key=api_key, model=model_choice,
                    )
                    if analysis.get("found"):
                        g_lines = [
                            f"Forward Guidance for {sa_ticker} ({_sa_today})",
                            "=" * 50,
                            f"Transcript date: {analysis.get('date', 'N/A')}",
                            f"Source: {analysis.get('source_url', 'N/A')}",
                            "",
                            analysis.get("forward_guidance", ""),
                        ]
                        guidance_file = TICKER_DATA_DIR / f"{_sa_today}_{sa_ticker}_forward_guidance.txt"
                        guidance_file.write_text("\n".join(g_lines))
                    cache["guidance"] = analysis
                except Exception as e:
                    logger.exception("Transcript fetch failed for %s", sa_ticker)
                    cache["guidance"] = {"error": str(e)}
            else:
                cache["guidance"] = {"error": "OpenAI API key required."}

            st.session_state[sa_cache_key] = cache

    # ── Load from cache (session state → file fallback) ───────────
    if sa_cache_key not in st.session_state:
        from datetime import date as _sa_date
        _sa_today = _sa_date.today().isoformat()

        # Try to restore from persisted files
        buzz_file = TICKER_DATA_DIR / f"{_sa_today}_{sa_ticker}_social_buzz.txt"
        guidance_file = TICKER_DATA_DIR / f"{_sa_today}_{sa_ticker}_forward_guidance.txt"

        if buzz_file.exists() or guidance_file.exists():
            cache = {"ticker": sa_ticker, "date": _sa_today, "_from_file": True}
            # Fetch price data so chart renders from cache
            try:
                cache["price_df"] = get_stock_data(sa_ticker, period="1y")
            except Exception:
                cache["price_df"] = None
            # Fetch prediction so metrics render from cache
            try:
                _restore_pred = StockReturnPredictor()
                _restore_pred.load()
                cache["prediction"] = _restore_pred.predict_ticker(sa_ticker, include_explanation=True)
            except Exception:
                cache["prediction"] = {}
            if buzz_file.exists():
                cache["buzz"] = {"_from_file": True, "text": buzz_file.read_text()}
            if guidance_file.exists():
                cache["guidance"] = {"found": True, "_from_file": True, "text": guidance_file.read_text()}
            st.session_state[sa_cache_key] = cache

    # ── Display cached results ────────────────────────────────────
    if sa_cache_key in st.session_state:
        sa_data = st.session_state[sa_cache_key]

        # ── 1. Big chart on top (full width) ──────────────────────
        price_df = sa_data.get("price_df")
        if price_df is None and not sa_data.get("_from_file"):
            pass  # file-only restore doesn't have price data
        elif price_df is not None and not price_df.empty:
            import plotly.graph_objects as go

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=price_df["Date"],
                        open=price_df["Open"],
                        high=price_df["High"],
                        low=price_df["Low"],
                        close=price_df["Close"],
                    )
                ]
            )
            fig.update_layout(
                title=f"{sa_ticker} — 1 Year",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 2. Concise prediction + company info row ──────────────
        pred = sa_data.get("prediction", {})
        info = sa_data.get("info")

        if pred.get("probability_gain") is not None:
            prob = pred["probability_gain"]
            signal = pred.get("signal", "HOLD")
            vol_surge = pred.get("volume_surge_3d")
            regime = pred.get("regime_confidence", 0.5)
            cal = pred.get("ticker_calibration", 1.0)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("P(≥20% in 3M)", f"{prob * 100:.1f}%")
            c2.metric("Signal", signal)
            c3.metric("Vol Surge 3d", f"{vol_surge:.2f}x" if vol_surge is not None else "N/A")
            c4.metric("Regime Conf.", f"{regime:.0%}")
            if cal < 1.0:
                c5.metric("Calibration", f"{cal:.2f}")
        elif pred.get("error"):
            st.info(pred["error"])

        if info:
            with st.expander("Company Info"):
                info_cols = st.columns(min(len(info), 4))
                for i, (key, value) in enumerate(info.items()):
                    info_cols[i % len(info_cols)].metric(key, str(value))

        # ── SHAP Explanation ───────────────────────────────────────
        shap_items = pred.get("explanation", [])
        if shap_items:
            st.markdown("---")
            st.markdown("**🔍 SHAP Explanation** (top contributing features)")
            shap_cols = st.columns(len(shap_items))
            for i, item in enumerate(shap_items):
                direction = "↑" if item["direction"] == "+" else "↓"
                shap_cols[i].metric(
                    item["feature"],
                    f"{item['feature_value']:.3f}",
                    f"{direction} {item['shap_value']:+.4f}",
                )

        # ── 3. Social Buzz ────────────────────────────────────────
        buzz = sa_data.get("buzz")
        if buzz and not buzz.get("error"):
            st.markdown("---")
            st.markdown(f"**📊 Social Media Buzz**")
            if buzz.get("_from_file"):
                st.text(buzz["text"])
            else:
                headlines = buzz.get("headlines", [])
                if headlines:
                    for item in headlines:
                        sent = item["sentiment"]
                        icon = "🟢" if sent > 0.1 else ("🔴" if sent < -0.1 else "⚪")
                        st.markdown(f"{icon} {item['headline']} `({sent:+.2f})`")
                sent_text = buzz.get("sentiment_summary", "")
                if sent_text:
                    st.caption(sent_text)

        # ── 4. Forward Guidance ───────────────────────────────────
        guidance = sa_data.get("guidance")
        if guidance:
            st.markdown("---")
            if guidance.get("error"):
                st.warning(f"Forward guidance: {guidance['error']}")
            elif guidance.get("found") or guidance.get("forward_guidance"):
                if guidance.get("_from_file"):
                    st.markdown(f"**📞 Forward Guidance** (loaded from file)")
                    st.text(guidance["text"])
                else:
                    st.markdown(f"**📞 Forward Guidance** (transcript from {guidance.get('date', 'N/A')})")
                    st.markdown(guidance.get("forward_guidance", "No guidance extracted."))
                    if guidance.get("source_url"):
                        st.caption(f"Source: [{guidance['source_url']}]({guidance['source_url']})")
            elif not guidance.get("found") and not guidance.get("error"):
                st.info(f"No earnings call transcript found for {sa_ticker}.")

        # ── 5. Chatbot ────────────────────────────────────────────
        st.markdown("---")
        sa_chat_key = f"sa_chat_{sa_ticker}"
        if sa_chat_key not in st.session_state:
            st.session_state[sa_chat_key] = []

        st.markdown(f"**💬 Ask about {sa_ticker}**")

        def _build_sa_context(tk: str, data: dict) -> str:
            parts = [f"Ticker: {tk}"]
            pred = data.get("prediction", {})
            if pred.get("probability_gain") is not None:
                parts.extend([
                    f"P(>=20% in 3M): {pred['probability_gain']:.1%}",
                    f"Signal: {pred.get('signal', 'N/A')}",
                    f"Vol Surge 3d: {pred.get('volume_surge_3d', 'N/A')}",
                    f"Regime Confidence: {pred.get('regime_confidence', 'N/A')}",
                    f"Ticker Calibration: {pred.get('ticker_calibration', 'N/A')}",
                ])
                # SHAP
                expl = pred.get("explanation_str", "")
                if expl:
                    parts.append(f"SHAP Explanation: {expl}")
            info = data.get("info")
            if info:
                parts.append(f"\nCompany Info: {json.dumps(info, default=str)}")
            buzz = data.get("buzz")
            if buzz and not buzz.get("error"):
                parts.append(f"\nSocial Media Buzz:\n{buzz.get('text', '')}")
            guidance = data.get("guidance")
            if guidance and not guidance.get("error"):
                if guidance.get("_from_file"):
                    parts.append(f"\nForward Guidance:\n{guidance.get('text', '')}")
                elif guidance.get("forward_guidance"):
                    parts.append(f"\nForward Guidance:\n{guidance.get('forward_guidance', '')}")
            return "\n".join(parts)

        for msg in st.session_state[sa_chat_key]:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")

        sa_chat_col1, sa_chat_col2 = st.columns([5, 1])
        with sa_chat_col1:
            sa_user_input = st.text_input(
                "Ask anything...",
                key=f"sa_chat_input_{sa_ticker}",
                label_visibility="collapsed",
                placeholder=f"Ask about {sa_ticker}...",
            )
        with sa_chat_col2:
            sa_send = st.button("Send", key=f"sa_chat_send_{sa_ticker}")

        if sa_send and sa_user_input:
            st.session_state[sa_chat_key].append({"role": "user", "content": sa_user_input})

            if not api_key:
                st.error("OpenAI API key required for chat.")
            else:
                context = _build_sa_context(sa_ticker, sa_data)
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a stock analysis assistant. You have detailed data about "
                            f"the stock {sa_ticker} from an AI stock predictor pipeline. "
                            "Answer the user's questions based on this data. Be concise and "
                            "data-driven.\n\n"
                            f"Available data:\n{context}\n\n"
                            "## Data Dictionary Reference\n"
                            "Use this to understand every column and metric:\n\n"
                            f"{_DATA_DICTIONARY}"
                        ),
                    },
                ]
                for msg in st.session_state[sa_chat_key]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                with st.spinner("Thinking..."):
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        resp = client.chat.completions.create(
                            model=model_choice,
                            messages=messages,
                            max_completion_tokens=1000,
                        )
                        reply = resp.choices[0].message.content
                        st.session_state[sa_chat_key].append(
                            {"role": "assistant", "content": reply}
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Chat error: {e}")


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
# Page: Model Explanations
# ---------------------------------------------------------------------------
elif page == "Model Explanations":
    st.title("Model Explanations")
    st.markdown(
        "Explore the trained model's per-fold metrics, feature importances, "
        "and SHAP explanations. Training is available on the Daily Picks Pipeline page."
    )

    # ── Load model ────────────────────────────────────────────────
    _me_predictor = None
    try:
        _me_predictor = StockReturnPredictor()
        _me_predictor.load()
    except (FileNotFoundError, RuntimeError):
        _me_predictor = None
        st.warning("No trained model found. Train the model on the Daily Picks Pipeline page.")

    # ── 1. Per-Fold Model Metrics ─────────────────────────────────
    if _me_predictor is not None:
        import pickle as _me_pkl

        st.subheader("Per-Fold Model Metrics")
        st.caption(
            "Results from the 4-stage walk-forward ensemble. Each fold trains on "
            "all prior data and evaluates on the next time window."
        )

        _inter_dir = Path(__file__).resolve().parent.parent.parent / "intermediates"
        _fold_files = sorted(_inter_dir.glob("fold*.pkl")) if _inter_dir.exists() else []

        if _fold_files:
            # Load per-fold metrics from intermediates
            _fold_metrics = []
            for _fp in _fold_files:
                _fidx = int(_fp.stem.replace("fold", ""))
                with open(_fp, "rb") as _ff:
                    _fd = _me_pkl.load(_ff)
                _fold_metrics.append({
                    "fold": _fidx,
                    "cls_auc": _fd.get("cls_auc"),
                    "huber_r2": _fd.get("huber_r2"),
                })

            # Evaluate pool >= 150 hit rate per fold
            _pool_eval = evaluate_folds_at_pool(min_pool=150)

            # Build per-fold display table
            _fold_rows = []
            _pool_folds = {f["fold"]: f for f in _pool_eval.get("folds", [])} if not _pool_eval.get("error") else {}
            for _fm in _fold_metrics:
                _pf = _pool_folds.get(_fm["fold"], {})
                _row = {
                    "Fold": _fm["fold"],
                    "Stage 1: Cls AUC": f"{_fm['cls_auc']:.4f}" if _fm["cls_auc"] is not None else "N/A",
                    "Stage 2: Huber R²": f"{_fm['huber_r2']:.4f}" if _fm["huber_r2"] is not None else "N/A",
                }
                if _pf:
                    _days = _pf.get("n_days", 0)
                    _hits = _pf.get("total_hits", 0)
                    _picks = _pf.get("total_picks", 0)
                    _hr = _pf.get("hit_rate", 0)
                    _row["Pool≥150 Days"] = _days
                    _row["Pool≥150 Hits"] = f"{_hits}/{_picks}" if _picks > 0 else "N/A"
                    _row["Pool≥150 Hit Rate"] = f"{_hr:.1%}" if _days > 0 else "N/A"
                else:
                    _row["Pool≥150 Days"] = 0
                    _row["Pool≥150 Hits"] = "N/A"
                    _row["Pool≥150 Hit Rate"] = "N/A"
                _fold_rows.append(_row)

            st.dataframe(
                pd.DataFrame(_fold_rows),
                use_container_width=True,
                hide_index=True,
            )

            # Aggregate summary
            if not _pool_eval.get("error"):
                _agg_days = _pool_eval.get("aggregate_days", 0)
                _agg_hits = _pool_eval.get("aggregate_hits", 0)
                _agg_picks = _pool_eval.get("aggregate_picks", 0)
                _agg_hr = _pool_eval.get("aggregate_hit_rate", 0)
                _valid_aucs = [fm["cls_auc"] for fm in _fold_metrics if fm["cls_auc"] is not None]
                _mean_auc = sum(_valid_aucs) / max(len(_valid_aucs), 1)
                _valid_r2s = [fm["huber_r2"] for fm in _fold_metrics if fm["huber_r2"] is not None]
                _mean_r2 = sum(_valid_r2s) / max(len(_valid_r2s), 1)

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Mean Cls AUC", f"{_mean_auc:.4f}")
                a2.metric("Mean Huber R²", f"{_mean_r2:.4f}")
                a3.metric(
                    "Aggregate Hit Rate (Pool≥150)",
                    f"{_agg_hr:.1%}",
                    help=f"{_agg_hits}/{_agg_picks} hits across {_agg_days} days",
                )
                a4.metric("Total Pool≥150 Days", _agg_days)
        else:
            st.info(
                "No fold intermediates found. Retrain the model on the "
                "Daily Picks Pipeline page to generate per-fold metrics."
            )

        # ── 2. SHAP Summary Plot ─────────────────────────────────────
        st.markdown("---")
        st.subheader("SHAP Feature Impact")
        st.caption(
            "Shows how each feature's value pushes the model's probability "
            "up or down. Red = high feature value, blue = low."
        )

        if st.button("Generate SHAP Plot", type="primary"):
            with st.spinner("Computing SHAP values (this may take a moment)..."):
                try:
                    import shap
                    import numpy as _np
                    import plotly.graph_objects as _shap_me_go
                    import plotly.express as _shap_me_px

                    # Get the underlying model
                    model = _me_predictor.automl.model.estimator
                    if hasattr(model, "feature_name_"):
                        model_features = model.feature_name_
                    elif hasattr(model, "feature_names_in_"):
                        model_features = list(model.feature_names_in_)
                    else:
                        model_features = _me_predictor.feature_names

                    # Build a background dataset from training CSV
                    _csv_path = Path(__file__).resolve().parent.parent.parent / "training_data_10y_full.csv"
                    if _csv_path.exists():
                        _bg_df = pd.read_csv(_csv_path, nrows=5000)
                        from stock_predictor.models.automl_model import _compute_derived_features
                        _bg_df = _compute_derived_features(_bg_df)
                        for col in _me_predictor.feature_names:
                            if col not in _bg_df.columns:
                                _bg_df[col] = 0.0
                        _bg_df = _bg_df[[c for c in model_features if c in _bg_df.columns]]
                        if _me_predictor.feature_medians is not None:
                            _bg_df = _bg_df.fillna(_me_predictor.feature_medians)
                        else:
                            _bg_df = _bg_df.fillna(0.0)

                        # Sample for SHAP computation
                        _sample = _bg_df.sample(min(500, len(_bg_df)), random_state=42)

                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer(_sample)

                        # Handle multi-output (binary classifier)
                        sv = shap_values.values
                        if sv.ndim == 3:
                            sv = sv[:, :, 1]  # class 1

                        _feat_names = list(_sample.columns)
                        _max_display = 20

                        # Rank features by mean |SHAP|
                        _mean_abs = _np.abs(sv).mean(axis=0)
                        _top_idx = _np.argsort(_mean_abs)[-_max_display:][::-1]

                        # ── Beeswarm plot (Plotly scatter) ──
                        _bee_rows = []
                        for rank, fi in enumerate(_top_idx):
                            feat = _feat_names[fi]
                            vals = sv[:, fi]
                            feat_vals = _sample.iloc[:, fi].values
                            # Normalize feature values to [0, 1] for color
                            fmin, fmax = _np.nanmin(feat_vals), _np.nanmax(feat_vals)
                            if fmax > fmin:
                                feat_norm = (feat_vals - fmin) / (fmax - fmin)
                            else:
                                feat_norm = _np.full_like(feat_vals, 0.5)
                            for j in range(len(vals)):
                                _bee_rows.append({
                                    "Feature": feat,
                                    "SHAP Value": float(vals[j]),
                                    "Feature Value (normalized)": float(feat_norm[j]),
                                    "Feature Value": float(feat_vals[j]),
                                    "_rank": rank,
                                })

                        _bee_df = pd.DataFrame(_bee_rows)
                        # Map feature names to numeric y positions + jitter
                        _ordered_feats = [_feat_names[i] for i in _top_idx][::-1]
                        _feat_to_y = {f: i for i, f in enumerate(_ordered_feats)}
                        _bee_df["_y"] = _bee_df["Feature"].map(_feat_to_y)
                        _bee_df["_y_jitter"] = _bee_df["_y"] + _np.random.default_rng(42).uniform(
                            -0.35, 0.35, size=len(_bee_df)
                        )

                        _bee_fig = _shap_me_px.scatter(
                            _bee_df,
                            x="SHAP Value",
                            y="_y_jitter",
                            color="Feature Value (normalized)",
                            color_continuous_scale=["#3B4CC0", "#B40426"],
                            hover_data={"Feature Value": ":.4f", "SHAP Value": ":+.4f", "Feature Value (normalized)": False, "_rank": False, "_y_jitter": False, "_y": False, "Feature": True},
                        )
                        _bee_fig.update_layout(
                            height=max(500, _max_display * 30),
                            xaxis_title="SHAP Value (impact on model output)",
                            yaxis_title="",
                            yaxis=dict(
                                tickvals=list(range(len(_ordered_feats))),
                                ticktext=_ordered_feats,
                            ),
                            coloraxis_colorbar_title="Feature<br>Value",
                        )
                        _bee_fig.update_traces(marker_size=3, marker_opacity=0.6)
                        st.plotly_chart(_bee_fig, use_container_width=True)

                        # ── Mean |SHAP| bar plot (Plotly) ──
                        st.markdown("#### Mean Absolute SHAP Impact")
                        _bar_names = [_feat_names[i] for i in _top_idx][::-1]
                        _bar_vals = [float(_mean_abs[i]) for i in _top_idx][::-1]
                        _bar_fig = _shap_me_go.Figure(_shap_me_go.Bar(
                            x=_bar_vals,
                            y=_bar_names,
                            orientation="h",
                            marker_color="#636EFA",
                            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
                        ))
                        _bar_fig.update_layout(
                            height=max(400, _max_display * 28),
                            xaxis_title="Mean |SHAP Value|",
                            yaxis_title="",
                            showlegend=False,
                        )
                        st.plotly_chart(_bar_fig, use_container_width=True)

                        st.session_state["_shap_computed"] = True
                    else:
                        st.warning("Training CSV not found. Cannot compute SHAP values.")

                except Exception as e:
                    logger.exception("SHAP computation failed")
                    st.error(f"SHAP computation failed: {e}")

        # ── 3. Feature Importances (Grouped) ─────────────────────────
        grouped = _me_predictor.get_grouped_feature_importance(top_n=25)
        if grouped:
            st.markdown("---")
            st.subheader("Feature Importances (Grouped by Correlation)")
            st.caption(
                "Classification model feature importances. Correlated features "
                "(Spearman |r| > 0.70) are summed into concept-level groups "
                "so importance isn't diluted across redundant features."
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

    # -- Ground truth evaluation (pool >= 150) -----------------------------
    st.markdown("---")
    st.subheader("🔍 Evaluate Ground Truth (Pool ≥ 150)")
    st.markdown(
        "Check all historical picks for ≥20% upside from the recorded "
        "closing price to date. Only includes days where elite pool ≥ 150. "
        "Updates existing records only if the new upside is higher."
    )
    if st.button("🔍 Evaluate Ground Truth"):
        with st.spinner("Checking price history for all recorded picks..."):
            try:
                df_gt = evaluate_ground_truth()
                if df_gt.empty:
                    st.info("No picks recorded yet.")
                else:
                    evaluated = df_gt[df_gt["hit_20pct"].notna()]
                    # Filter to pool >= 150
                    if "elite_pool_size" in evaluated.columns:
                        evaluated = evaluated[evaluated["elite_pool_size"] >= 150]
                    if evaluated.empty:
                        st.info("No picks with pool ≥ 150 old enough to evaluate yet.")
                    else:
                        hits = int(evaluated["hit_20pct"].sum())
                        total = len(evaluated)
                        prec = hits / total if total > 0 else 0
                        st.metric("Hit Rate (Pool ≥ 150)", f"{prec:.1%}", f"{hits}/{total} hits")
                        display_cols = ["date", "ticker", "probability", "close_price",
                                        "elite_pool_size", "max_upside_pct", "hit_20pct"]
                        display_cols = [c for c in display_cols if c in evaluated.columns]
                        st.dataframe(evaluated[display_cols], use_container_width=True)
            except Exception as e:
                st.error(f"Evaluation error: {e}")

    # Show precision over time (pool >= 150) if data exists
    st.markdown("---")
    st.subheader("Precision Over Time (Pool ≥ 150)")
    prec_df = get_precision_over_time(min_pool_size=150)
    if not prec_df.empty:
        st.line_chart(prec_df.set_index("date")["precision"])

        col_a, col_b, col_c = st.columns(3)
        overall_prec = prec_df["hits"].sum() / prec_df["total_picks"].sum() if prec_df["total_picks"].sum() > 0 else 0
        col_a.metric("Hit Rate (Pool ≥ 150)", f"{overall_prec:.1%}")
        col_b.metric("Total Picks", int(prec_df["total_picks"].sum()))
        col_c.metric("Days Tracked", len(prec_df))
    else:
        st.info("No evaluated picks with pool ≥ 150 yet.")

    # -- Retrain model -----------------------------------------------------
    st.markdown("---")
    st.subheader("🔄 Retrain Model")
    st.markdown(
        "Retrain the model with an extended dataset. **Incremental mode** "
        "only downloads new rows since the last training date — existing "
        "data is kept as-is. Saves to `training_data_extended.csv` "
        "(does NOT overwrite `training_data_10y_full.csv`)."
    )

    # Track whether a retrain is currently running
    if "retrain_in_progress" not in st.session_state:
        st.session_state["retrain_in_progress"] = False

    _retrain_running = st.session_state["retrain_in_progress"]

    if _retrain_running:
        st.warning("⏳ Model retraining is in progress. Please wait for it to finish.")

    retrain_col1, retrain_col2 = st.columns(2)
    with retrain_col1:
        retrain_time = st.slider("FLAML time budget (seconds)", 60, 600, 300, key="retrain_time",
                                 disabled=_retrain_running)
    with retrain_col2:
        retrain_folds = st.slider("Walk-forward folds", 3, 10, 5, key="retrain_folds",
                                  disabled=_retrain_running)

    if st.button("🔄 Retrain Model", type="primary", disabled=_retrain_running):
        import os as _os

        st.session_state["retrain_in_progress"] = True

        _project_root = Path(DEFAULT_CSV_PATH).parent
        _original_csv = _project_root / "training_data_10y_full.csv"
        _extended_csv = _project_root / "training_data_extended.csv"

        progress = st.progress(0)
        status = st.empty()

        try:
            # Use extended CSV if it already exists (previous retrain), else original
            _base_csv = _extended_csv if _extended_csv.exists() else _original_csv

            if _base_csv.exists():
                # ── Incremental mode ──
                status.info(f"Step 1/4: Loading existing dataset from {_base_csv.name}...")
                progress.progress(5)

                original_df = pd.read_csv(_base_csv)
                original_df["_date"] = pd.to_datetime(original_df["_date"])

                # Use ALL tickers from the existing CSV, not just the hardcoded 50
                _tickers = original_df["Ticker"].unique().tolist()

                _max_date = original_df["_date"].max()
                status.info(
                    f"Existing data: {len(original_df):,} rows, "
                    f"{len(_tickers)} tickers, "
                    f"latest date: {_max_date.date()}"
                )

                status.info("Step 2/4: Fetching new data incrementally...")
                progress.progress(10)

                new_rows_df = build_incremental_dataset(_tickers, original_df)
                progress.progress(25)

                if new_rows_df.empty:
                    status.info("No new rows to add — dataset is already up to date.")
                    combined = original_df
                else:
                    new_rows_df["_date"] = pd.to_datetime(new_rows_df["_date"])
                    combined = pd.concat([original_df, new_rows_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["Ticker", "_date"], keep="first")
                    combined = combined.sort_values(["Ticker", "_date"]).reset_index(drop=True)

                    status.info(
                        f"Extended dataset: {len(combined):,} rows "
                        f"(+{len(new_rows_df):,} new rows from incremental fetch)"
                    )
            else:
                # ── Full build (no existing CSV) ──
                from stock_predictor.config import get_eligible_tickers
                _tickers = get_eligible_tickers()
                status.info("Step 1/4: No existing dataset — building full training data...")
                progress.progress(5)
                combined = build_training_dataset(_tickers, include_sentiment=False)
                progress.progress(25)
                status.info(f"Built {len(combined):,} rows, {combined['Ticker'].nunique()} tickers")

            # Step 3: Save extended dataset (NOT overwriting original)
            combined.to_csv(_extended_csv, index=False)
            progress.progress(40)
            status.info(f"Saved extended dataset to {_extended_csv.name}")

            # Step 4: Retrain walk-forward on extended data
            status.info("Step 3/4: Training walk-forward model on extended dataset...")
            progress.progress(50)

            predictor = StockReturnPredictor()
            metrics = predictor.train_walk_forward(
                df=combined,
                time_budget=retrain_time,
                n_folds=retrain_folds,
            )
            progress.progress(90)

            # Step 4: Evaluate at pool >= 150 and save results
            status.info("Step 4/4: Evaluating folds at pool ≥ 150...")
            eval_file = _project_root / "fold_evaluation_pool150.txt"
            eval_result = evaluate_folds_at_pool(min_pool=150, output_file=eval_file)

            progress.progress(100)
            status.success(
                f"Retrain complete! {metrics.get('n_folds', 0)} folds, "
                f"aggregate hit rate: {metrics.get('aggregate_top10_hit_rate', 0):.1%}"
            )

            # Show aggregate summary
            if not eval_result.get("error"):
                st.markdown(f"**Pool ≥ 150 evaluation**: "
                            f"{eval_result['aggregate_hits']}/{eval_result['aggregate_picks']} hits "
                            f"({eval_result['aggregate_hit_rate']:.1%}) over "
                            f"{eval_result['aggregate_days']} days")

                # Show sample per fold (3 days each)
                for fr in eval_result.get("folds", []):
                    with st.expander(
                        f"Fold {fr['fold']} — {fr['n_days']} days, "
                        f"Hit rate: {fr['hit_rate']:.1%}"
                    ):
                        sample = fr["details"][:3]
                        for dd in sample:
                            picks_str = ", ".join(
                                f"{'✓' if p['hit'] else '✗'} {p['ticker']} ({p['actual_return']:.0%})"
                                for p in dd["picks"]
                            )
                            st.markdown(
                                f"**{dd['date']}** | Pool {dd['pool']} | "
                                f"{dd['hits']}/10 hits | {picks_str}"
                            )
                        if len(fr["details"]) > 3:
                            st.caption(f"... and {len(fr['details']) - 3} more days. "
                                       f"Full results in {eval_file.name}")

                st.info(f"Full evaluation written to `{eval_file.name}`")

            # Show fold metrics table
            if metrics.get("folds"):
                fold_table = []
                for f in metrics["folds"]:
                    fold_table.append({
                        "Fold": f["fold"],
                        "Test Period": f["test_period"],
                        "AUC": f["auc_test"],
                        "Hit Rate": f"{f['top10_hit_rate']:.1%}",
                        "Days": f["n_eval_days"],
                    })
                st.dataframe(pd.DataFrame(fold_table), use_container_width=True)

        except Exception as e:
            logger.exception("Retrain failed")
            st.error(f"Retrain error: {e}")
        finally:
            st.session_state["retrain_in_progress"] = False

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

                # Picks table — match Top Recommendations column names
                rename_map = {
                    "ticker": "Ticker",
                    "probability": "Model P(≥20%)",
                    "cls_proba": "Classifier P",
                    "pred_mfd": "Pred MFD",
                    "z_cls": "Z_cls",
                    "z_ltr": "Z_ltr",
                    "ensemble_score": "Score",
                    "elite_pool_size": "Elite Pool Size",
                    "signal": "Signal",
                    "volume_surge_3d": "Vol Surge 3d",
                    "regime_confidence": "Regime Confidence",
                    "ticker_calibration": "Ticker Calibration",
                    "sentiment_score": "Sentiment Polarity",
                    "sentiment_mentions": "Total Mentions",
                    "rsi_14": "RSI (14)",
                    "market_cap": "Market Cap",
                    "sector": "Sector",
                    "close_price": "Close Price",
                    "max_upside_pct": "Max Upside %",
                    "hit_20pct": "Hit 20%",
                }
                display_order = [
                    "Ticker", "Model P(≥20%)", "Classifier P", "Pred MFD",
                    "Z_cls", "Z_ltr", "Score", "Elite Pool Size", "Signal",
                    "Vol Surge 3d", "Regime Confidence", "Ticker Calibration",
                    "Sentiment Polarity", "Total Mentions", "RSI (14)",
                    "Market Cap", "Sector", "Close Price",
                    "Max Upside %", "Hit 20%",
                ]
                available_raw = [c for c in rename_map if c in df_day.columns]
                df_display = df_day[available_raw].rename(columns=rename_map)

                # Format columns
                if "Vol Surge 3d" in df_display.columns:
                    df_display["Vol Surge 3d"] = df_display["Vol Surge 3d"].apply(
                        lambda v: f"{v:.2f}x" if pd.notna(v) and v else "N/A"
                    )
                if "RSI (14)" in df_display.columns:
                    df_display["RSI (14)"] = df_display["RSI (14)"].apply(_fmt_rsi)
                if "Market Cap" in df_display.columns:
                    df_display["Market Cap"] = df_display["Market Cap"].apply(_fmt_mcap)

                ordered_cols = [c for c in display_order if c in df_display.columns]
                st.dataframe(
                    df_display[ordered_cols].style.format({
                        "Model P(≥20%)": "{:.1%}",
                        "Classifier P": "{:.1%}",
                        "Pred MFD": "{:.1%}",
                        "Z_cls": "{:+.2f}",
                        "Z_ltr": "{:+.2f}",
                        "Score": "{:.3f}",
                        "Sentiment Polarity": "{:+.3f}",
                    }, na_rep="N/A"),
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
