"""Generate a visual diagram of the full data transformation and ML pipeline."""

from graphviz import Digraph

dot = Digraph(
    "StockPredictorPipeline",
    format="png",
    engine="dot",
    graph_attr={
        "rankdir": "TB",
        "fontsize": "14",
        "fontname": "Helvetica",
        "bgcolor": "#FAFAFA",
        "pad": "0.5",
        "nodesep": "0.4",
        "ranksep": "0.6",
        "dpi": "150",
    },
    node_attr={
        "fontname": "Helvetica",
        "fontsize": "11",
        "style": "filled",
        "shape": "box",
        "margin": "0.15,0.08",
    },
    edge_attr={
        "fontname": "Helvetica",
        "fontsize": "9",
        "color": "#555555",
    },
)

# ─── Color palette ───
DATA_COLOR = "#E3F2FD"      # light blue
PROCESS_COLOR = "#FFF3E0"   # light orange
MODEL_COLOR = "#E8F5E9"     # light green
ENSEMBLE_COLOR = "#F3E5F5"  # light purple
OUTPUT_COLOR = "#FFEBEE"    # light red
PIPELINE_COLOR = "#FFF9C4"  # light yellow

# ═══════════════════════════════════════════════════════════════
# STAGE 1: DATA COLLECTION
# ═══════════════════════════════════════════════════════════════
with dot.subgraph(name="cluster_data") as c:
    c.attr(label="Stage 1: Data Collection (10 Years)", style="rounded,filled",
           fillcolor="#E3F2FD", color="#1565C0", fontcolor="#1565C0", fontsize="13")

    c.node("yfinance", "yFinance API\n(OHLCV, Market Cap,\nSector, 52w High/Low)", fillcolor=DATA_COLOR)
    c.node("sec", "SEC EDGAR\n(Net Income, Cash Flow,\nFiling Age)", fillcolor=DATA_COLOR)
    c.node("reddit", "Reddit Sentiment\n(Mention Count,\nMean Polarity, Bullish %)", fillcolor=DATA_COLOR)
    c.node("macro", "Macro Data\n(VIX, Treasury Rates,\nS&P 500, Dollar Index)", fillcolor=DATA_COLOR)
    c.node("target", "Target Variable\n(Forward_Max_Return_3M\n≥ 20% = Class 1)", fillcolor="#BBDEFB")

# ═══════════════════════════════════════════════════════════════
# STAGE 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
with dot.subgraph(name="cluster_features") as c:
    c.attr(label="Stage 2: Feature Engineering (49 Features)", style="rounded,filled",
           fillcolor="#FFF3E0", color="#E65100", fontcolor="#E65100", fontsize="13")

    c.node("tech", "Technical (18)\nVolatility_20d, RSI_14,\nMACD_norm, BB_width,\nDist_52w_High, SMA crosses", fillcolor=PROCESS_COLOR)
    c.node("fund", "Fundamental (8)\nP/E, P/B, Debt/Equity,\nROE, Revenue Growth,\nDividend Yield", fillcolor=PROCESS_COLOR)
    c.node("sent_feat", "Sentiment (6)\nReddit mentions/polarity,\nFinviz/StockTwits polarity,\nBullish ratio", fillcolor=PROCESS_COLOR)
    c.node("macro_feat", "Macro (10)\nVIX, Treasury 10Y/3M,\nYield curve, S&P returns,\nDollar index, Gold", fillcolor=PROCESS_COLOR)
    c.node("derived", "Derived (7)\nVolume_Surge_3d,\nPrice_vs_SMA200,\nRelative_Strength_vs_SPY,\nVolatility × Momentum", fillcolor=PROCESS_COLOR)

# ═══════════════════════════════════════════════════════════════
# STAGE 3: PREPROCESSING
# ═══════════════════════════════════════════════════════════════
with dot.subgraph(name="cluster_preprocess") as c:
    c.attr(label="Stage 3: Data Preprocessing", style="rounded,filled",
           fillcolor="#FFF3E0", color="#E65100", fontcolor="#E65100", fontsize="13")

    c.node("fill", "Semantic NaN Fill\n(0 for counts/ratios,\nmedian for continuous)", fillcolor=PROCESS_COLOR)
    c.node("log", "Log Transform\n(Dollar-denominated\nfeatures: Volume,\nMarket Cap, Revenue)", fillcolor=PROCESS_COLOR)
    c.node("clip", "Feature Clipping\n(1st-99th percentile\nWinsorization)", fillcolor=PROCESS_COLOR)
    c.node("sort", "Temporal Sort\n(by _date column,\n75/25 train/test split\nwith 5% purge gap)", fillcolor=PROCESS_COLOR)

# ═══════════════════════════════════════════════════════════════
# STAGE 4: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════
with dot.subgraph(name="cluster_models") as c:
    c.attr(label="Stage 4: Dual Model Training", style="rounded,filled",
           fillcolor="#E8F5E9", color="#2E7D32", fontcolor="#2E7D32", fontsize="13")

    c.node("flaml", "FLAML AutoML\n(Classification)\nObjective: Average Precision\nBest: XGBoost\nAUC: 0.725 (test)", fillcolor=MODEL_COLOR)
    c.node("ltr", "XGBoost LambdaMART\n(Learning-to-Rank)\nObjective: NDCG@10\nGroups: trading dates\nNDCG@10: 0.665 (test)", fillcolor=MODEL_COLOR)
    c.node("regime", "Regime Detection\n(GBR on macro features)\nAuto-discard if R² < 0\nStatus: DISCARDED", fillcolor="#FFCDD2")

# ═══════════════════════════════════════════════════════════════
# STAGE 5: ENSEMBLE & POST-PROCESSING
# ═══════════════════════════════════════════════════════════════
with dot.subgraph(name="cluster_ensemble") as c:
    c.attr(label="Stage 5: Ensemble Scoring (HPO-Optimized)", style="rounded,filled",
           fillcolor="#F3E5F5", color="#6A1B9A", fontcolor="#6A1B9A", fontsize="13")

    c.node("ensemble", "Two-Stage Ensemble\nscore = 0.4 × P(class)\n        + 0.6 × σ(LTR)\n(HPO: 180 configs searched)", fillcolor=ENSEMBLE_COLOR)
    c.node("vol_adj", "Volatility-Aware Scoring\nscore × (1 + 0.25 × vol_pctl)\nHigher volatility →\nhigher breakout chance", fillcolor=ENSEMBLE_COLOR)
    c.node("cal", "Per-Ticker Calibration\nDown-weight repeat\nfalse positives\n(hit rate < 30%)", fillcolor=ENSEMBLE_COLOR)

# ═══════════════════════════════════════════════════════════════
# STAGE 6: OUTPUT
# ═══════════════════════════════════════════════════════════════
with dot.subgraph(name="cluster_output") as c:
    c.attr(label="Stage 6: Daily Predictions & MLOps", style="rounded,filled",
           fillcolor="#FFEBEE", color="#C62828", fontcolor="#C62828", fontsize="13")

    c.node("rank", "Daily Cross-Sectional\nRanking\n(rank all tickers by\nfinal score per date)", fillcolor=OUTPUT_COLOR)
    c.node("top10", "Top-10 Stock Picks\nPrecision: 67.4%\n(up from 62.5% baseline)\n+4.9% improvement", fillcolor="#EF9A9A")
    c.node("csv", "MLOps Pipeline\n(daily_picks.csv)\nTicker, Price, Prob,\nSHAP, Sentiment, Vol Surge", fillcolor=PIPELINE_COLOR)
    c.node("gt", "Ground Truth\nEvaluation\n≥20% upside check\nOverwrite only if higher", fillcolor=PIPELINE_COLOR)
    c.node("streamlit", "Streamlit Dashboard\n10 pages incl.\nSocial Media Listener,\nDaily Picks History", fillcolor=PIPELINE_COLOR)

# ═══════════════════════════════════════════════════════════════
# EDGES
# ═══════════════════════════════════════════════════════════════

# Data collection → Feature engineering
dot.edge("yfinance", "tech", label="OHLCV")
dot.edge("yfinance", "fund", label="Fundamentals")
dot.edge("sec", "fund", label="SEC filings")
dot.edge("reddit", "sent_feat", label="Social data")
dot.edge("macro", "macro_feat", label="VIX, rates")
dot.edge("yfinance", "derived", label="Price/Vol")

# Target
dot.edge("yfinance", "target", label="3M forward\nmax return")

# Feature engineering → Preprocessing
for feat in ["tech", "fund", "sent_feat", "macro_feat", "derived"]:
    dot.edge(feat, "fill")

dot.edge("fill", "log")
dot.edge("log", "clip")
dot.edge("clip", "sort")

# Preprocessing → Models
dot.edge("sort", "flaml", label="X_train, y_train")
dot.edge("sort", "ltr", label="X_train, y_train\n+ date groups")
dot.edge("sort", "regime", label="Macro features\n(daily agg)")
dot.edge("target", "flaml", label="Binary target", style="dashed")
dot.edge("target", "ltr", label="Binary labels", style="dashed")

# Models → Ensemble
dot.edge("flaml", "ensemble", label="P(≥20% gain)")
dot.edge("ltr", "ensemble", label="σ(LTR score)")
dot.edge("ensemble", "vol_adj")
dot.edge("vol_adj", "cal")
dot.edge("regime", "vol_adj", label="Regime\nconfidence", style="dashed", color="#CCCCCC")

# Ensemble → Output
dot.edge("cal", "rank")
dot.edge("rank", "top10")
dot.edge("top10", "csv")
dot.edge("csv", "gt", label="Check\n≥20% upside")
dot.edge("top10", "streamlit")
dot.edge("csv", "streamlit", label="Precision\nchart")
dot.edge("gt", "streamlit", label="Ground truth\nmetrics")

# Render
output_path = "/home/ubuntu/repos/langchain/pipeline_diagram"
dot.render(output_path, cleanup=True)
print(f"Diagram saved to {output_path}.png")
