# Data Dictionary

This document describes every column in the training dataset (`training_data_10y_full.csv`), the model feature lists, and the daily picks output schema.

---

## 1. Training Dataset (`training_data_10y_full.csv`)

**Rows:** ~1.23 million | **Tickers:** 616 NASDAQ | **Period:** ~10 years | **Granularity:** 1 row per ticker per trading day

### Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `Ticker` | str | NASDAQ ticker symbol (e.g. AAPL, NVDA) |
| `_date` | str | Trading date (YYYY-MM-DD) |

### Target Column

| Column | Type | Description |
|--------|------|-------------|
| `Forward_Max_Return_3M` | float | Maximum forward drawup (MFD) within 3 months from `_date`. Measures peak upside from the closing price. Used as the regression target and binarized at ≥ 20% for classification. |

### Technical Features (24)

Derived from price and volume data. Inherently time-correct (computed from historical OHLCV).

| Column | Type | Description |
|--------|------|-------------|
| `Return_1d` | float | 1-day price return |
| `Return_5d` | float | 5-day (1 week) price return |
| `Return_20d` | float | 20-day (~1 month) price return |
| `Return_60d` | float | 60-day (~3 month) price return |
| `Price_to_SMA_20` | float | Price / 20-day simple moving average — short-term trend |
| `Price_to_SMA_50` | float | Price / 50-day SMA — medium-term trend |
| `Price_to_SMA_200` | float | Price / 200-day SMA — long-term trend |
| `Days_Since_SMA200_Cross` | int | Trading days since price last crossed the 200-day SMA |
| `Volatility_20d` | float | 20-day rolling standard deviation of returns |
| `Volatility_60d` | float | 60-day rolling standard deviation of returns |
| `Volatility_Contraction` | float | Ratio of short-term to long-term volatility (< 1 = contraction / squeeze) |
| `Momentum_Accel` | float | 5-day return minus 20-day return — momentum acceleration signal |
| `Volume_Price_Confirm` | float | Correlation between volume and price direction over recent window |
| `Dist_52w_High` | float | Distance from 52-week high as a fraction (0 = at high, negative = below) |
| `Dist_52w_Low` | float | Distance from 52-week low as a fraction (positive = above low) |
| `BB_Squeeze_Duration` | int | Consecutive days Bollinger Band width has been contracting |
| `Volume_Ratio` | float | Current volume / 20-day average volume |
| `Volume_Surge_3d` | float | 3-day volume surge relative to 20-day average |
| `Volume_Spike_Magnitude` | float | Peak single-day volume spike magnitude |
| `RSI_14` | float | 14-day Relative Strength Index (0-100) |
| `MACD` | float | MACD line value (12-26 EMA difference) |
| `MACD_Hist` | float | MACD histogram (MACD line minus signal line) |
| `BB_Width` | float | Bollinger Band width (upper - lower) / middle |
| `BB_Position` | float | Price position within Bollinger Bands (0 = lower, 1 = upper) |

### Historical Fundamental Features (12)

From quarterly financial filings (YFinance quarterly data). Time-aligned — each row uses the most recent filing available at that date.

| Column | Type | Description |
|--------|------|-------------|
| `hist_total_revenue` | float | Total revenue from most recent quarterly filing |
| `hist_operating_income` | float | Operating income (EBIT) |
| `hist_net_income` | float | Net income |
| `hist_diluted_eps` | float | Diluted earnings per share |
| `hist_operating_margin` | float | Operating margin (operating income / revenue) |
| `hist_profit_margin` | float | Net profit margin (net income / revenue) |
| `hist_total_assets` | float | Total assets from balance sheet |
| `hist_stockholders_equity` | float | Total stockholders' equity |
| `hist_debt_to_equity` | float | Total debt / stockholders' equity ratio |
| `hist_roe` | float | Return on equity (net income / equity) |
| `hist_roa` | float | Return on assets (net income / total assets) |
| `hist_capex` | float | Capital expenditures |
| `hist_earnings_growth_qoq` | float | Quarter-over-quarter earnings growth rate |

### Macroeconomic Features (8)

Market-wide indicators. Time-indexed to each trading day.

| Column | Type | Description |
|--------|------|-------------|
| `vix_close` | float | CBOE Volatility Index (VIX) closing value — market fear gauge |
| `treasury_3m` | float | 3-month US Treasury yield |
| `yield_curve_spread` | float | 10Y minus 3M Treasury yield spread (negative = inverted curve) |
| `sp500_return_20d` | float | S&P 500 20-day return — broad market momentum |
| `sp500_return_60d` | float | S&P 500 60-day return |
| `sp500_volatility_20d` | float | S&P 500 20-day realized volatility |
| `dollar_index_return_20d` | float | US Dollar Index 20-day return |
| `oil_return_20d` | float | Crude oil 20-day return |

### SEC EDGAR Features (2)

From SEC XBRL filings. Time-aligned to filing dates.

| Column | Type | Description |
|--------|------|-------------|
| `sec_operating_cash_flow` | float | Operating cash flow from most recent SEC filing |
| `sec_filing_age_days` | int | Days since the most recent SEC filing — staleness indicator |

### Insider Transaction Features (2)

From SEC Form 4 filings. Time-aligned to filing dates.

| Column | Type | Description |
|--------|------|-------------|
| `insider_net_buys_90d` | float | Net insider buy transactions in last 90 days (buys minus sells) |
| `insider_buy_ratio_90d` | float | Ratio of insider buys to total insider transactions in last 90 days |

### Dropped Features (in CSV but excluded from model)

These columns exist in the training CSV but are excluded from `ALL_FEATURE_NAMES` via `DROPPED_FEATURES` due to negative permutation importance or multicollinearity.

| Column | Reason Dropped |
|--------|---------------|
| `treasury_10y` | Highly correlated with `treasury_3m` and `yield_curve_spread` |
| `hist_current_ratio` | Negative permutation importance — adds noise |
| `hist_revenue_growth_qoq` | Negative permutation importance — redundant with margin features |
| `gold_return_20d` | Negative permutation importance — not predictive |
| `insider_total_transactions_90d` | Negative permutation importance — `insider_net_buys_90d` captures the signal |

---

## 2. Feature Lists Not in Training CSV

These features are defined in `ALL_FEATURE_NAMES` (via their source modules) for live inference via `build_training_row()`, but were not included when the 10-year training dataset was generated. At training time, they are filtered out by `[c for c in ALL_FEATURE_NAMES if c in df.columns]`. At inference time via `predict_batch()` (which uses the cached training CSV), they are also absent and filled with 0.0.

| Feature | Source Module | Why Absent |
|---------|--------------|------------|
| `earnings_surprise_pct` | `earnings_data.py` | Not included in dataset generation pipeline |
| `earnings_eps_actual` | `earnings_data.py` | Not included in dataset generation pipeline |
| `days_since_last_earnings` | `earnings_data.py` | Not included in dataset generation pipeline |
| `days_to_next_earnings` | `earnings_data.py` | Not included in dataset generation pipeline |
| `hist_total_debt` | `historical_fundamentals.py` | Not included in dataset generation pipeline |
| `hist_book_value_per_share` | `historical_fundamentals.py` | Not included in dataset generation pipeline |
| `hist_current_assets` | `historical_fundamentals.py` | Not included in dataset generation pipeline |
| `sec_net_income` | `sec_edgar.py` | Not included in dataset generation pipeline |
| `short_percent_of_float` | `short_interest.py` | Cross-sectional snapshot; not available historically |
| `short_ratio` | `short_interest.py` | Cross-sectional snapshot; not available historically |
| `short_interest_change` | `short_interest.py` | Cross-sectional snapshot; not available historically |
| `put_call_ratio` | `options_flow.py` | Cross-sectional snapshot; not available historically |
| `call_volume_ratio` | `options_flow.py` | Cross-sectional snapshot; not available historically |
| `iv_skew` | `options_flow.py` | Cross-sectional snapshot; not available historically |
| `reddit_mention_count_7d` | `reddit_sentiment.py` | Historical Reddit data not included in pipeline |
| `reddit_mean_sentiment_7d` | `reddit_sentiment.py` | Historical Reddit data not included in pipeline |
| `reddit_bullish_ratio_7d` | `reddit_sentiment.py` | Historical Reddit data not included in pipeline |

---

## 3. Model Training Configuration

| Parameter | Value |
|-----------|-------|
| **Features used** | 49 (intersection of ALL_FEATURE_NAMES and training CSV columns) |
| **Target** | `Forward_Max_Return_3M >= 0.20` (binary classification) |
| **Quality filter** | ≥ 126 rows per ticker AND > 50% non-null revenue |
| **Training method** | 5-fold expanding window walk-forward CV with 63-day embargo |
| **Preprocessing** | Per-fold median NaN fill + 1st/99th percentile clipping |
| **Stage 1** | FLAML classifier (metric: average precision, estimators: xgboost + lgbm) |
| **Stage 2** | XGBoost Huber regressor (predicts MFD magnitude, gate ≥ 25%) |
| **Stage 3** | Cross-sectional quantile transform (per-date rank percentiles) |
| **Stage 4** | LambdaMART LTR (objective: rank:ndcg, eval: ndcg@10) |

---

## 4. Daily Picks Output Schema (`daily_picks.csv`)

Generated by the scheduler and Top Recommendations page.

| Column | Type | Description |
|--------|------|-------------|
| `date` | str | Date picks were generated (YYYY-MM-DD) |
| `rank` | int | Rank within top-10 (1 = best) |
| `ticker` | str | Stock ticker symbol |
| `close_price` | float | Closing price at time of prediction |
| `probability` | float | Legacy probability field (alias for ensemble_score) |
| `signal` | str | Signal label (BUY / HOLD) |
| `ensemble_score` | float | Final 4-stage pipeline score |
| `elite_pool_size` | int | Number of stocks passing classifier + Huber gates |
| `cls_proba` | float | Stage 1 classifier probability |
| `pred_mfd` | float | Stage 2 Huber predicted MFD |
| `z_cls` | float | Cross-sectional Z-score of classifier probability |
| `z_ltr` | float | Cross-sectional Z-score of LTR score |
| `ltr_score` | float | Raw LambdaMART score |
| `classification_score` | float | Legacy classification score |
| `volume_surge_3d` | float | 3-day volume surge at prediction time |
| `regime_confidence` | float | Market regime confidence (not currently used) |
| `ticker_calibration` | float | Per-ticker calibration factor (default 1.0) |
| `volatility_20d` | float | 20-day volatility at prediction time |
| `sentiment_score` | float | Social media sentiment polarity |
| `sentiment_mentions` | int | Social media mention count |
| `shap_top_features` | str | Top SHAP features driving the prediction |
| `market_cap` | float | Market capitalization at inference time (display only) |
| `sector` | str | GICS sector from YFinance |
| `max_upside_pct` | float | Ground truth: max upside achieved within 3 months (filled retroactively) |
| `hit_20pct` | bool | Ground truth: whether stock achieved ≥ 20% upside (filled retroactively) |
| `ground_truth_date` | str | Date ground truth was last evaluated |

---

## 5. Scoring Formula

```
score = max(Z_cls, 0) × max(Z_ltr, 0) × min(pool / 75, 2.0)
```

- **Z_cls**: Z-score of classifier probability across elite pool
- **Z_ltr**: Z-score of LTR score across elite pool
- **pool**: Elite pool size (stocks passing both classifier P ≥ 0.50 and Huber MFD ≥ 0.25 gates)
- Picks are only recorded when pool ≥ 75 (MIN_ELITE_POOL)
- Top 10 stocks by score are selected as daily picks
