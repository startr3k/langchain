"""Historical fundamental data from YFinance quarterly filings.

Provides properly time-aligned financial metrics (revenue, EPS, margins,
book value, etc.) from quarterly income statements, balance sheets, and
cash-flow statements.  Each training row gets the fundamentals that were
actually known at that point in time — eliminating the look-ahead bias
that comes from using today's snapshot for historical rows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Features extracted from quarterly filings
HIST_FUNDAMENTAL_FEATURES = [
    # Income statement
    "hist_total_revenue",
    "hist_operating_income",
    "hist_net_income",
    "hist_diluted_eps",
    # Margins (derived)
    "hist_operating_margin",
    "hist_profit_margin",
    # Balance sheet
    "hist_total_assets",
    "hist_total_debt",
    "hist_stockholders_equity",
    "hist_book_value_per_share",
    "hist_current_assets",
    # Ratios (derived)
    "hist_debt_to_equity",
    "hist_current_ratio",
    "hist_roe",
    "hist_roa",
    # Cash flow
    "hist_capex",
    # Growth (QoQ)
    "hist_revenue_growth_qoq",
    "hist_earnings_growth_qoq",
]


def _safe_get(df: pd.DataFrame, row: str, col) -> float:
    """Safely get a value from a DataFrame, returning NaN on failure."""
    try:
        val = df.loc[row, col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        return float(val)
    except (KeyError, TypeError, ValueError):
        return np.nan


def get_historical_fundamentals(ticker: str) -> pd.DataFrame:
    """Fetch quarterly fundamentals and return a time-indexed DataFrame.

    Each row is a quarter-end date with all financial metrics that were
    known at that time.  The DataFrame is sorted oldest-first.

    Returns:
        DataFrame indexed by quarter-end Timestamp with HIST_FUNDAMENTAL_FEATURES
        columns, or empty DataFrame on failure.
    """
    try:
        stock = yf.Ticker(ticker)

        # Combine quarterly + annual data for broader coverage
        # YFinance quarterly only provides ~5 recent quarters;
        # annual provides ~5 years with one data point per year.
        inc_q = stock.quarterly_financials
        bs_q = stock.quarterly_balance_sheet
        cf_q = stock.quarterly_cashflow

        inc_a = stock.financials
        bs_a = stock.balance_sheet
        cf_a = stock.cashflow

        def _merge_frames(quarterly, annual):
            """Merge quarterly and annual, preferring quarterly when dates overlap."""
            if quarterly is None or quarterly.empty:
                return annual if annual is not None and not annual.empty else pd.DataFrame()
            if annual is None or annual.empty:
                return quarterly
            # Use quarterly dates + any annual dates not in quarterly
            q_dates = set(quarterly.columns)
            extra_annual_dates = [d for d in annual.columns if d not in q_dates]
            if extra_annual_dates:
                return pd.concat([quarterly, annual[extra_annual_dates]], axis=1)
            return quarterly

        inc = _merge_frames(inc_q, inc_a)
        bs = _merge_frames(bs_q, bs_a)
        cf = _merge_frames(cf_q, cf_a)

        if inc.empty:
            logger.debug("No financials data for %s", ticker)
            return pd.DataFrame()

        # Columns are Timestamps of period-end dates
        quarters = sorted(inc.columns)

        records: list[dict] = []
        prev_revenue = np.nan
        prev_net_income = np.nan

        for q in quarters:
            rec: dict = {"_quarter_date": q}

            # Income statement
            revenue = _safe_get(inc, "Total Revenue", q)
            gross_profit = _safe_get(inc, "Gross Profit", q)
            op_income = _safe_get(inc, "Operating Income", q)
            net_income = _safe_get(inc, "Net Income", q)
            diluted_eps = _safe_get(inc, "Diluted EPS", q)
            ebitda = _safe_get(inc, "EBITDA", q)
            if np.isnan(ebitda):
                ebitda = _safe_get(inc, "Normalized EBITDA", q)

            rec["hist_total_revenue"] = revenue
            rec["hist_gross_profit"] = gross_profit
            rec["hist_operating_income"] = op_income
            rec["hist_net_income"] = net_income
            rec["hist_diluted_eps"] = diluted_eps
            rec["hist_ebitda"] = ebitda

            # Margins
            if not np.isnan(revenue) and revenue != 0:
                rec["hist_gross_margin"] = gross_profit / revenue if not np.isnan(gross_profit) else np.nan
                rec["hist_operating_margin"] = op_income / revenue if not np.isnan(op_income) else np.nan
                rec["hist_profit_margin"] = net_income / revenue if not np.isnan(net_income) else np.nan
                rec["hist_ebitda_margin"] = ebitda / revenue if not np.isnan(ebitda) else np.nan
            else:
                rec["hist_gross_margin"] = np.nan
                rec["hist_operating_margin"] = np.nan
                rec["hist_profit_margin"] = np.nan
                rec["hist_ebitda_margin"] = np.nan

            # Balance sheet
            if bs is not None and not bs.empty and q in bs.columns:
                total_assets = _safe_get(bs, "Total Assets", q)
                total_debt = _safe_get(bs, "Total Debt", q)
                equity = _safe_get(bs, "Stockholders Equity", q)
                shares = _safe_get(bs, "Ordinary Shares Number", q)
                cur_assets = _safe_get(bs, "Current Assets", q)
                cur_liab = _safe_get(bs, "Current Liabilities", q)

                rec["hist_total_assets"] = total_assets
                rec["hist_total_debt"] = total_debt
                rec["hist_stockholders_equity"] = equity
                rec["hist_current_assets"] = cur_assets
                rec["hist_current_liabilities"] = cur_liab

                # Book value per share
                if not np.isnan(equity) and not np.isnan(shares) and shares > 0:
                    rec["hist_book_value_per_share"] = equity / shares
                else:
                    rec["hist_book_value_per_share"] = np.nan

                # Ratios
                if not np.isnan(total_debt) and not np.isnan(equity) and equity != 0:
                    rec["hist_debt_to_equity"] = total_debt / equity
                else:
                    rec["hist_debt_to_equity"] = np.nan

                if not np.isnan(cur_assets) and not np.isnan(cur_liab) and cur_liab != 0:
                    rec["hist_current_ratio"] = cur_assets / cur_liab
                else:
                    rec["hist_current_ratio"] = np.nan

                # ROE / ROA (annualized from quarterly net income)
                ann_income = net_income * 4 if not np.isnan(net_income) else np.nan
                if not np.isnan(ann_income) and not np.isnan(equity) and equity != 0:
                    rec["hist_roe"] = ann_income / equity
                else:
                    rec["hist_roe"] = np.nan
                if not np.isnan(ann_income) and not np.isnan(total_assets) and total_assets != 0:
                    rec["hist_roa"] = ann_income / total_assets
                else:
                    rec["hist_roa"] = np.nan
            else:
                for col in [
                    "hist_total_assets", "hist_total_debt",
                    "hist_stockholders_equity", "hist_book_value_per_share",
                    "hist_current_assets", "hist_current_liabilities",
                    "hist_debt_to_equity", "hist_current_ratio",
                    "hist_roe", "hist_roa",
                ]:
                    rec[col] = np.nan

            # Cash flow
            if cf is not None and not cf.empty and q in cf.columns:
                rec["hist_free_cash_flow"] = _safe_get(cf, "Free Cash Flow", q)
                rec["hist_capex"] = _safe_get(cf, "Capital Expenditure", q)
            else:
                rec["hist_free_cash_flow"] = np.nan
                rec["hist_capex"] = np.nan

            # QoQ growth
            if not np.isnan(revenue) and not np.isnan(prev_revenue) and prev_revenue != 0:
                rec["hist_revenue_growth_qoq"] = (revenue - prev_revenue) / abs(prev_revenue)
            else:
                rec["hist_revenue_growth_qoq"] = np.nan

            if not np.isnan(net_income) and not np.isnan(prev_net_income) and prev_net_income != 0:
                rec["hist_earnings_growth_qoq"] = (net_income - prev_net_income) / abs(prev_net_income)
            else:
                rec["hist_earnings_growth_qoq"] = np.nan

            prev_revenue = revenue
            prev_net_income = net_income

            records.append(rec)

        result = pd.DataFrame(records)
        result["_quarter_date"] = pd.to_datetime(result["_quarter_date"])
        result = result.sort_values("_quarter_date").reset_index(drop=True)
        return result

    except Exception:
        logger.exception("Error fetching historical fundamentals for %s", ticker)
        return pd.DataFrame()


def align_fundamentals_to_dates(
    fundamentals_df: pd.DataFrame,
    dates: pd.DatetimeIndex | pd.Index,
) -> pd.DataFrame:
    """Map each date to the most recent quarter's fundamentals.

    For each date, we find the latest quarter-end that is on or before
    that date.  This ensures no future data leakage — a row from 2023-05
    gets Q1-2023 fundamentals, not Q2-2023 (which hasn't been reported yet).

    Args:
        fundamentals_df: Output of get_historical_fundamentals().
        dates: The dates of the training rows to align to.

    Returns:
        DataFrame with same length as dates, containing fundamental columns.
    """
    if fundamentals_df.empty:
        return pd.DataFrame(
            {col: np.nan for col in HIST_FUNDAMENTAL_FEATURES},
            index=range(len(dates)),
        )

    quarter_dates = fundamentals_df["_quarter_date"].values
    feature_cols = [c for c in HIST_FUNDAMENTAL_FEATURES if c in fundamentals_df.columns]

    aligned_rows: list[dict] = []
    for d in dates:
        ts = pd.Timestamp(d)
        # Find most recent quarter on or before this date
        mask = quarter_dates <= ts
        if mask.any():
            idx = np.where(mask)[0][-1]
            row = fundamentals_df.iloc[idx]
            aligned_rows.append({col: row.get(col, np.nan) for col in feature_cols})
        else:
            aligned_rows.append({col: np.nan for col in feature_cols})

    return pd.DataFrame(aligned_rows)
