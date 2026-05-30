"""Compare model accuracy across different prediction horizons."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent))

from stock_predictor.data.feature_engineering import (
    ALL_FEATURE_NAMES,
    FUNDAMENTAL_FEATURES,
    SENTIMENT_FEATURES,
    TECHNICAL_FEATURES,
)
from stock_predictor.data.yfinance_client import (
    NASDAQ_TOP_TICKERS,
    compute_technical_features,
    get_fundamentals_features,
    get_stock_data,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HORIZONS = {
    "1 Week (5d)": 5,
    "2 Weeks (10d)": 10,
    "1 Month (21d)": 21,
    "3 Months (63d)": 63,
    "6 Months (126d)": 126,
}

TICKERS = NASDAQ_TOP_TICKERS[:15]
TIME_BUDGET = 60  # seconds per model


def build_dataset_all_horizons(tickers: list[str]) -> pd.DataFrame:
    """Build dataset once with forward returns for all horizons."""
    all_rows = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker, period="5y")
            if df.empty or len(df) < 300:
                logger.warning("Skipping %s — insufficient history", ticker)
                continue

            df = compute_technical_features(df)

            # Compute forward returns for all horizons
            for label, days in HORIZONS.items():
                col = f"Forward_{days}d"
                df[col] = df["Close"].shift(-days) / df["Close"] - 1

            # Fundamentals
            fundamentals = get_fundamentals_features(ticker)

            # Valid rows: need SMA_200 and all horizons to have values
            horizon_cols = [f"Forward_{d}d" for d in HORIZONS.values()]
            valid_mask = df["SMA_200"].notna()
            for hcol in horizon_cols:
                valid_mask = valid_mask & df[hcol].notna()
            valid_df = df[valid_mask]

            if valid_df.empty:
                continue

            # Sample up to 200 points
            sample_indices = np.linspace(
                0, len(valid_df) - 1, min(200, len(valid_df)), dtype=int
            )
            sampled = valid_df.iloc[sample_indices]

            for _, row in sampled.iterrows():
                data_point = {"Ticker": ticker}
                for col in TECHNICAL_FEATURES:
                    data_point[col] = row.get(col, np.nan)
                for col in FUNDAMENTAL_FEATURES:
                    data_point[col] = fundamentals.get(col, np.nan)
                for hcol in horizon_cols:
                    data_point[hcol] = row[hcol]
                all_rows.append(data_point)

            logger.info("Processed %s: %d samples", ticker, len(sampled))
        except Exception:
            logger.exception("Error processing %s", ticker)

    return pd.DataFrame(all_rows)


def compute_mape(y_true, y_pred):
    """Compute MAPE, excluding near-zero actuals."""
    mask = np.abs(y_true) > 0.001  # avoid division by near-zero
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_and_evaluate(df: pd.DataFrame, target_col: str, time_budget: int) -> dict:
    """Train a model for a specific horizon and return metrics."""
    feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    valid = y.notna()
    X = X[valid]
    y = y[valid]

    medians = X.median()
    X = X.fillna(medians)

    automl = AutoML()
    automl.fit(
        X_train=X,
        y_train=y,
        task="regression",
        time_budget=time_budget,
        metric="r2",
        estimator_list=["xgboost", "lgbm", "rf", "extra_tree"],
        eval_method="cv",
        n_splits=5,
        verbose=0,
    )

    # Training set metrics
    y_pred_train = automl.predict(X)
    r2_train = r2_score(y, y_pred_train)
    mae_train = mean_absolute_error(y, y_pred_train)
    rmse_train = float(np.sqrt(mean_squared_error(y, y_pred_train)))
    mape_train = compute_mape(y.values, y_pred_train)

    # Cross-val R² (best_loss is 1 - R² for r2 metric)
    cv_r2 = 1 - automl.best_loss if automl.best_loss is not None else 0

    # Manual hold-out evaluation (last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    y_pred_test = automl.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mape_test = compute_mape(y_test.values, y_pred_test)

    return {
        "r2_train": round(r2_train, 4),
        "r2_cv": round(cv_r2, 4),
        "r2_test": round(r2_test, 4),
        "mae_train": round(mae_train, 4),
        "mae_test": round(mae_test, 4),
        "rmse_train": round(rmse_train, 4),
        "rmse_test": round(rmse_test, 4),
        "mape_train": round(mape_train, 1),
        "mape_test": round(mape_test, 1),
        "best_estimator": automl.best_estimator,
        "samples": len(X),
    }


def main():
    print(f"\n{'='*80}")
    print("MULTI-HORIZON MODEL ACCURACY COMPARISON")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Time budget per model: {TIME_BUDGET}s")
    print(f"{'='*80}\n")

    # Build dataset once
    print("Building dataset with all forward return horizons...")
    df = build_dataset_all_horizons(TICKERS)
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns\n")

    results = {}
    for label, days in HORIZONS.items():
        target_col = f"Forward_{days}d"
        print(f"\n--- Training model for {label} ---")
        metrics = train_and_evaluate(df, target_col, TIME_BUDGET)
        results[label] = metrics
        print(f"  R² (train/CV/test): {metrics['r2_train']} / {metrics['r2_cv']} / {metrics['r2_test']}")
        print(f"  MAE (train/test):   {metrics['mae_train']} / {metrics['mae_test']}")
        print(f"  RMSE (train/test):  {metrics['rmse_train']} / {metrics['rmse_test']}")
        print(f"  MAPE (train/test):  {metrics['mape_train']}% / {metrics['mape_test']}%")
        print(f"  Model:              {metrics['best_estimator']}")

    # Summary table
    print(f"\n\n{'='*90}")
    print("COMPARISON SUMMARY (Test Set = last 20% of data)")
    print(f"{'='*90}")
    print(f"{'Horizon':<20} {'R²(Test)':>9} {'MAE(Test)':>10} {'RMSE(Test)':>11} {'MAPE(Test)':>11} {'R²(CV)':>8} {'Model':>12}")
    print("-" * 90)
    for label, m in results.items():
        print(
            f"{label:<20} {m['r2_test']:>9.4f} {m['mae_test']:>10.4f} "
            f"{m['rmse_test']:>11.4f} {m['mape_test']:>10.1f}% "
            f"{m['r2_cv']:>8.4f} {m['best_estimator']:>12}"
        )

    # Find best by test R²
    best_r2 = max(results.items(), key=lambda x: x[1]["r2_test"])
    best_mape = min(results.items(), key=lambda x: x[1]["mape_test"] if not np.isnan(x[1]["mape_test"]) else 999)
    print(f"\n{'='*90}")
    print(f"BEST HORIZON BY TEST R²:   {best_r2[0]} (R²={best_r2[1]['r2_test']:.4f})")
    print(f"BEST HORIZON BY TEST MAPE: {best_mape[0]} (MAPE={best_mape[1]['mape_test']:.1f}%)")
    print(f"{'='*90}")

    print("\nNote: MAPE = Mean Absolute Percentage Error (lower is better)")
    print("      R² = Coefficient of determination (higher is better, max 1.0)")
    print("      Negative R² means model is worse than predicting the mean")


if __name__ == "__main__":
    main()
