"""Run full training on the 10-year dataset."""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from stock_predictor.models.automl_model import StockReturnPredictor

DATA_PATH = "training_data_10y_full.csv"
print("Loading training data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Rows: {len(df):,}, Cols: {df.shape[1]}")

predictor = StockReturnPredictor()
print("\nStarting training (time_budget=120s)...")
metrics = predictor.train(df=df, time_budget=120)

print("\n=== TRAINING RESULTS ===")
print(f"Best estimator: {metrics.get('best_estimator')}")
print(f"AUC-ROC: train={metrics.get('auc_train', 0):.4f}, test={metrics.get('auc_roc', 0):.4f}")
print(f"Avg Precision: train={metrics.get('ap_train', 0):.4f}, test={metrics.get('avg_precision', 0):.4f}")

ltr = metrics.get("ltr", {})
print(f"\nLTR: status={ltr.get('status')}")
if ltr.get("status") == "trained":
    print(f"  NDCG@10: train={ltr.get('ndcg10_train')}, test={ltr.get('ndcg10_test')}, gap={ltr.get('ndcg10_gap')}")

regime = metrics.get("regime", {})
print(f"\nRegime: status={regime.get('status')}")
if regime.get("status") == "trained":
    print(f"  R²: train={regime.get('r2_train')}, test={regime.get('r2_test')}, gap={regime.get('r2_gap')}")
elif regime.get("status") == "discarded":
    print(f"  Discarded: R² test={regime.get('r2_test')} (negative, overfitting)")

cal = metrics.get("ticker_calibration", {})
print(f"\nTicker Calibration: status={cal.get('status')}")
if cal.get("status") == "trained":
    print(f"  Tracked: {cal.get('tickers_tracked')}, Penalized: {cal.get('tickers_penalized')}")

top_n = metrics.get("top_n", {})
for k in ("top_10", "top_20", "top_50"):
    t = top_n.get(k, {})
    if t:
        print(f"\n{k}: {t.get('hits')}/{t.get('total')} ({t.get('hit_rate', 0):.0%}), avg return={t.get('avg_peak_return', 0):.1%}")

# Check saved files
from pathlib import Path
model_dir = Path("stock_predictor/models/saved")
for f in sorted(model_dir.iterdir()):
    print(f"  Saved: {f.name} ({f.stat().st_size / 1024:.0f} KB)")

print("\n=== DONE ===")
