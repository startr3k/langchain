"""Multicollinearity analysis and grouped permutation importance.

Computes:
1. Correlation matrix for all model features
2. Identifies correlated feature groups (|r| > threshold)
3. Runs grouped permutation importance to measure the true predictive
   contribution of each feature *group*, avoiding the split-dilution
   problem that affects tree-based feature importances.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = "training_data_10y_full.csv"
MODEL_PATH = "stock_predictor/models/saved/stock_predictor_model.pkl"
FEATURE_NAMES_PATH = "stock_predictor/models/saved/feature_names.pkl"
CORR_THRESHOLD = 0.70


def load_data():
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    automl = joblib.load(MODEL_PATH)

    available = [f for f in feature_names if f in df.columns]
    logger.info(f"Features: {len(available)}/{len(feature_names)} available in dataset")

    df = df.sort_values("_date").reset_index(drop=True)
    n = len(df)
    gap = int(n * 0.05)
    test_start = int(n * 0.75) + gap
    test_df = df.iloc[test_start:].copy()
    logger.info(f"Test set: {len(test_df)} rows")

    X_test = test_df[available].copy()
    y_test = (test_df["Forward_Max_Return_3M"] >= 0.20).astype(int)

    return X_test, y_test, automl, available


def compute_correlation_matrix(X: pd.DataFrame):
    """Compute pairwise Spearman correlation (robust to non-linearity)."""
    logger.info("\n=== CORRELATION MATRIX (Spearman) ===")
    # Fill NaN with median for correlation computation
    X_filled = X.fillna(X.median())
    corr = X_filled.corr(method="spearman")
    return corr


def find_correlated_groups(corr: pd.DataFrame, threshold: float = CORR_THRESHOLD):
    """Find groups of features with |correlation| > threshold using union-find."""
    features = list(corr.columns)
    parent = {f: f for f in features}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Find all highly correlated pairs
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                high_corr_pairs.append((features[i], features[j], r))
                union(features[i], features[j])

    # Build groups
    groups = defaultdict(list)
    for f in features:
        groups[find(f)].append(f)

    # Separate into correlated groups and singletons
    correlated_groups = {k: v for k, v in groups.items() if len(v) > 1}
    singletons = [v[0] for v in groups.values() if len(v) == 1]

    return correlated_groups, singletons, high_corr_pairs


def grouped_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    groups: dict[str, list[str]],
    singletons: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
):
    """Compute permutation importance, shuffling correlated groups together.

    For correlated groups, all features in the group are shuffled
    simultaneously so we measure the true contribution of the entire
    *concept*, not just one feature from the group.
    """
    rng = np.random.RandomState(random_state)
    X_filled = X.fillna(X.median())

    # Baseline AUC
    try:
        base_proba = model.predict_proba(X_filled)[:, 1]
    except AttributeError:
        base_proba = model.predict(X_filled)
    base_auc = roc_auc_score(y, base_proba)
    logger.info(f"\nBaseline test AUC: {base_auc:.4f}")

    results = []

    # Grouped features
    for group_name, group_features in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X_filled.copy()
            perm_idx = rng.permutation(len(X_perm))
            for feat in group_features:
                X_perm[feat] = X_perm[feat].values[perm_idx]
            try:
                perm_proba = model.predict_proba(X_perm)[:, 1]
            except AttributeError:
                perm_proba = model.predict(X_perm)
            perm_auc = roc_auc_score(y, perm_proba)
            drops.append(base_auc - perm_auc)

        results.append({
            "name": " + ".join(group_features),
            "type": "group",
            "features": group_features,
            "importance_mean": np.mean(drops),
            "importance_std": np.std(drops),
            "n_features": len(group_features),
        })

    # Singleton features
    for feat in singletons:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_filled.copy()
            X_perm[feat] = X_perm[feat].values[rng.permutation(len(X_perm))]
            try:
                perm_proba = model.predict_proba(X_perm)[:, 1]
            except AttributeError:
                perm_proba = model.predict(X_perm)
            perm_auc = roc_auc_score(y, perm_proba)
            drops.append(base_auc - perm_auc)

        results.append({
            "name": feat,
            "type": "single",
            "features": [feat],
            "importance_mean": np.mean(drops),
            "importance_std": np.std(drops),
            "n_features": 1,
        })

    results.sort(key=lambda x: x["importance_mean"], reverse=True)
    return results, base_auc


def main():
    X_test, y_test, automl, feature_names = load_data()

    # --- Step 1: Correlation matrix ---
    corr = compute_correlation_matrix(X_test)

    # --- Step 2: Find correlated groups ---
    correlated_groups, singletons, high_corr_pairs = find_correlated_groups(corr)

    logger.info(f"\n=== HIGHLY CORRELATED PAIRS (|r| > {CORR_THRESHOLD}) ===")
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    logger.info(f"{'Feature A':<35} {'Feature B':<35} {'Spearman r':>10}")
    logger.info("-" * 82)
    for a, b, r in high_corr_pairs:
        logger.info(f"{a:<35} {b:<35} {r:>10.3f}")

    logger.info(f"\n=== CORRELATED FEATURE GROUPS (|r| > {CORR_THRESHOLD}) ===")
    group_labels = {}
    for i, (_, members) in enumerate(correlated_groups.items(), 1):
        label = f"Group {i}"
        logger.info(f"\n{label}: {members}")
        # Show within-group correlations
        for mi in range(len(members)):
            for mj in range(mi + 1, len(members)):
                r = corr.loc[members[mi], members[mj]]
                logger.info(f"  {members[mi]} <-> {members[mj]}: r = {r:.3f}")
        for m in members:
            group_labels[m] = label

    logger.info(f"\nSingleton features (not correlated with any other): {len(singletons)}")

    # --- Step 3: Grouped permutation importance ---
    logger.info("\n=== GROUPED PERMUTATION IMPORTANCE ===")
    logger.info("(Correlated features shuffled together to measure concept-level importance)")
    t0 = time.time()
    results, base_auc = grouped_permutation_importance(
        X_test, y_test, automl, correlated_groups, singletons, n_repeats=5,
    )
    elapsed = time.time() - t0
    logger.info(f"Completed in {elapsed:.1f}s")

    # --- Step 4: Display results ---
    logger.info(f"\n{'Rank':<6} {'Feature/Group':<65} {'Type':<8} {'AUC Drop':>10} {'± Std':>8}")
    logger.info("=" * 100)
    for i, r in enumerate(results, 1):
        name = r["name"]
        if len(name) > 62:
            name = name[:59] + "..."
        logger.info(
            f"{i:<6} {name:<65} {r['type']:<8} {r['importance_mean']:>10.5f} {r['importance_std']:>8.5f}"
        )

    # --- Step 5: Compare with split-based importance ---
    logger.info("\n=== COMPARISON: Split-Based vs Grouped Permutation ===")
    model_inner = automl.model.estimator
    if hasattr(model_inner, "feature_importances_"):
        importances = model_inner.feature_importances_
        if hasattr(model_inner, "feature_name_"):
            names = model_inner.feature_name_
        elif hasattr(model_inner, "feature_names_in_"):
            names = list(model_inner.feature_names_in_)
        else:
            names = feature_names
        split_imp = dict(zip(names, importances))

        logger.info(f"\n{'Feature':<35} {'Split Imp':>10} {'Group':>10} {'Perm AUC Drop':>15}")
        logger.info("-" * 72)

        # Map each feature to its permutation result
        feat_to_perm = {}
        for r in results:
            for f in r["features"]:
                feat_to_perm[f] = r

        for feat in sorted(split_imp, key=split_imp.get, reverse=True):
            si = split_imp[feat]
            gl = group_labels.get(feat, "singleton")
            perm_r = feat_to_perm.get(feat)
            perm_val = perm_r["importance_mean"] if perm_r else 0
            logger.info(f"{feat:<35} {si:>10} {gl:>10} {perm_val:>15.5f}")

    # --- Summary ---
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total features: {len(feature_names)}")
    logger.info(f"Correlated groups: {len(correlated_groups)}")
    logger.info(f"Features in groups: {sum(len(v) for v in correlated_groups.values())}")
    logger.info(f"Singleton features: {len(singletons)}")
    logger.info(f"High correlation pairs (|r| > {CORR_THRESHOLD}): {len(high_corr_pairs)}")
    logger.info(f"Baseline test AUC: {base_auc:.4f}")

    # Highlight discrepancies
    logger.info("\n=== KEY DISCREPANCIES (split importance misleading) ===")
    for r in results:
        if r["type"] == "group" and r["importance_mean"] > 0.001:
            # Check if any member has zero split importance
            zero_split = [f for f in r["features"] if split_imp.get(f, 0) == 0]
            if zero_split:
                logger.info(
                    f"  {r['name']}: group AUC drop = {r['importance_mean']:.5f}, "
                    f"but {zero_split} have ZERO split importance"
                )


if __name__ == "__main__":
    main()
