"""Random Forest Classifier Re-training on Real External APISentry Dataset.

Performs:
1. Feature extraction via `data.external.adapter.adapt_external_row` to 12 canonical features.
2. Leakage-free 70/15/15 stratified train/val/test splitting with persistent indices.
3. Sklearn Pipeline wrapping StandardScaler and RandomForestClassifier.
4. 5-fold Stratified CV on training set only using RandomizedSearchCV (30 iterations).
5. Evaluation on validation set and strict one-touch evaluation on test set.
6. Persistence of model pipeline, split indices, and JSON evaluation metrics.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Suppress harmless sklearn nested joblib parallel warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.external.adapter import adapt_external_row
from ml.config import FEATURE_NAMES

DATASET_PATH = Path("data/external/apisentry/dataset_processed.csv")
ARTIFACTS_DIR = Path("ml/artifacts")
SPLIT_INDICES_PATH = ARTIFACTS_DIR / "rf_external_split_indices.json"
MODEL_PATH = ARTIFACTS_DIR / "rf_external.joblib"
METRICS_PATH = ARTIFACTS_DIR / "rf_external_metrics.json"


def load_and_adapt_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load APISentry CSV and adapt rows to the canonical 12-dimensional feature space."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"APISentry dataset not found at {DATASET_PATH}")

    raw_df = pd.read_csv(DATASET_PATH)
    records = raw_df.to_dict(orient="records")

    adapted_features: List[Dict[str, float]] = [adapt_external_row(r) for r in records]
    X_df = pd.DataFrame(adapted_features)[FEATURE_NAMES]

    # Validate assertions
    assert X_df.shape[1] == 12, f"Expected exactly 12 features, got {X_df.shape[1]}"
    assert not X_df.isna().any().any(), "Feature matrix contains NaN values"
    assert not np.isinf(X_df.to_numpy()).any(), "Feature matrix contains infinite values"

    y_multiclass = raw_df["label"].copy()
    y_binary = (y_multiclass != "normal").astype(int)

    return X_df, y_multiclass, y_binary


def split_data(
    X_df: pd.DataFrame, y_multiclass: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a stratified 70/15/15 train/val/test split and persist indices."""
    n_samples = len(X_df)
    indices = np.arange(n_samples)

    # 70% train (7,917), 30% temp (3,393)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=y_multiclass,
        random_state=42,
    )

    # Split 30% temp evenly into 15% val (1,696) and 15% test (1,697)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=y_multiclass.iloc[temp_idx],
        random_state=42,
    )

    # Ensure mutual exclusivity and complete coverage
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    assert len(train_set.intersection(val_set)) == 0, "Train and Val overlap"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test overlap"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test overlap"
    assert len(train_set | val_set | test_set) == n_samples, "Indices do not cover all rows"

    # Persist split indices for subsequent models (IF, SHAP)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "train": [int(i) for i in train_idx],
        "val": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
    }
    with open(SPLIT_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    return train_idx, val_idx, test_idx


def train_and_evaluate() -> Dict[str, Any]:
    """Train Random Forest with 5-fold CV hyperparameter tuning and evaluate."""
    X_df, y_multiclass, y_binary = load_and_adapt_dataset()
    train_idx, val_idx, test_idx = split_data(X_df, y_multiclass)

    X_train, y_train = X_df.iloc[train_idx], y_multiclass.iloc[train_idx]
    X_val, y_val = X_df.iloc[val_idx], y_multiclass.iloc[val_idx]
    X_test, y_test = X_df.iloc[test_idx], y_multiclass.iloc[test_idx]
    y_test_binary = y_binary.iloc[test_idx]

    # Pipeline with StandardScaler and RandomForestClassifier
    pipeline = Pipeline(
        [
            ("standardscaler", StandardScaler()),
            ("randomforestclassifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    param_distributions = {
        "randomforestclassifier__n_estimators": [100, 200, 300],
        "randomforestclassifier__max_depth": [8, 12, 16, None],
        "randomforestclassifier__min_samples_split": [2, 5, 10],
        "randomforestclassifier__min_samples_leaf": [1, 2, 4],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_idx = search.best_index_
    cv_macro_f1_mean = float(search.cv_results_["mean_test_score"][best_idx])
    cv_macro_f1_std = float(search.cv_results_["std_test_score"][best_idx])

    # 1. Validation Set Metrics (Multiclass)
    y_val_pred = best_pipeline.predict(X_val)
    val_macro_f1 = float(f1_score(y_val, y_val_pred, average="macro"))
    val_accuracy = float(accuracy_score(y_val, y_val_pred))

    # 2. Test Set Evaluation (Touch test set ONCE)
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)

    test_macro_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
    test_weighted_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))
    test_accuracy = float(accuracy_score(y_test, y_test_pred))

    raw_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    per_class_metrics: Dict[str, Any] = {}
    for cls_name, vals in raw_report.items():
        if isinstance(vals, dict):
            per_class_metrics[cls_name] = {
                "precision": float(vals.get("precision", 0.0)),
                "recall": float(vals.get("recall", 0.0)),
                "f1_score": float(vals.get("f1-score", 0.0)),
                "support": int(vals.get("support", 0)),
            }

    conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()

    # 3. Test Binary Metrics (Attack vs Normal)
    y_test_pred_binary = (y_test_pred != "normal").astype(int)
    bin_precision = float(precision_score(y_test_binary, y_test_pred_binary, zero_division=0))
    bin_recall = float(recall_score(y_test_binary, y_test_pred_binary, zero_division=0))
    bin_f1 = float(f1_score(y_test_binary, y_test_pred_binary, zero_division=0))

    # Binary ROC-AUC: Probability of attack is (1.0 - P(normal))
    classes_list = list(best_pipeline.named_steps["randomforestclassifier"].classes_)
    if "normal" in classes_list:
        normal_idx = classes_list.index("normal")
        attack_proba = 1.0 - y_test_proba[:, normal_idx]
    else:
        attack_proba = y_test_pred_binary.astype(float)
    bin_roc_auc = float(roc_auc_score(y_test_binary, attack_proba))

    # Save Pipeline and Metrics
    joblib.dump(best_pipeline, MODEL_PATH)

    metrics_payload = {
        "dataset": "APISentry (Zenodo 22658802)",
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "best_params": {str(k): v for k, v in search.best_params_.items()},
        "cv_macro_f1_mean": round(cv_macro_f1_mean, 6),
        "cv_macro_f1_std": round(cv_macro_f1_std, 6),
        "val_macro_f1": round(val_macro_f1, 6),
        "val_accuracy": round(val_accuracy, 6),
        "test_multiclass": {
            "macro_f1": round(test_macro_f1, 6),
            "weighted_f1": round(test_weighted_f1, 6),
            "accuracy": round(test_accuracy, 6),
            "per_class": per_class_metrics,
        },
        "test_binary": {
            "precision": round(bin_precision, 6),
            "recall": round(bin_recall, 6),
            "f1": round(bin_f1, 6),
            "roc_auc": round(bin_roc_auc, 6),
        },
        "confusion_matrix": conf_matrix,
        "random_state": 42,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Print Clean Console Summary
    print("=" * 60)
    print("  RF EXTERNAL — APISentry Re-training")
    print("=" * 60)
    print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}  |  Test: {len(test_idx):,}")
    print(f"  CV folds: 5 (macro-F1 = {cv_macro_f1_mean:.4f} ± {cv_macro_f1_std:.4f})")
    print(f"  Best params: {search.best_params_}")
    print("  " + "─" * 45)
    print("  Validation (multiclass):")
    print(f"    macro-F1   : {val_macro_f1:.4f}")
    print(f"    accuracy   : {val_accuracy:.4f}")
    print("  Test (multiclass):")
    print(f"    macro-F1   : {test_macro_f1:.4f}")
    print(f"    accuracy   : {test_accuracy:.4f}")
    print("  Test (binary attack/normal):")
    print(f"    precision  : {bin_precision:.4f}")
    print(f"    recall     : {bin_recall:.4f}")
    print(f"    F1         : {bin_f1:.4f}")
    print(f"    ROC-AUC    : {bin_roc_auc:.4f}")

    if cv_macro_f1_std > 0.10:
        print("  WARNING: Model unstable across folds (CV macro-F1 std > 0.10)")

    if test_macro_f1 > 0.98 and cv_macro_f1_mean > 0.98 and cv_macro_f1_std < 0.01:
        print(
            "  WARNING: Suspiciously high scores — consider investigating feature overlap "
            "or class separability. This is common when attacks are trivially distinct."
        )

    print("  " + "─" * 45)
    print("  Artifacts saved:")
    print(f"    {MODEL_PATH}")
    print(f"    {METRICS_PATH}")
    print(f"    {SPLIT_INDICES_PATH}")
    print("=" * 60)

    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()

