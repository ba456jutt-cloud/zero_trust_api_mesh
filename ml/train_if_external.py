"""Isolation Forest Re-training on External APISentry Dataset with Percentile Calibration.

Re-trains Isolation Forest using:
1. Exact same train/val/test split indices as `rf_external.joblib`.
2. Normal training samples only (`label_binary == 0`, ~4,628 samples).
3. Reusable PercentileCalibrator (QuantileTransformer) to eliminate hardcoded score offsets.
4. Data-driven APPROVE threshold at 95th percentile of normal traffic.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from data.external.adapter import adapt_external_row
from ml.calibration import PercentileCalibrator
from ml.config import FEATURE_NAMES

DATASET_PATH = Path("data/external/apisentry/dataset_processed.csv")
ARTIFACTS_DIR = Path("ml/artifacts")
SPLIT_INDICES_PATH = ARTIFACTS_DIR / "rf_external_split_indices.json"
MODEL_PATH = ARTIFACTS_DIR / "if_external.joblib"
CALIBRATOR_PATH = ARTIFACTS_DIR / "if_transformer_external.joblib"
METRICS_PATH = ARTIFACTS_DIR / "if_external_metrics.json"


def train_and_evaluate() -> Dict[str, Any]:
    """Train Isolation Forest on normal samples with percentile calibration."""
    if not SPLIT_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"Split indices not found at {SPLIT_INDICES_PATH}. Run ml.train_rf_external first."
        )

    with open(SPLIT_INDICES_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # 1. Load and adapt APISentry dataset
    raw_df = pd.read_csv(DATASET_PATH)
    records = raw_df.to_dict(orient="records")
    adapted_features = [adapt_external_row(r) for r in records]
    X_df = pd.DataFrame(adapted_features)[FEATURE_NAMES]
    y_multiclass = raw_df["label"].copy()
    y_binary = raw_df["label_binary"].copy()

    # 2. Filter training set to normal samples only (label_binary == 0)
    train_normal_mask = (y_binary.iloc[train_idx] == 0)
    X_train_normal = X_df.iloc[train_idx][train_normal_mask]
    n_train_normal = len(X_train_normal)

    # 3. Fit Isolation Forest on normal training data
    contamination = 0.01
    n_estimators = 200
    if_model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    if_model.fit(X_train_normal)

    # 4. Compute raw scores on training normals (higher = more anomalous)
    raw_train_normal = -1.0 * if_model.decision_function(X_train_normal)

    # 5. Fit PercentileCalibrator on normal raw scores
    calibrator = PercentileCalibrator(random_state=42)
    calibrator.fit(raw_train_normal)

    # 6. Compute calibrated scores on training normals
    cal_train_normal = calibrator.transform(raw_train_normal)
    mean_cal_normal = float(np.mean(cal_train_normal))
    std_cal_normal = float(np.std(cal_train_normal))
    p50_cal_normal = float(np.percentile(cal_train_normal, 50.0))
    p95_cal_normal = float(calibrator.approve_threshold(95.0))
    p99_cal_normal = float(np.percentile(cal_train_normal, 99.0))

    # 7. Evaluate on Test Set
    X_test = X_df.iloc[test_idx]
    y_test_binary = y_binary.iloc[test_idx]
    y_test_multiclass = y_multiclass.iloc[test_idx]

    raw_test = -1.0 * if_model.decision_function(X_test)
    cal_test = calibrator.transform(raw_test)

    # Classification at p95 threshold: score >= p95 -> attack (1), else normal (0)
    y_pred_binary = (cal_test >= p95_cal_normal).astype(int)

    test_precision = float(precision_score(y_test_binary, y_pred_binary, zero_division=0))
    test_recall = float(recall_score(y_test_binary, y_pred_binary, zero_division=0))
    test_f1 = float(f1_score(y_test_binary, y_pred_binary, zero_division=0))
    test_roc_auc = float(roc_auc_score(y_test_binary, cal_test))

    # Per-class mean calibrated scores
    test_df = pd.DataFrame({"label": y_test_multiclass, "cal_score": cal_test})
    per_class_series = test_df.groupby("label")["cal_score"].mean()
    per_class_scores: Dict[str, float] = {
        cls_name: round(float(val), 6) for cls_name, val in per_class_series.items()
    }

    # Safety check on calibration integrity
    normal_mean = per_class_scores.get("normal", 0.0)
    if normal_mean > 0.60:
        raise ValueError(
            f"Per-class mean score for 'normal' is {normal_mean:.4f} (> 0.60). Calibration compromised."
        )

    # 8. Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(if_model, MODEL_PATH)
    calibrator.save(CALIBRATOR_PATH)

    metrics_payload = {
        "dataset": "APISentry (Zenodo 22658802)",
        "contamination": contamination,
        "n_estimators": n_estimators,
        "n_train_normal": n_train_normal,
        "approve_threshold_p95": round(p95_cal_normal, 6),
        "training_normal_calibrated": {
            "mean": round(mean_cal_normal, 6),
            "std": round(std_cal_normal, 6),
            "p50": round(p50_cal_normal, 6),
            "p95": round(p95_cal_normal, 6),
            "p99": round(p99_cal_normal, 6),
        },
        "test_binary": {
            "precision": round(test_precision, 6),
            "recall": round(test_recall, 6),
            "f1": round(test_f1, 6),
            "roc_auc": round(test_roc_auc, 6),
        },
        "test_per_attack_mean_score": per_class_scores,
        "calibration_method": "PercentileCalibrator (QuantileTransformer uniform, fit on train normal raw IF scores)",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 9. Clean console output
    print("=" * 60)
    print("  IF EXTERNAL — APISentry Re-training (percentile-calibrated)")
    print("=" * 60)
    print(f"  Train normal: {n_train_normal:,} samples (from train split, label_binary=0)")
    print(f"  IF config: contamination={contamination}, n_estimators={n_estimators}")
    print("  " + "─" * 45)
    print("  Calibrated training normals:")
    print(
        f"    mean={mean_cal_normal:.4f}, std={std_cal_normal:.4f}, "
        f"p50={p50_cal_normal:.4f}, p95={p95_cal_normal:.4f}, p99={p99_cal_normal:.4f}"
    )
    print(f"  APPROVE threshold (95th pct): {p95_cal_normal:.4f}")
    print("  " + "─" * 45)
    print("  Test (binary attack vs normal, threshold=APPROVE_THRESHOLD):")
    print(f"    precision : {test_precision:.4f}")
    print(f"    recall    : {test_recall:.4f}")
    print(f"    F1        : {test_f1:.4f}")
    print(f"    ROC-AUC   : {test_roc_auc:.4f}")
    print("  Test per-class mean calibrated score:")
    for cls_name, score in sorted(per_class_scores.items()):
        print(f"    {cls_name:<20}: {score:.4f}")

    if test_f1 < 0.70:
        print("  NOTE: IF alone is weaker than RF (F1 < 0.70); it contributes")
        print("        unsupervised novelty detection, not primary classification.")

    if test_f1 > 0.98:
        print("  WARNING: IF test binary F1 > 0.98; investigate potential overfitting.")

    print("  " + "─" * 45)
    print("  OLD approach (for comparison):")
    print(f"    Old hack: score = {0.38} - (raw * 1.2)")
    print("    Old resulting normal mean: 0.4605 (above 0.4 threshold = broken)")
    print("  New approach:")
    print("    Calibrated normal mean: ~0.50 (uniform by construction)")
    print("    Threshold at 95th percentile: derived, defensible")
    print("  " + "─" * 45)
    print("  Artifacts saved:")
    print(f"    {MODEL_PATH}")
    print(f"    {CALIBRATOR_PATH}")
    print(f"    {METRICS_PATH}")
    print("=" * 60)

    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
