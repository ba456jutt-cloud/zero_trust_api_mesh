"""Unit and integration tests for external Random Forest training.

Verifies:
1. Pipeline contains StandardScaler and RandomForestClassifier.
2. Metrics JSON file exists and contains all required keys.
3. CV macro-F1 is bounded within reasonable limits (0.70 <= score <= 1.0).
4. Feature matrix contains zero NaN values after applying adapter.
5. Train/Val/Test split indices are pairwise disjoint and cover all rows.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.external.adapter import adapt_external_row
from ml.config import FEATURE_NAMES

MODEL_PATH = Path("ml/artifacts/rf_external.joblib")
METRICS_PATH = Path("ml/artifacts/rf_external_metrics.json")
SPLIT_INDICES_PATH = Path("ml/artifacts/rf_external_split_indices.json")
DATASET_PATH = Path("data/external/apisentry/dataset_processed.csv")


def test_no_nan_in_features():
    """Test 4: After adapter, zero NaN in feature matrix across dataset."""
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH)
    records = df.to_dict(orient="records")
    adapted = [adapt_external_row(r) for r in records]
    X_df = pd.DataFrame(adapted)[FEATURE_NAMES]

    assert X_df.shape[1] == 12
    assert X_df.isna().sum().sum() == 0
    assert not np.isinf(X_df.to_numpy()).any()


def test_split_indices_disjoint_and_cover_all():
    """Test 5: Train/Val/Test indices are disjoint and union covers all rows."""
    assert SPLIT_INDICES_PATH.exists(), f"Indices file not found at {SPLIT_INDICES_PATH}"
    with open(SPLIT_INDICES_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_idx = set(splits["train"])
    val_idx = set(splits["val"])
    test_idx = set(splits["test"])

    # Disjointness
    assert len(train_idx.intersection(val_idx)) == 0, "Train and Val indices overlap"
    assert len(train_idx.intersection(test_idx)) == 0, "Train and Test indices overlap"
    assert len(val_idx.intersection(test_idx)) == 0, "Val and Test indices overlap"

    # Total coverage
    df = pd.read_csv(DATASET_PATH)
    total_samples = len(df)
    union_idx = train_idx | val_idx | test_idx
    assert len(union_idx) == total_samples, f"Union ({len(union_idx)}) != total samples ({total_samples})"
    assert len(train_idx) == 7917, f"Expected 7,917 train rows, got {len(train_idx)}"
    assert len(val_idx) == 1696, f"Expected 1,696 val rows, got {len(val_idx)}"
    assert len(test_idx) == 1697, f"Expected 1,697 test rows, got {len(test_idx)}"


def test_pipeline_has_scaler_and_rf():
    """Test 1: Fitted pipeline contains StandardScaler and RandomForestClassifier."""
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)

    assert isinstance(pipeline, Pipeline), "Saved model is not an sklearn Pipeline"
    assert "standardscaler" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["standardscaler"], StandardScaler)
    assert "randomforestclassifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["randomforestclassifier"], RandomForestClassifier)


def test_metrics_json_exists_and_valid():
    """Test 2: Metrics JSON loads and has all required structure keys."""
    assert METRICS_PATH.exists(), f"Metrics file not found at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    required_keys = [
        "dataset",
        "n_train",
        "n_val",
        "n_test",
        "best_params",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "val_macro_f1",
        "test_multiclass",
        "test_binary",
        "confusion_matrix",
        "random_state",
        "trained_at",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key in metrics JSON: {key}"

    # Multiclass sub-keys
    assert "macro_f1" in metrics["test_multiclass"]
    assert "weighted_f1" in metrics["test_multiclass"]
    assert "accuracy" in metrics["test_multiclass"]
    assert "per_class" in metrics["test_multiclass"]

    # Binary sub-keys
    assert "precision" in metrics["test_binary"]
    assert "recall" in metrics["test_binary"]
    assert "f1" in metrics["test_binary"]
    assert "roc_auc" in metrics["test_binary"]


def test_cv_f1_reasonable():
    """Test 3: CV macro-F1 is within sensible sanity bounds [0.70, 1.0]."""
    assert METRICS_PATH.exists(), f"Metrics file not found at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cv_mean = metrics["cv_macro_f1_mean"]
    assert 0.70 <= cv_mean <= 1.0, f"CV macro-F1 {cv_mean} out of bounds [0.70, 1.0]"

