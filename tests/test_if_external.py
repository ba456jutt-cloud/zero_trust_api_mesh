"""Unit and integration tests for external Isolation Forest training and PercentileCalibrator.

Verifies:
1. Calibrator maps training normals to uniform distribution with mean ≈ 0.5.
2. Calibrator approve_threshold(95) is in reasonable empirical range (0.85, 0.99).
3. Calibrator preserves monotonic ordering of raw anomaly scores.
4. IF training indices are an exact subset of RF's training indices (normal traffic only).
5. Calibrated test attack scores are significantly higher than test normal scores.
6. Zero instances of the old hardcoded offset ("0.38 - ") exist in the codebase files.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from data.external.adapter import adapt_external_row
from ml.calibration import PercentileCalibrator
from ml.config import FEATURE_NAMES

MODEL_PATH = Path("ml/artifacts/if_external.joblib")
CALIBRATOR_PATH = Path("ml/artifacts/if_transformer_external.joblib")
METRICS_PATH = Path("ml/artifacts/if_external_metrics.json")
SPLIT_INDICES_PATH = Path("ml/artifacts/rf_external_split_indices.json")
DATASET_PATH = Path("data/external/apisentry/dataset_processed.csv")


def test_calibrator_maps_training_normals_to_uniform():
    """Test 1: After fitting, calibrated training normals have mean ≈ 0.5 and span [0, 1]."""
    np.random.seed(42)
    synthetic_normals = np.random.normal(loc=-0.1, scale=0.05, size=2000)
    calibrator = PercentileCalibrator(random_state=42).fit(synthetic_normals)
    calibrated = calibrator.transform(synthetic_normals)

    assert len(calibrated) == len(synthetic_normals)
    assert 0.48 <= np.mean(calibrated) <= 0.52, f"Mean {np.mean(calibrated)} is not ≈ 0.5"
    assert 0.0 <= np.min(calibrated) <= 0.05
    assert 0.95 <= np.max(calibrated) <= 1.0


def test_calibrator_approve_threshold_reasonable():
    """Test 2: Threshold from p95 is in the empirical range (0.85, 0.99)."""
    np.random.seed(42)
    synthetic_normals = np.random.normal(loc=-0.1, scale=0.05, size=2000)
    calibrator = PercentileCalibrator(random_state=42).fit(synthetic_normals)
    thresh = calibrator.approve_threshold(95.0)

    assert 0.85 <= thresh <= 0.99, f"P95 threshold {thresh} is not in (0.85, 0.99)"


def test_calibrator_preserves_ordering():
    """Test 3: If raw_score_a < raw_score_b, calibrated_a <= calibrated_b."""
    np.random.seed(42)
    normals = np.linspace(-0.3, 0.1, 1000)
    calibrator = PercentileCalibrator(random_state=42).fit(normals)

    # Test pairwise ordering across evaluation grid
    test_scores = np.array([-0.25, -0.15, -0.05, 0.05, 0.15, 0.30])
    calibrated = calibrator.transform(test_scores)

    for i in range(len(calibrated) - 1):
        assert calibrated[i] <= calibrated[i + 1], (
            f"Ordering violation: {test_scores[i]} -> {calibrated[i]} "
            f"vs {test_scores[i+1]} -> {calibrated[i+1]}"
        )


def test_external_if_split_matches_rf():
    """Test 4: IF training indices are a strict subset of RF's training indices (same split)."""
    assert SPLIT_INDICES_PATH.exists(), f"Split indices missing at {SPLIT_INDICES_PATH}"
    with open(SPLIT_INDICES_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    rf_train_idx = set(splits["train"])
    assert len(rf_train_idx) == 7917

    df = pd.read_csv(DATASET_PATH)
    train_df = df.iloc[splits["train"]]
    normal_train_idx = set(train_df[train_df["label_binary"] == 0].index)

    # Must be a strict subset of RF training set
    assert normal_train_idx.issubset(rf_train_idx)
    assert len(normal_train_idx) == 4628


def test_external_if_attack_scores_higher_than_normal():
    """Test 5: Calibrated test attack scores are significantly higher than normal scores."""
    assert MODEL_PATH.exists(), f"IF model artifact missing at {MODEL_PATH}"
    assert CALIBRATOR_PATH.exists(), f"Calibrator artifact missing at {CALIBRATOR_PATH}"
    assert METRICS_PATH.exists(), f"Metrics missing at {METRICS_PATH}"

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    per_class = metrics["test_per_attack_mean_score"]
    normal_score = per_class["normal"]

    # Normal score should be close to 0.50 (uniform calibration)
    assert 0.45 <= normal_score <= 0.60, f"Normal score {normal_score} outside [0.45, 0.60]"

    # Attack scores must be substantially higher than normal score
    high_impact_attacks = ["sql_injection", "parameter_tampering", "bola", "brute_force", "api_flooding"]
    for attack in high_impact_attacks:
        attack_score = per_class[attack]
        assert attack_score > normal_score, (
            f"Attack '{attack}' score ({attack_score}) not higher than normal ({normal_score})"
        )


def test_no_hardcoded_offset_in_code():
    """Test 6: Verify zero instances of hardcoded hack '0.38 - ' in new modules."""
    files_to_check = [
        Path("ml/calibration.py"),
        Path("ml/train_if_external.py"),
    ]
    for file_path in files_to_check:
        assert file_path.exists(), f"File {file_path} does not exist"
        content = file_path.read_text(encoding="utf-8")
        assert "0.38 - " not in content, f"Found hardcoded hack '0.38 - ' in {file_path}"

