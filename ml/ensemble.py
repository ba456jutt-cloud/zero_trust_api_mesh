"""  """"""Ensemble Scoring and Verdict Determination Engine.

Combines predictions from Random Forest, Isolation Forest, and LSTM Autoencoder
using regulatory sandbox weighted blending and deterministic pre-ML safeguards:
- RF Weight: 0.45 (Supervised classification)
- IF Weight: 0.30 (Spatial unsupervised anomaly)
- LSTM Weight: 0.25 (Sequential reconstruction error)

Thresholds:
- Score < 0.40  -> APPROVE (Normal pass-through)
- 0.40 <= Score <= 0.70 -> FLAG (Warning header + SOC alert)
- Score > 0.70  -> BLOCK (403 Forbidden + SHAP explanation)
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
import structlog

from backend.config import VerdictEnum, settings
from ml.calibration import PercentileCalibrator
from ml.config import ml_settings

logger = structlog.get_logger(__name__)

# Cached module-level calibrator
_IF_CALIBRATOR_PATH = Path("ml/artifacts/if_transformer.joblib")
_IF_CALIBRATOR: Optional[PercentileCalibrator] = None


def get_if_calibrator() -> Optional[PercentileCalibrator]:
    """Retrieve or lazily load the cached synthetic PercentileCalibrator instance."""
    global _IF_CALIBRATOR
    if _IF_CALIBRATOR is None and _IF_CALIBRATOR_PATH.exists():
        try:
            _IF_CALIBRATOR = PercentileCalibrator.load(_IF_CALIBRATOR_PATH)
        except Exception as exc:
            logger.error("failed_to_load_if_calibrator", error=str(exc))
    return _IF_CALIBRATOR


def predict_if_score(
    model: IsolationForest,
    vector: List[float],
    calibrator: Optional[PercentileCalibrator] = None,
) -> float:
    """Compute percentile-calibrated anomaly score from Isolation Forest.

    Replaces legacy affine hack with empirical CDF calibration.

    Args:
        model: Fitted Isolation Forest model.
        vector: 12-dimensional raw feature vector.
        calibrator: Optional explicit calibrator; defaults to cached synthetic calibrator.

    Returns:
        float anomaly score in [0.0, 1.0].
    """
    try:
        # decision_function outputs > 0 for inliers, < 0 for outliers
        raw_decision = float(model.decision_function([vector])[0])
        raw_anomaly = -1.0 * raw_decision

        cal = calibrator or getattr(model, "_calibrator", None) or get_if_calibrator()
        if cal is not None:
            calibrated = float(cal.transform(np.array([raw_anomaly]))[0])
            return float(max(0.0, min(1.0, calibrated)))

        # Fallback if calibrator artifact is not yet created
        return float(1.0 / (1.0 + np.exp(-10.0 * raw_anomaly)))
    except Exception as exc:
        logger.error("isolation_forest_prediction_failed", error=str(exc))
        return 0.5


def calculate_ensemble_score(
    rf_score: float,
    if_score: float,
    lstm_score: float,
    feature_dict: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted anomaly score with pre-ML deterministic overrides.

    Args:
        rf_score: Supervised attack probability in [0.0, 1.0].
        if_score: Unsupervised spatial anomaly score in [0.0, 1.0].
        lstm_score: Normalized sequential reconstruction error in [0.0, 1.0].
        feature_dict: Raw extracted features for hard pre-ML rule checks.

    Returns:
        float ensemble anomaly score in [0.0, 1.0].
    """
    weighted_sum = (
        (ml_settings.rf_weight * rf_score)
        + (ml_settings.if_weight * if_score)
        + (ml_settings.lstm_weight * lstm_score)
    )

    # Pre-ML deterministic hard overrides
    if feature_dict:
        # Scope mismatch against FAPI requirements is an immediate hard block
        if feature_dict.get("scope_mismatch", 0.0) >= 0.8:
            weighted_sum = max(weighted_sum, 0.85)

        # High-confidence SQL injection syntax triggers immediate block
        if feature_dict.get("payload_sql_score", 0.0) >= 0.8:
            weighted_sum = max(weighted_sum, 0.90)

        # High-velocity API flooding (> 15 req/sec) triggers immediate block
        if feature_dict.get("request_rate_10s", 0.0) >= 0.8:
            weighted_sum = max(weighted_sum, 0.88)

    return float(max(0.0, min(1.0, weighted_sum)))


def determine_verdict(anomaly_score: float) -> VerdictEnum:
    """Map composite anomaly score to SBP regulatory verdict.

    Args:
        anomaly_score: Normalized score in [0.0, 1.0].

    Returns:
        VerdictEnum: APPROVE, FLAG, or BLOCK.
    """
    if anomaly_score < settings.approve_threshold:
        return VerdictEnum.APPROVE
    if anomaly_score <= settings.block_threshold:
        return VerdictEnum.FLAG
    return VerdictEnum.BLOCK
