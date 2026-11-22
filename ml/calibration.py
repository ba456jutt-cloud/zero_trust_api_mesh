"""Reusable Percentile-Based Anomaly Score Calibrator.

Replaces arbitrary hardcoded linear offsets with an empirical cumulative
distribution function (CDF) fitted on normal training traffic using scikit-learn's
QuantileTransformer.
"""

from pathlib import Path
from typing import Union
import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer


class PercentileCalibrator:
    """Maps raw anomaly scores to [0, 1] using empirical CDF of training normals.

    Higher raw score represents higher anomaly likelihood (e.g., `-1.0 * decision_function(X)`).
    Calibrated scores are uniform in [0, 1] for normal traffic:
    - Median normal traffic scores ~0.50
    - The 95th percentile of normal traffic yields the empirical APPROVE threshold (~0.95)
    - Extreme anomalies (outliers relative to normal CDF) saturate toward 1.0
    """

    def __init__(self, random_state: int = 42):
        self._quantile_transformer: Union[QuantileTransformer, None] = None
        self._calibrated_training_normals: Union[np.ndarray, None] = None
        self._random_state = random_state

    def fit(self, normal_raw_scores: np.ndarray) -> "PercentileCalibrator":
        """Fit empirical CDF on raw anomaly scores of NORMAL training samples.

        Args:
            normal_raw_scores: 1D array of raw scores where higher = more anomalous.

        Returns:
            Self (fitted instance).
        """
        arr = np.asarray(normal_raw_scores, dtype=float).ravel()
        if len(arr) == 0:
            raise ValueError("Cannot fit PercentileCalibrator on empty array.")

        n_quantiles = min(1000, max(10, len(arr)))
        self._quantile_transformer = QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=n_quantiles,
            random_state=self._random_state,
        )
        self._quantile_transformer.fit(arr.reshape(-1, 1))

        # Store calibrated training normals to accurately compute empirical percentiles
        self._calibrated_training_normals = np.clip(
            self._quantile_transformer.transform(arr.reshape(-1, 1)).ravel(),
            0.0,
            1.0,
        )
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores to calibrated [0, 1] probabilities.

        Args:
            raw_scores: 1D or 2D array of raw scores (higher = more anomalous).

        Returns:
            1D array of calibrated anomaly scores in [0.0, 1.0].
        """
        if self._quantile_transformer is None:
            raise ValueError("PercentileCalibrator must be fitted before calling transform.")

        arr = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        calibrated = self._quantile_transformer.transform(arr).ravel()
        return np.clip(calibrated, 0.0, 1.0)

    def approve_threshold(self, percentile: float = 95.0) -> float:
        """Return the calibrated score at the given percentile of training normals.

        For percentile=95.0, this is the empirical APPROVE threshold.
        Derived from data without magic numbers.

        Args:
            percentile: Percentile value in [0.0, 100.0] (default 95.0).

        Returns:
            Calibrated score threshold float.
        """
        if self._calibrated_training_normals is None:
            raise ValueError("PercentileCalibrator must be fitted before calculating threshold.")
        return float(np.percentile(self._calibrated_training_normals, percentile))

    def save(self, path: Union[str, Path]) -> None:
        """Serialize calibrator instance to disk."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PercentileCalibrator":
        """Load serialized calibrator instance from disk."""
        obj = joblib.load(path)
        if not isinstance(obj, PercentileCalibrator):
            raise TypeError(f"Loaded object is not a PercentileCalibrator: {type(obj)}")
        return obj
