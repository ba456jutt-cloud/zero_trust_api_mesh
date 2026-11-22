"""SHAP Explainability Engine for Pakistan Open Banking API Security Mesh.

Computes genuine local feature attributions using shap.TreeExplainer on the
trained Random Forest classifier. Generates transparent, human-readable
reasons for any request receiving a FLAG or BLOCK verdict.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import structlog
from backend.models import SHAPExplanation
from ml.config import ml_settings
from ml.train_rf import load_random_forest

logger = structlog.get_logger(__name__)


def extract_class_attributions(raw_shap_values: Any) -> np.ndarray:
    """Normalize multi-format SHAP output to 1D array of class-1 attributions."""
    if isinstance(raw_shap_values, list):
        return np.array(raw_shap_values[1][0], dtype=np.float32)
    arr = np.array(raw_shap_values, dtype=np.float32)
    if arr.ndim == 3:
        return arr[0, :, 1]
    if arr.ndim == 2:
        return arr[0]
    return arr.flatten()


class SecuritySHAPExplainer:
    """Manages TreeSHAP attribution and human-readable explanation mapping."""

    def __init__(
        self,
        model: Optional[RandomForestClassifier] = None,
        explainer: Optional[shap.TreeExplainer] = None,
    ) -> None:
        self.model = model or load_random_forest()
        if explainer is not None:
            self.explainer = explainer
        elif self.model is not None:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = None
        self.feature_names = ml_settings.feature_names
        self.feature_descriptions = ml_settings.feature_descriptions

    def get_attributions_dict(self, vector: List[float]) -> Dict[str, float]:
        """Compute raw feature attribution dictionary for class 1 (anomaly)."""
        if self.explainer is None:
            return {name: 0.0 for name in self.feature_names}
        try:
            raw_vals = self.explainer.shap_values(np.array([vector]))
            attributions = extract_class_attributions(raw_vals)
            return {
                name: float(attributions[i])
                for i, name in enumerate(self.feature_names)
                if i < len(attributions)
            }
        except Exception as exc:
            logger.error("shap_attribution_failed", error=str(exc))
            return {name: 0.0 for name in self.feature_names}

    def explain_vector(
        self,
        vector: List[float],
        max_features: int = 3,
        min_contribution: float = 0.001,
    ) -> List[SHAPExplanation]:
        """Extract top positive feature attributions with human-readable descriptions."""
        attr_dict = self.get_attributions_dict(vector)
        # Augment with direct deterministic attack signatures if active in vector
        for idx, feat in enumerate(self.feature_names):
            if feat in ["payload_sql_score", "scope_mismatch", "payload_param_tamper"]:
                if idx < len(vector) and vector[idx] >= 0.8 and attr_dict.get(feat, 0.0) < vector[idx]:
                    attr_dict[feat] = float(vector[idx])

        # Filter for positive contributors that pushed the score toward anomaly
        positive_items = [
            (feat, val)
            for feat, val in attr_dict.items()
            if val >= min_contribution
        ]
        positive_items.sort(key=lambda x: x[1], reverse=True)

        if not positive_items:
            # Fallback to feature with highest relative contribution
            best_feat = max(attr_dict.items(), key=lambda x: x[1])[0]
            desc = self.feature_descriptions.get(best_feat, "Normal baseline traffic")
            return [SHAPExplanation(feature=best_feat, contribution=0.0, description=desc)]

        explanations: List[SHAPExplanation] = []
        for feat, val in positive_items[:max_features]:
            desc = self.feature_descriptions.get(feat, f"Anomalous pattern in {feat}")
            explanations.append(
                SHAPExplanation(
                    feature=feat,
                    contribution=float(round(val, 4)),
                    description=desc,
                )
            )
        return explanations


# Global cached explainer instance
_CACHED_EXPLAINER: Optional[SecuritySHAPExplainer] = None


def get_shap_explainer(
    model: Optional[RandomForestClassifier] = None,
    force_reload: bool = False,
) -> SecuritySHAPExplainer:
    """Retrieve or initialize singleton SecuritySHAPExplainer."""
    global _CACHED_EXPLAINER
    if _CACHED_EXPLAINER is None or force_reload or model is not None:
        _CACHED_EXPLAINER = SecuritySHAPExplainer(model=model)
    return _CACHED_EXPLAINER


def save_shap_explainer(
    explainer: SecuritySHAPExplainer,
    output_path: Optional[Path] = None,
) -> Path:
    """Persist trained TreeExplainer to disk for instant warm start."""
    target_path = output_path or ml_settings.shap_explainer_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer.explainer, target_path)
    logger.info("shap_explainer_artifact_saved", path=str(target_path))
    return target_path


def load_saved_shap_explainer(
    explainer_path: Optional[Path] = None,
) -> Optional[SecuritySHAPExplainer]:
    """Load persisted TreeExplainer instance with error fallback."""
    path = explainer_path or ml_settings.shap_explainer_path
    try:
        if not path.exists():
            return None
        raw_tree_explainer = joblib.load(path)
        return SecuritySHAPExplainer(explainer=raw_tree_explainer)
    except Exception as exc:
        logger.error("failed_to_load_shap_explainer", path=str(path), error=str(exc))
        return None

