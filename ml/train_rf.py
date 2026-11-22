"""Random Forest Supervised Classifier for API Anomaly Detection.

Trains on normal baseline traffic (label=0) and the 7 attack vectors (label=1)
to provide high-precision supervised anomaly scoring and feature importance
for SHAP explainability.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import structlog
from data.attack_traffic import generate_all_attacks_dataset
from data.normal_traffic import generate_normal_dataset
from ml.config import ml_settings

logger = structlog.get_logger(__name__)


def predict_rf_probability(model: RandomForestClassifier, vector: List[float]) -> float:
    """Predict anomaly probability P(label=1 | x) using Random Forest.

    Args:
        model: Trained RandomForestClassifier instance.
        vector: 12-dimensional feature vector.

    Returns:
        float between 0.0 (benign) and 1.0 (anomalous).
    """
    try:
        probabilities = model.predict_proba([vector])[0]
        # Class 1 corresponds to anomaly/attack
        if len(probabilities) > 1:
            return float(probabilities[1])
        return float(probabilities[0])
    except Exception as exc:
        logger.error("rf_prediction_failed", error=str(exc))
        return 0.5


def build_training_dataset(
    num_normal_sessions: int = 1000,
    attack_multipliers: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble balanced dataset of normal (label=0) and attack (label=1) samples."""
    normal_vectors = generate_normal_dataset(
        num_sessions=num_normal_sessions, reset_tracker_state=True
    )
    all_attack_vectors: List[List[float]] = []
    for _ in range(attack_multipliers):
        attack_vecs, _, _ = generate_all_attacks_dataset(reset_tracker_state=True)
        all_attack_vectors.extend(attack_vecs)

    x_data = np.array(normal_vectors + all_attack_vectors, dtype=np.float32)
    y_data = np.array(
        [0] * len(normal_vectors) + [1] * len(all_attack_vectors), dtype=np.int32
    )
    return x_data, y_data


def compute_rf_metrics(
    model: RandomForestClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute verified classification metrics on held-out test split."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "test_samples": int(len(y_test)),
        "report": report,
    }


def train_random_forest(
    num_normal_sessions: int = 1000,
    attack_multipliers: int = 2,
    test_size: float = 0.20,
    save_artifact: bool = True,
    output_path: Optional[Path] = None,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train RandomForestClassifier and evaluate on stratified test set."""
    target_path = output_path or ml_settings.rf_model_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("building_rf_dataset", normal_sessions=num_normal_sessions)
    x_all, y_all = build_training_dataset(
        num_normal_sessions=num_normal_sessions,
        attack_multipliers=attack_multipliers,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=test_size, random_state=42, stratify=y_all
    )

    logger.info("training_random_forest", train_samples=len(x_train), test_samples=len(x_test))
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        max_features=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    metrics = compute_rf_metrics(model, x_test, y_test)
    logger.info(
        "random_forest_evaluation",
        precision=round(metrics["precision"], 4),
        recall=round(metrics["recall"], 4),
        f1=round(metrics["f1_score"], 4),
        roc_auc=round(metrics["roc_auc"], 4),
    )

    if save_artifact:
        joblib.dump(model, target_path)
        logger.info("random_forest_artifact_saved", path=str(target_path))

    return model, metrics


def load_random_forest(model_path: Optional[Path] = None) -> Optional[RandomForestClassifier]:
    """Load persisted RandomForestClassifier from disk with error safety."""
    path = model_path or ml_settings.rf_model_path
    try:
        if not path.exists():
            logger.warning("rf_model_file_not_found", path=str(path))
            return None
        return joblib.load(path)
    except Exception as exc:
        logger.error("failed_to_load_rf_model", path=str(path), error=str(exc))
        return None


if __name__ == "__main__":
    train_random_forest(save_artifact=True)

