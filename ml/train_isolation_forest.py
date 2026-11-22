"""Isolation Forest Unsupervised Anomaly Detection Model.

Trains strictly on normal baseline synthetic traffic (label=0)
to detect behavioral outliers and unknown attack patterns.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
import structlog
from data.attack_traffic import generate_all_attacks_dataset
from data.normal_traffic import generate_normal_dataset
from ml.calibration import PercentileCalibrator
from ml.config import ml_settings
from ml.ensemble import predict_if_score

logger = structlog.get_logger(__name__)



def train_isolation_forest(
    num_sessions: int = 400,
    contamination: float = 0.03,
    save_artifact: bool = True,
    output_path: Optional[Path] = None,
) -> Tuple[IsolationForest, Dict[str, float]]:
    """Train Isolation Forest on normal baseline and validate against attack vectors."""
    target_path = output_path or ml_settings.if_model_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(8)
    np.random.seed(8)
    logger.info("generating_normal_training_data", sessions=num_sessions)
    train_vectors = generate_normal_dataset(num_sessions=num_sessions, reset_tracker_state=True)

    logger.info("training_isolation_forest", samples=len(train_vectors), contamination=contamination)
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_vectors)

    # Fit percentile calibrator on normal training scores
    raw_train = -1.0 * model.decision_function(train_vectors)
    calibrator = PercentileCalibrator(random_state=42).fit(raw_train)
    model._calibrator = calibrator

    # Evaluate separation on normal vs attack traffic
    eval_normal = generate_normal_dataset(num_sessions=50, reset_tracker_state=True)
    attack_vectors, _, _ = generate_all_attacks_dataset(reset_tracker_state=True)

    normal_scores = [predict_if_score(model, v) for v in eval_normal]
    attack_scores = [predict_if_score(model, v) for v in attack_vectors]

    avg_normal = sum(normal_scores) / len(normal_scores) if normal_scores else 0.0
    avg_attack = sum(attack_scores) / len(attack_scores) if attack_scores else 0.0
    separation_margin = avg_attack - avg_normal

    metrics = {
        "training_samples": float(len(train_vectors)),
        "eval_normal_samples": float(len(eval_normal)),
        "eval_attack_samples": float(len(attack_vectors)),
        "avg_normal_score": avg_normal,
        "avg_attack_score": avg_attack,
        "separation_margin": separation_margin,
    }

    logger.info(
        "isolation_forest_evaluation",
        avg_normal_score=round(avg_normal, 4),
        avg_attack_score=round(avg_attack, 4),
        separation_margin=round(separation_margin, 4),
    )

    if save_artifact:
        joblib.dump(model, target_path)
        logger.info("isolation_forest_artifact_saved", path=str(target_path))

        calibrator_path = target_path.parent / "if_transformer.joblib"
        calibrator.save(calibrator_path)
        logger.info("isolation_forest_calibrator_saved", path=str(calibrator_path))

        cal_train = calibrator.transform(raw_train)
        meta_payload = {
            "source": "synthetic",
            "model_artifact": target_path.name,
            "n_normal_samples": int(len(train_vectors)),
            "calibrated_normal_stats": {
                "mean": float(np.mean(cal_train)),
                "std": float(np.std(cal_train)),
                "p50": float(np.percentile(cal_train, 50)),
                "p95": float(np.percentile(cal_train, 95)),
                "p99": float(np.percentile(cal_train, 99)),
            },
            "approve_threshold_p95": float(np.percentile(cal_train, 95)),
            "old_hack": "0.38 - (raw * 1.2)",
            "new_method": "PercentileCalibrator (QuantileTransformer uniform)",
        }
        meta_path = target_path.parent / "if_calibration_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)
        logger.info("isolation_forest_meta_saved", path=str(meta_path))

    return model, metrics


def load_isolation_forest(path: Optional[Path] = None) -> Optional[IsolationForest]:
    """Load serialized Isolation Forest model from disk."""
    model_path = path or ml_settings.if_model_path
    if not model_path.exists():
        logger.warning("isolation_forest_artifact_missing", path=str(model_path))
        return None
    try:
        model: IsolationForest = joblib.load(model_path)
        return model
    except Exception as exc:
        logger.error("isolation_forest_load_failed", path=str(model_path), error=str(exc))
        return None


if __name__ == "__main__":
    train_isolation_forest()

