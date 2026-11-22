"""Master Orchestrator for API Security Mesh ML Models.

Executes end-to-end training pipeline for all anomaly detection models:
1. Fits and saves StandardScaler on normal baseline feature vectors
2. Trains Isolation Forest for unsupervised spatial anomaly detection
3. Trains Random Forest for supervised attack classification
4. Trains PyTorch LSTM Autoencoder for temporal/sequential anomaly detection
"""

import random
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import structlog
from data.normal_traffic import generate_normal_dataset
from ml.config import ml_settings
from ml.train_isolation_forest import train_isolation_forest
from ml.train_lstm import train_lstm_autoencoder
from ml.train_rf import train_random_forest

logger = structlog.get_logger(__name__)


def fit_and_save_scaler(
    num_sessions: int = 400,
    output_path: Path | None = None,
) -> StandardScaler:
    """Fit StandardScaler on normal baseline feature vectors and persist."""
    target_path = output_path or ml_settings.scaler_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("fitting_feature_scaler", sessions=num_sessions)
    normal_vectors = generate_normal_dataset(num_sessions=num_sessions, reset_tracker_state=True)

    scaler = StandardScaler()
    scaler.fit(normal_vectors)

    joblib.dump(scaler, target_path)
    logger.info("scaler_artifact_saved", path=str(target_path))
    return scaler


def verify_artifacts_exist() -> Dict[str, bool]:
    """Verify that all required model artifacts are persisted and non-empty."""
    required = {
        "scaler": ml_settings.scaler_path,
        "isolation_forest": ml_settings.if_model_path,
        "random_forest": ml_settings.rf_model_path,
        "lstm_autoencoder": ml_settings.lstm_model_path,
    }
    status = {}
    for name, path in required.items():
        exists_and_valid = path.exists() and path.stat().st_size > 0
        status[name] = exists_and_valid
        if not exists_and_valid:
            logger.error("missing_or_empty_artifact", name=name, path=str(path))
    return status


def train_all_models(
    num_sessions: int = 400,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Run full reproducible training pipeline across all security models."""
    random.seed(8)
    np.random.seed(8)
    logger.info("starting_full_ml_training_pipeline")

    # Step 1: Feature Scaler
    scaler = fit_and_save_scaler(num_sessions=num_sessions)

    # Step 2: Isolation Forest
    logger.info("stage_2_isolation_forest")
    _, if_metrics = train_isolation_forest(
        num_sessions=num_sessions,
        save_artifact=save_artifacts,
    )

    # Step 3: Random Forest
    logger.info("stage_3_random_forest")
    _, rf_metrics = train_random_forest(
        num_normal_sessions=num_sessions,
        save_artifact=save_artifacts,
    )

    # Step 4: LSTM Autoencoder
    logger.info("stage_4_lstm_autoencoder")
    _, lstm_metrics = train_lstm_autoencoder(
        num_sessions=num_sessions,
        epochs=15,
        save_artifact=save_artifacts,
    )

    artifacts_status = verify_artifacts_exist()
    all_valid = all(artifacts_status.values())

    summary: Dict[str, Any] = {
        "all_artifacts_valid": all_valid,
        "scaler_fitted": scaler is not None,
        "isolation_forest": if_metrics,
        "random_forest": {
            "precision": rf_metrics["precision"],
            "recall": rf_metrics["recall"],
            "f1_score": rf_metrics["f1_score"],
            "roc_auc": rf_metrics["roc_auc"],
        },
        "lstm_autoencoder": {
            "initial_loss": lstm_metrics["initial_loss"],
            "final_loss": lstm_metrics["final_loss"],
            "separation_margin": lstm_metrics["separation_margin"],
        },
        "artifacts": artifacts_status,
    }

    logger.info("full_ml_training_pipeline_completed", all_valid=all_valid)
    return summary


if __name__ == "__main__":
    train_all_models()

