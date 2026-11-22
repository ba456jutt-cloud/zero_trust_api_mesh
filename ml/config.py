"""Machine Learning and Feature Engineering Configuration.

Centralizes:
- Feature vector definitions and human-readable names
- Ensemble model weights (Random Forest, Isolation Forest, LSTM Autoencoder)
- Artifact file paths
"""

from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel, Field

# Ordered list of extracted features fed into the ML ensemble
FEATURE_NAMES: List[str] = [
    "scope_mismatch",
    "sequence_anomaly",
    "benford_deviation",
    "request_rate_10s",
    "request_rate_60s",
    "distinct_accounts_ratio",
    "token_reuse_count",
    "payload_sql_score",
    "payload_param_tamper",
    "payload_length",
    "is_post_method",
    "endpoint_risk_weight",
]

# Human-readable descriptions for SHAP explainability attribution
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "scope_mismatch": "OAuth token scope does not authorize requested endpoint",
    "sequence_anomaly": "Navigation bypassed mandatory prior steps (e.g. skipped consent/accounts)",
    "benford_deviation": "Transaction amount violates natural Benford's Law distribution",
    "request_rate_10s": "High burst request velocity in 10-second window",
    "request_rate_60s": "Elevated cumulative request volume in 60-second window",
    "distinct_accounts_ratio": "Abnormally high number of distinct accounts probed (credential stuffing)",
    "token_reuse_count": "Replay of previously observed token/JTI",
    "payload_sql_score": "Presence of SQL injection syntax or tampering tokens in payload",
    "payload_param_tamper": "Account ID parameter mismatch against authorized session consent",
    "payload_length": "Abnormal request body payload size",
    "is_post_method": "State-mutating HTTP POST operation",
    "endpoint_risk_weight": "Inherent regulatory risk weighting of the targeted banking endpoint",
}


class MLSettings(BaseModel):
    """Configuration for ML models, artifacts, and ensemble weights."""

    feature_names: List[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    feature_descriptions: Dict[str, str] = Field(default_factory=lambda: dict(FEATURE_DESCRIPTIONS))

    # Ensemble model weights (must normalize to 1.0)
    rf_weight: float = 0.45
    if_weight: float = 0.30
    lstm_weight: float = 0.25

    # Artifact storage paths
    artifacts_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts")
    rf_model_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "rf_model.joblib")
    if_model_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "isolation_forest.joblib")
    lstm_model_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "lstm_autoencoder.pt")
    scaler_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "feature_scaler.joblib")
    shap_explainer_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts" / "shap_explainer.joblib")

    # Endpoint Risk Multipliers
    endpoint_risk_map: Dict[str, float] = Field(
        default_factory=lambda: {
            "/transfer": 1.0,
            "/accounts": 0.5,
            "/transactions": 0.5,
            "/consent": 0.3,
        }
    )


def get_ml_settings() -> MLSettings:
    """Retrieve ML configuration instance."""
    return MLSettings()


ml_settings = get_ml_settings()

