"""Real-Time Security Inference Pipeline.

Integrates feature extraction, LRU sliding window tracking, ensemble ML models,
and SHAP explainability into a unified high-throughput inference facade.
Strict SLA: Target latency < 200 ms per request.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
from uuid import uuid4
from fastapi import Request
import joblib
from sklearn.preprocessing import StandardScaler
import structlog
from backend.config import VerdictEnum, settings
from backend.middleware.interceptor import set_active_pipeline
from backend.models import SecurityEvaluationResult, SHAPExplanation
from ml.config import ml_settings
from ml.ensemble import calculate_ensemble_score, determine_verdict, predict_if_score
from ml.feature_engineering import extract_features_from_request, feature_dict_to_vector
from ml.shap_explainer import SecuritySHAPExplainer, get_shap_explainer
from ml.sliding_window import tracker
from ml.train_isolation_forest import load_isolation_forest
from ml.train_lstm import load_lstm_autoencoder, predict_lstm_score
from ml.train_rf import load_random_forest, predict_rf_probability

logger = structlog.get_logger(__name__)


class SecurityPipeline:
    """Production-grade security evaluation pipeline for Open Banking APIs."""

    def __init__(
        self,
        scaler: Optional[StandardScaler] = None,
        rf_model: Optional[Any] = None,
        if_model: Optional[Any] = None,
        lstm_model: Optional[Any] = None,
        explainer: Optional[SecuritySHAPExplainer] = None,
    ) -> None:
        self.scaler = scaler or self._load_scaler()
        self.rf_model = rf_model or load_random_forest()
        self.if_model = if_model or load_isolation_forest()
        # Set n_jobs=1 for single-sample inference to avoid multi-thread pool overhead
        if hasattr(self.rf_model, "n_jobs") and self.rf_model is not None:
            self.rf_model.n_jobs = 1
        if hasattr(self.if_model, "n_jobs") and self.if_model is not None:
            self.if_model.n_jobs = 1
        self.lstm_model = lstm_model or load_lstm_autoencoder()
        self.explainer = explainer or get_shap_explainer(model=self.rf_model)

    def _load_scaler(self) -> Optional[StandardScaler]:
        try:
            if ml_settings.scaler_path.exists():
                return joblib.load(ml_settings.scaler_path)
            return None
        except Exception as exc:
            logger.error("failed_to_load_scaler", error=str(exc))
            return None

    def evaluate_features(
        self,
        features: Dict[str, float],
        endpoint: str,
        method: str,
        client_id: Optional[str] = None,
        request_id: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> SecurityEvaluationResult:
        """Score feature vector through ensemble models and compute explanations."""
        t0 = start_time or time.perf_counter()
        vector = feature_dict_to_vector(features)

        # 1. Random Forest prediction
        rf_score = predict_rf_probability(self.rf_model, vector) if self.rf_model else 0.0

        # 2. Isolation Forest prediction
        if_score = predict_if_score(self.if_model, vector) if self.if_model else 0.0

        # 3. LSTM Autoencoder prediction
        lstm_score = predict_lstm_score(self.lstm_model, vector) if self.lstm_model else 0.0

        # 4. Ensemble weighted combination + pre-ML overrides
        ensemble_score = calculate_ensemble_score(
            rf_score=rf_score,
            if_score=if_score,
            lstm_score=lstm_score,
            feature_dict=features,
        )

        # 5. Verdict determination
        verdict = determine_verdict(ensemble_score)

        # 6. SHAP explanation on FLAG or BLOCK
        explanations: List[SHAPExplanation] = []
        if verdict != VerdictEnum.APPROVE and self.explainer:
            explanations = self.explainer.explain_vector(vector, max_features=3)
            # Ensure critical pre-ML override features are explicitly attributed
            override_features = [
                ("payload_sql_score", "SQL syntax patterns detected in payload"),
                ("scope_mismatch", "Requested resource requires OAuth scopes not present in token"),
                ("payload_param_tamper", "Parameter tampering / BOLA unauthorized object access detected"),
            ]
            existing_feats = {e.feature for e in explanations}
            for feat_name, feat_desc in override_features:
                if features.get(feat_name, 0.0) >= 0.8 and feat_name not in existing_feats:
                    explanations.insert(
                        0,
                        SHAPExplanation(
                            feature=feat_name,
                            contribution=float(round(features[feat_name], 4)),
                            description=feat_desc,
                        ),
                    )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SecurityEvaluationResult(
            verdict=verdict,
            anomaly_score=round(ensemble_score, 4),
            latency_ms=round(latency_ms, 2),
            explanations=explanations,
            client_id=client_id,
            endpoint=endpoint,
            http_method=method,
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request_id or str(uuid4()),
        )

    async def evaluate_http_request(
        self, request: Request, body_bytes: bytes
    ) -> SecurityEvaluationResult:
        """Extract HTTP telemetry and execute real-time security assessment."""
        t0 = time.perf_counter()
        path = request.url.path
        method = request.method
        headers = {k.lower(): v for k, v in request.headers.items()}
        client_id = headers.get(settings.header_client_id)
        consent_id = headers.get(settings.header_consent_id)

        features = extract_features_from_request(
            endpoint=path,
            method=method,
            headers=headers,
            body_bytes=body_bytes,
            client_key=client_id,
            account_id=consent_id,
            token_id=consent_id,
        )

        return self.evaluate_features(
            features=features,
            endpoint=path,
            method=method,
            client_id=client_id,
            start_time=t0,
        )


_GLOBAL_PIPELINE: Optional[SecurityPipeline] = None


def get_security_pipeline(force_reload: bool = False) -> SecurityPipeline:
    """Retrieve or initialize singleton SecurityPipeline."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None or force_reload:
        _GLOBAL_PIPELINE = SecurityPipeline()
    return _GLOBAL_PIPELINE


def register_pipeline_with_interceptor() -> SecurityPipeline:
    """Register ML security pipeline with Starlette interceptor middleware."""
    pipeline = get_security_pipeline()
    set_active_pipeline(pipeline.evaluate_http_request)
    logger.info("ml_security_pipeline_registered_with_interceptor")
    return pipeline
