"""Unit and Integration tests for Ensemble Scorer and ML Inference Pipeline."""

import json
import pytest
from httpx import ASGITransport, AsyncClient
from backend.config import VerdictEnum, settings
from backend.main import create_app
from backend.models import SecurityBlockResponse, SecurityEvaluationResult
from ml.config import ml_settings
from ml.ensemble import calculate_ensemble_score, determine_verdict
from ml.pipeline import get_security_pipeline, register_pipeline_with_interceptor


def test_ensemble_weights_normalization() -> None:
    """Verify that ensemble model weights strictly sum to 1.0."""
    total_weight = ml_settings.rf_weight + ml_settings.if_weight + ml_settings.lstm_weight
    assert abs(total_weight - 1.0) < 1e-5


def test_ensemble_scoring_and_verdicts() -> None:
    """Verify weighted blending and threshold mapping to APPROVE, FLAG, BLOCK."""
    # 1. Normal traffic (low scores across all models)
    low_score = calculate_ensemble_score(rf_score=0.1, if_score=0.2, lstm_score=0.1)
    assert low_score < 0.40
    assert determine_verdict(low_score) == VerdictEnum.APPROVE

    # 2. Suspicious traffic (medium scores)
    medium_score = calculate_ensemble_score(rf_score=0.5, if_score=0.55, lstm_score=0.5)
    assert 0.40 <= medium_score <= 0.70
    assert determine_verdict(medium_score) == VerdictEnum.FLAG

    # 3. High confidence anomaly
    high_score = calculate_ensemble_score(rf_score=0.9, if_score=0.8, lstm_score=0.85)
    assert high_score > 0.70
    assert determine_verdict(high_score) == VerdictEnum.BLOCK


def test_pre_ml_hard_rule_overrides() -> None:
    """Verify hard pre-ML triggers force BLOCK verdicts even if model scores are low."""
    # Scope mismatch override
    scope_features = {"scope_mismatch": 1.0, "payload_sql_score": 0.0}
    score_scope = calculate_ensemble_score(0.1, 0.1, 0.1, feature_dict=scope_features)
    assert score_scope >= 0.85
    assert determine_verdict(score_scope) == VerdictEnum.BLOCK

    # SQL injection override
    sql_features = {"scope_mismatch": 0.0, "payload_sql_score": 1.0}
    score_sql = calculate_ensemble_score(0.1, 0.1, 0.1, feature_dict=sql_features)
    assert score_sql >= 0.90
    assert determine_verdict(score_sql) == VerdictEnum.BLOCK


def test_security_pipeline_evaluation_and_latency() -> None:
    """Verify end-to-end evaluation produces valid result under 200ms latency budget."""
    pipeline = get_security_pipeline()

    # Normal features
    normal_features = {feat: 0.0 for feat in ml_settings.feature_names}
    res_normal = pipeline.evaluate_features(
        features=normal_features,
        endpoint="/accounts",
        method="GET",
        client_id="tpp_valid_01",
    )
    assert isinstance(res_normal, SecurityEvaluationResult)
    assert res_normal.verdict == VerdictEnum.APPROVE
    assert res_normal.latency_ms < 200.0

    # Attack features (SQL injection)
    attack_features = {feat: 0.0 for feat in ml_settings.feature_names}
    attack_features["payload_sql_score"] = 1.0
    attack_features["endpoint_risk_weight"] = 1.0
    res_attack = pipeline.evaluate_features(
        features=attack_features,
        endpoint="/transfer",
        method="POST",
        client_id="tpp_attacker_01",
    )
    assert res_attack.verdict == VerdictEnum.BLOCK
    assert len(res_attack.explanations) > 0
    assert res_attack.explanations[0].contribution > 0.0
    assert res_attack.latency_ms < 200.0


@pytest.mark.asyncio
async def test_live_mesh_interceptor_with_pipeline_integration() -> None:
    """Verify live HTTP request through interceptor with full ML pipeline active."""
    app = create_app()
    register_pipeline_with_interceptor()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Legitimate compliant consent request -> Must PASS (APPROVE)
        valid_consent_payload = {
            "client_id": "tpp_nayapay_pk_01",
            "tpp_name": "NayaPay Partner App",
            "scopes": ["accounts:read", "transfer:write"],
            "account_ids": ["ACC-PK-0012948"],
            "valid_until": "2026-12-31T23:59:59Z",
        }
        res_valid = await client.post(
            "/consent",
            json=valid_consent_payload,
            headers={
                settings.header_client_id: "tpp_nayapay_pk_01",
                settings.header_consent_scope: "consent:write accounts:read transfer:write",
            },
        )
        assert res_valid.status_code == 201
        assert "x-mesh-latency-ms" in res_valid.headers

        # Malicious SQL injection transfer request -> Must BLOCK (HTTP 403)
        malicious_transfer = {
            "consent_id": "CONSENT-001' OR 1=1 --",
            "debtor_account": "PK36SCBL0000001123456701",
            "creditor_account": "PK36HABB0000009988776602",
            "amount": 5000.0,
            "currency": "PKR",
            "purpose_code": "SALA",
            "narrative": "UNION SELECT password FROM users --",
        }
        res_blocked = await client.post(
            "/transfer",
            json=malicious_transfer,
            headers={
                settings.header_client_id: "tpp_attacker_sql",
                settings.header_consent_id: "CONSENT-001",
                settings.header_consent_scope: "transfer:write",
            },
        )
        assert res_blocked.status_code == 403
        data = res_blocked.json()
        assert data["status"] == "FORBIDDEN"
        assert data["verdict"] == "BLOCK"
        assert len(data["explanations"]) > 0
        assert any(e["feature"] == "payload_sql_score" for e in data["explanations"])
