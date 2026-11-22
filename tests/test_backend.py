"""Unit tests for Backend Configuration and Data Models."""

import pytest
from pydantic import ValidationError
from backend.config import VerdictEnum, get_settings
from backend.models import (
    AccountInfo,
    ConsentCreateRequest,
    SecurityBlockResponse,
    SecurityEvaluationResult,
    SHAPExplanation,
    TransferRequest,
)


def test_settings_initialization() -> None:
    """Verify core SBP sandbox constraints in settings."""
    settings = get_settings()
    assert settings.target_latency_ms == 200.0
    assert settings.approve_threshold == 0.4
    assert settings.block_threshold == 0.7
    assert settings.max_pattern_entries == 10000
    assert "/transfer" in settings.endpoint_scope_map
    assert "transfer:write" in settings.endpoint_scope_map["/transfer"]


def test_shap_explanation_schema() -> None:
    """Verify SHAP local explanation structure."""
    explanation = SHAPExplanation(
        feature="scope_mismatch",
        contribution=0.45,
        description="OAuth scope does not permit write operation on /transfer",
    )
    assert explanation.feature == "scope_mismatch"
    assert explanation.contribution == 0.45


def test_security_evaluation_result() -> None:
    """Verify security evaluation result container."""
    result = SecurityEvaluationResult(
        verdict=VerdictEnum.BLOCK,
        anomaly_score=0.88,
        latency_ms=45.2,
        endpoint="/transfer",
        http_method="POST",
        explanations=[
            SHAPExplanation(feature="scope_mismatch", contribution=0.52),
        ],
    )
    assert result.verdict == VerdictEnum.BLOCK
    assert result.anomaly_score == 0.88
    assert result.latency_ms < 200.0
    assert len(result.explanations) == 1


def test_transfer_validation() -> None:
    """Verify validation rules on financial transfers (e.g. positive amounts)."""
    valid_transfer = TransferRequest(
        consent_id="CONSENT-1029",
        debtor_account="PK36SCBL0000001123456701",
        creditor_account="PK36HABB0000009988776602",
        amount=15000.0,
    )
    assert valid_transfer.amount == 15000.0

    # Ensure negative amounts are rejected
    with pytest.raises(ValidationError):
        TransferRequest(
            consent_id="CONSENT-1029",
            debtor_account="PK36SCBL0000001123456701",
            creditor_account="PK36HABB0000009988776602",
            amount=-500.0,
        )

