"""Unit tests for Feature Engineering and Pre-ML Security Checks."""

import pytest
from ml.config import FEATURE_NAMES
from ml.feature_engineering import (
    calculate_benford_deviation,
    check_parameter_tampering,
    check_scope_mismatch,
    check_sql_injection_score,
    extract_features_from_request,
    feature_dict_to_vector,
)


def test_benford_law_deviation() -> None:
    """Verify Benford's Law deviation scores across leading digits."""
    # Digit 1 has highest organic probability (~30.1%) -> lowest surprise
    score_1 = calculate_benford_deviation(1500.0)
    # Digit 2 has ~17.6%
    score_2 = calculate_benford_deviation(2500.0)
    # Digit 9 has lowest organic probability (~4.6%) -> highest surprise
    score_9 = calculate_benford_deviation(9500.0)

    assert score_1 < score_2 < score_9
    assert score_1 < 0.45
    assert score_9 > 0.85

    # Zero or missing amounts should return 0.0
    assert calculate_benford_deviation(None) == 0.0
    assert calculate_benford_deviation(0.0) == 0.0


def test_scope_mismatch_check() -> None:
    """Verify OAuth scope enforcement against endpoint requirements."""
    # Transfer requires transfer:write
    assert check_scope_mismatch("/transfer", "transfer:write") == 0.0
    assert check_scope_mismatch("/transfer", "accounts:read") == 1.0
    assert check_scope_mismatch("/transfer", "") == 1.0
    assert check_scope_mismatch("/transfer", None) == 1.0

    # Accounts requires accounts:read
    assert check_scope_mismatch("/accounts", "accounts:read") == 0.0
    assert check_scope_mismatch("/accounts", "transfer:write") == 1.0


def test_sql_injection_detection() -> None:
    """Verify SQL injection signature scoring."""
    malicious_payload = '{"narrative": "Invoice 101\' OR 1=1 --"}'
    safe_payload = '{"narrative": "Monthly salary settlement"}'

    assert check_sql_injection_score(malicious_payload) >= 0.85
    assert check_sql_injection_score(safe_payload) == 0.0
    assert check_sql_injection_score("") == 0.0


def test_parameter_tampering_bola() -> None:
    """Verify detection of account parameter substitution (BOLA)."""
    headers = {"x-consent-id": "CONSENT-MOCK-001"}

    # Legitimate account tied to mock consent
    safe_body = {"debtor_account": "PK36SCBL0000001123456701"}
    assert check_parameter_tampering(safe_body, headers) == 0.0

    # Tampered account (unauthorized debtor)
    tampered_body = {"debtor_account": "PK36HABB0000009988776602"}
    assert check_parameter_tampering(tampered_body, headers) == 0.90


def test_full_feature_vector_extraction() -> None:
    """Verify end-to-end extraction produces 12 valid normalized features."""
    endpoint = "/transfer"
    method = "POST"
    headers = {
        "x-consent-scope": "transfer:write",
        "x-client-id": "tpp_client_test",
        "x-consent-id": "CONSENT-MOCK-001",
    }
    body = b'{"debtor_account": "PK36SCBL0000001123456701", "amount": 12500.0}'

    features = extract_features_from_request(
        endpoint=endpoint,
        method=method,
        headers=headers,
        body_bytes=body,
    )

    # Must contain all 12 defined features
    assert len(features) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in features
        val = features[name]
        assert 0.0 <= val <= 1.0, f"Feature {name} value {val} out of bounds [0, 1]"

    # Convert to vector
    vector = feature_dict_to_vector(features)
    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vector)


def test_extract_features_uses_request_timestamp() -> None:
    """Verify feature extraction honors simulated timestamps passed in request."""
    from ml.sliding_window import tracker
    tracker.clear()

    req1 = {
        "endpoint": "/consent",
        "method": "POST",
        "headers": {"x-client-id": "client_time_test"},
        "body": {},
        "timestamp": 0.0,
    }
    feats1 = extract_features_from_request(req1)
    # First request: 1 request in 10s window -> 1/100 = 0.01
    assert feats1["request_rate_10s"] == 0.01

    req2 = {
        "endpoint": "/accounts",
        "method": "GET",
        "headers": {"x-client-id": "client_time_test"},
        "body": {},
        "timestamp": 1.0,
    }
    feats2 = extract_features_from_request(req2)
    # Second request 1s later: 2 requests in 10s window -> 2/100 = 0.02
    assert feats2["request_rate_10s"] == 0.02

    # Third request 15s later: t=16.0 (t=0 and t=1 are outside the 10s window)
    req3 = {
        "endpoint": "/transfer",
        "method": "POST",
        "headers": {"x-client-id": "client_time_test"},
        "body": {},
        "timestamp": 16.0,
    }
    feats3 = extract_features_from_request(req3)
    assert feats3["request_rate_10s"] == 0.01


