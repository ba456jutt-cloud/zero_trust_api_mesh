"""Unit tests for Synthetic Normal Traffic Generator."""

from collections import Counter
import pytest
from data.normal_traffic import (
    generate_benford_amount,
    generate_normal_dataset,
    generate_normal_session,
)
from ml.config import FEATURE_NAMES


def test_benford_law_amount_distribution() -> None:
    """Verify that sampled amounts empirically follow Benford's Law."""
    sample_size = 1500
    amounts = [generate_benford_amount() for _ in range(sample_size)]

    leading_digits = [int(str(f"{a:.2f}").lstrip("0").replace(".", "")[0]) for a in amounts]
    counts = Counter(leading_digits)

    # Digit 1 should be the most common (~30%)
    freq_1 = counts[1] / sample_size
    # Digit 9 should be the least common (~4.6%)
    freq_9 = counts[9] / sample_size

    assert 0.22 <= freq_1 <= 0.38, f"Digit 1 frequency {freq_1:.3f} deviated from Benford ~0.301"
    assert 0.02 <= freq_9 <= 0.09, f"Digit 9 frequency {freq_9:.3f} deviated from Benford ~0.046"
    assert counts[1] > counts[2] > counts[9]


def test_normal_session_conformance() -> None:
    """Verify normal session follows Open Banking sequential navigation and valid scopes."""
    session = generate_normal_session(client_id="tpp_verified_01")
    assert len(session) == 3

    endpoints = [req.endpoint for req in session]
    assert endpoints == ["/consent", "/accounts", "/transfer"]

    # Scopes must match endpoint requirements
    assert session[0].headers["x-consent-scope"] == "consent:write"
    assert session[1].headers["x-consent-scope"] == "accounts:read"
    assert session[2].headers["x-consent-scope"] == "transfer:write"

    # All timestamps must be strictly monotonic
    assert session[0].timestamp < session[1].timestamp < session[2].timestamp


def test_normal_dataset_features_cleanliness() -> None:
    """Verify generated normal feature vectors have zero false positive anomalies."""
    vectors = generate_normal_dataset(num_sessions=15, reset_tracker_state=True)
    assert len(vectors) == 45  # 15 sessions * 3 requests

    idx_scope = FEATURE_NAMES.index("scope_mismatch")
    idx_seq = FEATURE_NAMES.index("sequence_anomaly")
    idx_sql = FEATURE_NAMES.index("payload_sql_score")
    idx_tamper = FEATURE_NAMES.index("payload_param_tamper")

    for v in vectors:
        assert len(v) == 12
        # Normal traffic must not have scope mismatches, sequence jumps, or malicious payloads
        assert v[idx_scope] == 0.0
        assert v[idx_seq] == 0.0
        assert v[idx_sql] == 0.0
        assert v[idx_tamper] == 0.0

