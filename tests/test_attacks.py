"""Unit tests for Synthetic Attack Traffic Generator covering all 7 attack vectors."""

import pytest
from data.attack_traffic import (
    ATTACK_VECTOR_NAMES,
    generate_all_attacks_dataset,
    generate_api_flooding_attack,
    generate_bola_attack,
    generate_bruteforce_login_attack,
    generate_credential_stuffing_attack,
    generate_parameter_tampering_attack,
    generate_sql_injection_attack,
    generate_token_replay_attack,
)


def test_seven_attack_vectors_completeness() -> None:
    """Verify that all 7 required SBP attack vectors are defined."""
    expected_vectors = {
        "sql_injection",
        "parameter_tampering",
        "brute_force_login",
        "credential_stuffing",
        "bola",
        "api_flooding",
        "token_replay",
    }
    assert set(ATTACK_VECTOR_NAMES) == expected_vectors


def test_credential_stuffing_exact_burst_size() -> None:
    """Verify credential stuffing generates exactly a burst of 20 distinct targets."""
    requests = generate_credential_stuffing_attack()
    assert len(requests) == 20
    distinct_accounts = {req.account_id for req in requests}
    assert len(distinct_accounts) == 20


def test_api_flooding_exact_burst_size() -> None:
    """Verify API flooding generates a burst of 100 requests in 10s window."""
    requests = generate_api_flooding_attack()
    assert len(requests) == 100
    duration = requests[-1].timestamp - requests[0].timestamp
    assert duration <= 10.0, f"Burst duration {duration:.2f}s exceeded 10.0s window"


def test_sql_injection_patterns() -> None:
    """Verify SQL injection payload generation."""
    requests = generate_sql_injection_attack()
    assert len(requests) >= 1
    narrative = requests[0].body.get("narrative", "")
    assert any(tok in narrative.lower() for tok in ["' or 1=1", "union select", "drop table"])


def test_bola_unauthorized_account() -> None:
    """Verify BOLA payload targets an unauthorized debtor account."""
    requests = generate_bola_attack()
    assert len(requests) >= 1
    # BOLA tries to drain account not matching the consent's primary ownership
    debtor = requests[0].body.get("debtor_account")
    assert debtor == "PK36HABB0000009988776602"


def test_all_attacks_dataset_generation() -> None:
    """Verify master attack dataset generation produces labeled vectors for all 7 vectors."""
    vectors, labels, attack_types = generate_all_attacks_dataset(reset_tracker_state=True)

    assert len(vectors) > 0
    assert len(vectors) == len(labels) == len(attack_types)

    # Every vector has 12 features
    for v in vectors:
        assert len(v) == 12

    # Every label is 1 (attack)
    assert all(lbl == 1 for lbl in labels)

    # Every attack vector must be represented
    unique_types = set(attack_types)
    assert unique_types == set(ATTACK_VECTOR_NAMES)

