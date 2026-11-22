"""Synthetic Attack Traffic Generator for Pakistan Open Banking APIs.

Generates all 7 mandatory attack vectors defined in SBP Sandbox specifications:
1. SQL Injection: Payloads containing SQL syntax in narrative or query params
2. Parameter Tampering: Manipulating parameters (path traversal, negative amounts)
3. Brute-Force Login: Rapid sequential authentication requests
4. Credential Stuffing: Burst of 20 distinct account IDs targeted by one origin
5. BOLA: Broken Object Level Authorization (cross-account unauthorized access)
6. API Flooding: Burst of 100 requests in a short 10s window
7. Token Replay: Repeated reuse of the same authorization token/JTI

Outputs labeled feature vectors (label=1) for supervised evaluation & training.
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from backend.config import settings
from data.normal_traffic import SYNTHETIC_IBANS, SyntheticHTTPRequest
from ml.feature_engineering import extract_features_from_request, feature_dict_to_vector
from ml.sliding_window import tracker

ATTACK_VECTOR_NAMES: List[str] = [
    "sql_injection",
    "parameter_tampering",
    "brute_force_login",
    "credential_stuffing",
    "bola",
    "api_flooding",
    "token_replay",
]


def generate_sql_injection_attack(
    client_id: str = "tpp_attacker_sqli",
    base_time: float = 1_700_000_000.0,
) -> List[SyntheticHTTPRequest]:
    """Generate request payload with SQL injection attack patterns."""
    sqli_payloads = [
        "Vendor Payment' OR 1=1 --",
        "Salary' UNION SELECT username, password FROM users --",
        "Rent 2026'; DROP TABLE transactions; --",
    ]
    chosen_sqli = random.choice(sqli_payloads)
    return [
        SyntheticHTTPRequest(
            endpoint="/transfer",
            method="POST",
            headers={
                settings.header_client_id: client_id,
                settings.header_consent_id: "CONSENT-MOCK-001",
                settings.header_consent_scope: "transfer:write",
            },
            body={
                "consent_id": "CONSENT-MOCK-001",
                "debtor_account": "PK36SCBL0000001123456701",
                "creditor_account": "PK36HABB0000009988776602",
                "amount": 5000.0,
                "currency": "PKR",
                "purpose_code": "SALA",
                "narrative": chosen_sqli,
            },
            timestamp=base_time,
            client_id=client_id,
        )
    ]


def generate_parameter_tampering_attack(
    client_id: str = "tpp_attacker_tamper",
    base_time: float = 1_700_000_000.0,
) -> List[SyntheticHTTPRequest]:
    """Generate request with path traversal or tampered account identifier."""
    tampered_accounts = [
        "../ACC-PK-001",
        "PK36SCBL0000001123456701' OR '1'='1",
        "NULL_ACCOUNT_ID",
    ]
    return [
        SyntheticHTTPRequest(
            endpoint="/transfer",
            method="POST",
            headers={
                settings.header_client_id: client_id,
                settings.header_consent_id: "CONSENT-MOCK-001",
                settings.header_consent_scope: "transfer:write",
            },
            body={
                "consent_id": "CONSENT-MOCK-001",
                "debtor_account": random.choice(tampered_accounts),
                "creditor_account": "PK36HABB0000009988776602",
                "amount": 99999.0,  # Unnatural round number (violates Benford)
                "currency": "PKR",
                "purpose_code": "COMM",
                "narrative": "Tampered transfer attempt",
            },
            timestamp=base_time,
            client_id=client_id,
        )
    ]


def generate_bruteforce_login_attack(
    client_id: str = "tpp_attacker_bruteforce",
    base_time: float = 1_700_000_000.0,
) -> List[SyntheticHTTPRequest]:
    """Generate high-frequency repeated authentication requests (15 requests in 1.5s)."""
    requests: List[SyntheticHTTPRequest] = []
    for i in range(15):
        requests.append(
            SyntheticHTTPRequest(
                endpoint="/consent",
                method="POST",
                headers={
                    settings.header_client_id: client_id,
                    settings.header_consent_scope: "invalid_auth_scope",
                },
                body={
                    "client_id": client_id,
                    "tpp_name": "Brute Force Attacker",
                    "scopes": ["admin:superuser"],
                    "account_ids": ["ACC-TARGET"],
                    "valid_until": "2026-12-31T23:59:59Z",
                },
                timestamp=base_time + (i * 0.1),
                client_id=client_id,
            )
        )
    return requests


def generate_credential_stuffing_attack(
    client_id: str = "tpp_attacker_stuffing",
    base_time: float = 1_700_000_000.0,
) -> List[SyntheticHTTPRequest]:
    """Generate a burst of exactly 20 distinct target account requests from one origin."""
    requests: List[SyntheticHTTPRequest] = []
    for i in range(20):
        target_account = f"ACC-PK-TARGET-{i:03d}"
        requests.append(
            SyntheticHTTPRequest(
                endpoint="/accounts",
                method="GET",
                headers={
                    settings.header_client_id: client_id,
                    settings.header_consent_id: f"CONSENT-STUFF-{i:03d}",
                    settings.header_consent_scope: "accounts:read",
                },
                body={},
                timestamp=base_time + (i * 0.2),  # 20 accounts in 4 seconds
                client_id=client_id,
                account_id=target_account,
            )
        )
    return requests


def generate_bola_attack(
    client_id: str = "tpp_attacker_bola",
    base_time: float = 1_700_000_000.0,
) -> List[SyntheticHTTPRequest]:
    """Generate BOLA attack: accesses Account B using consent belonging to Account A."""
    # Step 1: Normal consent for PK36SCBL0000001123456701
    # Step 2: Immediate transfer attempting to drain another customer's account PK36HABB0000009988776602
    return [
        SyntheticHTTPRequest(
            endpoint="/transfer",
            method="POST",
            headers={
                settings.header_client_id: client_id,
                settings.header_consent_id: "CONSENT-MOCK-001",
                settings.header_consent_scope: "transfer:write",
            },
            body={
                "consent_id": "CONSENT-MOCK-001",
                "debtor_account": "PK36HABB0000009988776602",  # Not owner's account
                "creditor_account": "PK36MEZN0000007788990004",
                "amount": 250_000.0,
                "currency": "PKR",
                "purpose_code": "COMM",
                "narrative": "Unauthorized cross-tenant drain",
            },
            timestamp=base_time,
            client_id=client_id,
            account_id="PK36HABB0000009988776602",
        )
    ]


def generate_api_flooding_attack(
    client_id: str = "tpp_attacker_flood",
    base_time: float = 1_700_000_000.0,
) -> List[SyntheticHTTPRequest]:
    """Generate high-velocity burst of 100 requests in a 10-second window."""
    requests: List[SyntheticHTTPRequest] = []
    for i in range(100):
        requests.append(
            SyntheticHTTPRequest(
                endpoint="/accounts",
                method="GET",
                headers={
                    settings.header_client_id: client_id,
                    settings.header_consent_id: "CONSENT-MOCK-001",
                    settings.header_consent_scope: "accounts:read",
                },
                body={},
                timestamp=base_time + (i * 0.08),  # 100 requests in 8 seconds
                client_id=client_id,
            )
        )
    return requests


def generate_token_replay_attack(
    client_id: str = "tpp_attacker_replay",
    base_time: float = 1_700_000_000.0,
) -> List[SyntheticHTTPRequest]:
    """Generate rapid re-use of identical OAuth token/JTI across 10 requests."""
    replayed_token = "replayed_jti_token_pk_9999"
    requests: List[SyntheticHTTPRequest] = []
    for i in range(10):
        requests.append(
            SyntheticHTTPRequest(
                endpoint="/transfer",
                method="POST",
                headers={
                    settings.header_client_id: client_id,
                    settings.header_consent_id: replayed_token,
                    settings.header_consent_scope: "transfer:write",
                    "authorization": f"Bearer {replayed_token}",
                },
                body={
                    "consent_id": replayed_token,
                    "debtor_account": "PK36SCBL0000001123456701",
                    "creditor_account": "PK36HABB0000009988776602",
                    "amount": 1000.0,
                    "currency": "PKR",
                    "purpose_code": "SALA",
                    "narrative": "Replay attack transaction",
                },
                timestamp=base_time + (i * 0.3),
                client_id=client_id,
            )
        )
    return requests


def generate_all_attacks_dataset(
    reset_tracker_state: bool = True,
) -> Tuple[List[List[float]], List[int], List[str]]:
    """Synthesize labeled attack feature vectors (label=1) for all 7 attack vectors.

    Returns:
        Tuple of (feature_vectors, labels, attack_names)
    """
    if reset_tracker_state:
        tracker.clear()

    vectors: List[List[float]] = []
    labels: List[int] = []
    attack_types: List[str] = []

    generators = [
        ("sql_injection", generate_sql_injection_attack),
        ("parameter_tampering", generate_parameter_tampering_attack),
        ("brute_force_login", generate_bruteforce_login_attack),
        ("credential_stuffing", generate_credential_stuffing_attack),
        ("bola", generate_bola_attack),
        ("api_flooding", generate_api_flooding_attack),
        ("token_replay", generate_token_replay_attack),
    ]

    current_sim_time = 1_800_000_000.0

    for attack_name, gen_func in generators:
        # Generate multiple batches per attack vector for robust dataset coverage
        for batch_idx in range(10):
            client = f"{attack_name}_bot_{batch_idx}"
            attack_requests = gen_func(client_id=client, base_time=current_sim_time)
            for req in attack_requests:
                body_bytes = json.dumps(req.body).encode("utf-8") if req.body else b""
                features = extract_features_from_request(
                    endpoint=req.endpoint,
                    method=req.method,
                    headers=req.headers,
                    body_bytes=body_bytes,
                    client_key=req.client_id,
                    account_id=req.account_id,
                    token_id=req.headers.get("authorization", req.headers.get(settings.header_consent_id)),
                )
                vec = feature_dict_to_vector(features)
                vectors.append(vec)
                labels.append(1)
                attack_types.append(attack_name)

            current_sim_time += 100.0

    return vectors, labels, attack_types

