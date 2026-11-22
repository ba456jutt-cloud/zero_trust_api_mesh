"""Synthetic Normal Traffic Generator for Pakistan Open Banking APIs.

Generates realistic, fully compliant Open Banking interaction traces:
- Complies strictly with the zero real customer data constraint (100% synthetic).
- Follows natural navigation flow: /consent -> /accounts -> /transactions -> /transfer.
- Generates transfer amounts following Benford's Law distribution (PKR 500 - PKR 500,000).
- Carries valid FAPI headers (matching scopes, client IDs, consent IDs).
- Produces training feature vectors labeled as 0 (normal).
"""

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.config import settings
from ml.feature_engineering import calculate_benford_deviation, extract_features_from_request, feature_dict_to_vector
from ml.sliding_window import tracker

# Realistic synthetic Pakistani financial entities (simulated)
SYNTHETIC_CLIENTS: List[str] = [
    "tpp_nayapay_pk_01",
    "tpp_sadapay_pk_02",
    "tpp_easypaisa_api_03",
    "tpp_jazzcash_open_04",
    "tpp_fintech_partner_05",
]

SYNTHETIC_IBANS: List[str] = [
    "PK36SCBL0000001123456701",
    "PK36HABB0000009988776602",
    "PK36MCBB0000005544332203",
    "PK36MEZN0000007788990004",
    "PK36UBL0000003322110005",
]

SYNTHETIC_PURPOSE_CODES: List[str] = ["SALA", "COMM", "UTIL", "EDUC", "RENT"]


@dataclass
class SyntheticHTTPRequest:
    """Represents a simulated HTTP request with headers and payload."""

    endpoint: str
    method: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    timestamp: float
    client_id: str
    account_id: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "headers": self.headers,
            "body": self.body,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
            "account_id": self.account_id,
        }


def generate_normal_request(
    endpoint: Optional[str] = None,
    client_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> SyntheticHTTPRequest:
    """Generate a single legitimate HTTP request adhering to banking protocol."""
    ep = endpoint or random.choice(["/consent", "/accounts", "/transfer"])
    client = client_id or random.choice(SYNTHETIC_CLIENTS)
    t = timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp()
    consent_id = f"CONSENT-PK-{random.randint(10000, 99999)}"
    source_iban = random.choice(SYNTHETIC_IBANS)
    dest_iban = random.choice([i for i in SYNTHETIC_IBANS if i != source_iban])

    if ep == "/consent":
        return SyntheticHTTPRequest(
            endpoint="/consent",
            method="POST",
            headers={
                settings.header_client_id: client,
                settings.header_consent_scope: "consent:write",
            },
            body={
                "client_id": client,
                "tpp_name": "Certified Open Banking TPP",
                "scopes": ["accounts:read", "transfer:write"],
                "account_ids": [source_iban],
                "valid_until": "2027-01-01T00:00:00Z",
            },
            timestamp=t,
            client_id=client,
        )
    elif ep in ("/accounts", "/transactions"):
        return SyntheticHTTPRequest(
            endpoint=ep,
            method="GET",
            headers={
                settings.header_client_id: client,
                settings.header_consent_id: consent_id,
                settings.header_consent_scope: "accounts:read",
            },
            body={},
            timestamp=t,
            client_id=client,
            account_id=source_iban,
        )
    else:  # /transfer
        return SyntheticHTTPRequest(
            endpoint="/transfer",
            method="POST",
            headers={
                settings.header_client_id: client,
                settings.header_consent_id: consent_id,
                settings.header_consent_scope: "transfer:write",
            },
            body={
                "consent_id": consent_id,
                "debtor_account": source_iban,
                "creditor_account": dest_iban,
                "amount": generate_benford_amount(),
                "currency": "PKR",
                "purpose_code": random.choice(SYNTHETIC_PURPOSE_CODES),
                "narrative": "Standard Raast P2P transfer",
            },
            timestamp=t,
            client_id=client,
            account_id=source_iban,
        )



def generate_benford_amount(min_pkr: float = 500.0, max_pkr: float = 500_000.0) -> float:
    """Sample a monetary amount adhering strictly to Benford's Law.

    By sampling log10(X) uniformly across [log10(min), log10(max)],
    the first digit distribution follows P(D=d) = log10(1 + 1/d).
    """
    log_min = math.log10(min_pkr)
    log_max = math.log10(max_pkr)
    random_log = random.uniform(log_min, log_max)
    # Round to realistic 2-decimal financial precision
    amount = round(10.0 ** random_log, 2)
    return amount


def generate_normal_session(
    client_id: Optional[str] = None,
    base_time: Optional[float] = None,
) -> List[SyntheticHTTPRequest]:
    """Generate a single legitimate user session following proper Open Banking flow.

    Simulates realistic human/API pacing with 1-5 second delays between workflow steps.
    """
    client = client_id or random.choice(SYNTHETIC_CLIENTS)
    t = base_time or datetime.now(timezone.utc).timestamp()
    consent_id = f"CONSENT-PK-{random.randint(10000, 99999)}"
    if client in SYNTHETIC_CLIENTS:
        source_iban = SYNTHETIC_IBANS[SYNTHETIC_CLIENTS.index(client) % len(SYNTHETIC_IBANS)]
    else:
        source_iban = random.choice(SYNTHETIC_IBANS)
    dest_iban = random.choice([i for i in SYNTHETIC_IBANS if i != source_iban])

    requests: List[SyntheticHTTPRequest] = []

    # Step 1: /consent (Create Consent)
    requests.append(
        SyntheticHTTPRequest(
            endpoint="/consent",
            method="POST",
            headers={
                settings.header_client_id: client,
                settings.header_consent_scope: "consent:write",
            },
            body={
                "client_id": client,
                "tpp_name": "Certified Open Banking TPP",
                "scopes": ["accounts:read", "transfer:write"],
                "account_ids": [source_iban],
                "valid_until": "2027-01-01T00:00:00Z",
            },
            timestamp=t,
            client_id=client,
        )
    )

    # Step 2: /accounts (Read Accounts) - 1.0 to 5.0s human delay
    t += random.uniform(1.0, 5.0)
    requests.append(
        SyntheticHTTPRequest(
            endpoint="/accounts",
            method="GET",
            headers={
                settings.header_client_id: client,
                settings.header_consent_id: consent_id,
                settings.header_consent_scope: "accounts:read",
            },
            body={},
            timestamp=t,
            client_id=client,
            account_id=source_iban,
        )
    )

    # Step 3: /transfer (Execute Payment with Benford Amount) - 1.0 to 5.0s delay
    t += random.uniform(1.0, 5.0)
    transfer_amount = generate_benford_amount()
    requests.append(
        SyntheticHTTPRequest(
            endpoint="/transfer",
            method="POST",
            headers={
                settings.header_client_id: client,
                settings.header_consent_id: consent_id,
                settings.header_consent_scope: "transfer:write",
            },
            body={
                "consent_id": consent_id,
                "debtor_account": source_iban,
                "creditor_account": dest_iban,
                "amount": transfer_amount,
                "currency": "PKR",
                "purpose_code": random.choice(SYNTHETIC_PURPOSE_CODES),
                "narrative": "Standard Raast P2P transfer",
            },
            timestamp=t,
            client_id=client,
            account_id=source_iban,
        )
    )

    return requests


def generate_normal_dataset(
    num_sessions: int = 150,
    reset_tracker_state: bool = True,
    return_requests: bool = False,
) -> Any:
    """Synthesize labeled normal feature vectors (label=0) for ML training."""
    if reset_tracker_state:
        tracker.clear()

    dataset_requests: List[SyntheticHTTPRequest] = []
    dataset_vectors: List[List[float]] = []
    current_sim_time = 1_700_000_000.0

    for _ in range(num_sessions):
        client = random.choice(SYNTHETIC_CLIENTS)
        session_requests = generate_normal_session(client_id=client, base_time=current_sim_time)
        for req in session_requests:
            features = extract_features_from_request(req)
            vector = feature_dict_to_vector(features)
            dataset_vectors.append(vector)
            dataset_requests.append(req)
            current_sim_time = req.timestamp

        # Advance simulated time with realistic inter-session gap (30s to 300s)
        current_sim_time += random.uniform(30.0, 300.0)

    if return_requests:
        return dataset_requests
    return dataset_vectors


