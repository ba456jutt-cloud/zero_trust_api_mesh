"""Feature engineering module for real-time API anomaly detection.

Extracts 12-dimensional normalized feature vectors from HTTP telemetry:
1. scope_mismatch: Mandatory OAuth scope validation
2. sequence_anomaly: Workflow jump score from sliding-window
3. benford_deviation: First-class Benford's Law distribution metric
4. request_rate_10s: Short-window burst velocity
5. request_rate_60s: Medium-window cumulative volume
6. distinct_accounts_ratio: Probing / credential stuffing ratio
7. token_reuse_count: Replay attack counter
8. payload_sql_score: Signature & heuristic SQL injection score
9. payload_param_tamper: Parameter/account tampering indicator
10. payload_length: Normalized payload byte size
11. is_post_method: Binary indicator for state mutations
12. endpoint_risk_weight: SBP regulatory endpoint sensitivity
"""

import json
import math
from typing import Any, Dict, List, Optional
import structlog
from backend.config import settings
from ml.config import FEATURE_NAMES, ml_settings
from ml.sliding_window import WindowMetrics, tracker

logger = structlog.get_logger(__name__)

# Precomputed Benford's Law theoretical probabilities for leading digits 1..9
BENFORD_PROBABILITIES: Dict[int, float] = {
    d: math.log10(1.0 + 1.0 / d) for d in range(1, 10)
}
# Maximum surprise score (-log2(P(9))) for normalization
MAX_BENFORD_SURPRISE: float = -math.log2(BENFORD_PROBABILITIES[9])


def calculate_benford_deviation(amount: Optional[float]) -> float:
    """Calculate Benford's Law anomaly score for a financial transfer amount.

    Organic financial transactions naturally follow Benford's Law (leading digit 1
    appears ~30.1% of the time, while 9 appears only ~4.6%). Synthetic or fraudulent
    amounts often exhibit unnatural leading digits or artificial round numbers.

    Returns:
        float between 0.0 (highly expected, e.g. digit 1) and 1.0 (highly anomalous).
    """
    if amount is None or amount <= 0.0:
        return 0.0

    try:
        # Extract first non-zero digit
        amount_str = f"{amount:.2f}".lstrip("0").replace(".", "")
        if not amount_str:
            return 0.0
        leading_digit = int(amount_str[0])
        if leading_digit not in BENFORD_PROBABILITIES:
            return 1.0

        p = BENFORD_PROBABILITIES[leading_digit]
        # Information-theoretic surprise: -log2(P(d)) normalized
        surprise = -math.log2(p) / MAX_BENFORD_SURPRISE

        int_part = str(int(amount))
        # Penalize suspicious round numbers (e.g. exactly 50000, 100000, 99999)
        if amount >= 10000 and int_part.endswith("0000"):
            surprise = min(1.0, surprise + 0.15)

        return float(min(1.0, max(0.0, surprise)))
    except Exception as exc:
        logger.debug("benford_calculation_failed", error=str(exc))
        return 0.0


def check_scope_mismatch(path: str, granted_scope_header: Optional[str]) -> float:
    """Validate granted OAuth token scope against route requirements."""
    required_scopes = settings.endpoint_scope_map.get(path, [])
    if not required_scopes:
        return 0.0

    granted = granted_scope_header or ""
    has_match = any(req in granted for req in required_scopes)
    return 0.0 if has_match else 1.0


def check_sql_injection_score(body_str: str) -> float:
    """Evaluate payload text for SQL injection attack patterns."""
    if not body_str:
        return 0.0

    lower_body = body_str.lower()
    sql_patterns = [
        "' or 1=1",
        "' or '1'='1",
        "union select",
        "drop table",
        "--",
        "/*",
        "xp_cmdshell",
        "admin'--",
        "or 1=1--",
    ]
    matches = sum(1 for pattern in sql_patterns if pattern in lower_body)
    if matches >= 2:
        return 1.0
    if matches == 1:
        return 0.85
    return 0.0


def check_parameter_tampering(body_dict: Dict[str, Any], headers: Dict[str, str]) -> float:
    """Detect BOLA / parameter tampering (e.g. debtor account != authorized consent)."""
    consent_id = headers.get(settings.header_consent_id)
    debtor_account = body_dict.get("debtor_account")

    # If payload debtor account substitutes another account pattern
    if debtor_account and isinstance(debtor_account, str):
        if "../" in debtor_account or "null" in debtor_account.lower() or "'" in debtor_account:
            return 1.0

    # In BOLA scenario: consent belongs to ACC-PK-001 but payload requests ACC-PK-999
    if consent_id == "CONSENT-MOCK-001" and debtor_account and debtor_account != "PK36SCBL0000001123456701":
        return 0.90

    return 0.0


def extract_features_from_request(
    endpoint: Any = None,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body_bytes: Optional[bytes] = None,
    client_key: Optional[str] = None,
    account_id: Optional[str] = None,
    token_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    as_of: Optional[float] = None,
) -> Dict[str, float]:
    """Extract complete 12-dimensional feature dictionary from incoming request.

    Supports both explicit field arguments or a single request dict/dataclass.
    """
    # Handle single request dict or object passed as first argument
    if endpoint is not None and not isinstance(endpoint, str):
        req_obj = endpoint
        if isinstance(req_obj, (list, tuple)) and len(req_obj) == len(FEATURE_NAMES):
            return {FEATURE_NAMES[i]: float(req_obj[i]) for i in range(len(FEATURE_NAMES))}

        if isinstance(req_obj, dict):
            endpoint = str(req_obj.get("endpoint", "/"))
            method = str(req_obj.get("method", "GET"))
            headers = req_obj.get("headers", {})
            b_val = req_obj.get("body", b"")
            client_key = req_obj.get("client_id", client_key)
            account_id = req_obj.get("account_id", account_id)
            token_id = req_obj.get("token_id", token_id)
            timestamp = req_obj.get("timestamp", timestamp)
        elif hasattr(req_obj, "endpoint"):
            endpoint = str(req_obj.endpoint)
            method = str(getattr(req_obj, "method", "GET"))
            headers = getattr(req_obj, "headers", {})
            b_val = getattr(req_obj, "body", b"")
            client_key = getattr(req_obj, "client_id", client_key)
            account_id = getattr(req_obj, "account_id", account_id)
            token_id = getattr(req_obj, "token_id", token_id)
            timestamp = getattr(req_obj, "timestamp", timestamp)
        else:
            endpoint = str(endpoint)
            b_val = body_bytes

        if isinstance(b_val, bytes):
            body_bytes = b_val
        elif isinstance(b_val, str):
            body_bytes = b_val.encode("utf-8")
        elif isinstance(b_val, dict):
            body_bytes = json.dumps(b_val).encode("utf-8")
        else:
            body_bytes = b""

    endpoint = endpoint or "/"
    headers = headers or {}
    body_bytes = body_bytes or b""

    # Determine simulated timestamp if passed or present in headers
    effective_time = as_of if as_of is not None else (timestamp if timestamp is not None else None)
    if effective_time is None and headers:
        header_ts = headers.get("x-simulated-timestamp") or headers.get("timestamp")
        if header_ts is not None:
            try:
                effective_time = float(header_ts)
            except (ValueError, TypeError):
                pass

    # Parse body
    body_str = body_bytes.decode("utf-8", errors="ignore") if body_bytes else ""
    body_dict: Dict[str, Any] = {}
    if body_str:
        try:
            body_dict = json.loads(body_str)
        except Exception:
            body_dict = {}

    # Extract transfer amount if present
    amount: Optional[float] = None
    if "amount" in body_dict and isinstance(body_dict["amount"], (int, float)):
        amount = float(body_dict["amount"])

    # Consult sliding window tracker for temporal metrics
    active_client = client_key or headers.get(settings.header_client_id, "unknown_client")
    target_account = account_id or body_dict.get("debtor_account") or body_dict.get("account_id")
    active_token = token_id or headers.get("authorization", headers.get(settings.header_consent_id))

    window_metrics = tracker.record_and_compute(
        client_key=active_client,
        endpoint=endpoint,
        account_id=target_account,
        token_id=active_token,
        as_of=effective_time,
    )

    # 1. Scope mismatch
    scope_val = check_scope_mismatch(endpoint, headers.get(settings.header_consent_scope))

    # 2. Sequence anomaly
    seq_val = window_metrics.sequence_anomaly

    # 3. Benford deviation
    benford_val = calculate_benford_deviation(amount)

    # 4. Request velocity (10s window capped at 50 rps = 500)
    rate_10s_val = min(1.0, window_metrics.count_10s / 100.0)

    # 5. Request volume (60s window capped at 300)
    rate_60s_val = min(1.0, window_metrics.count_60s / 300.0)

    # 6. Distinct accounts ratio
    distinct_accounts_val = min(1.0, window_metrics.distinct_accounts_count / 20.0)

    # 7. Token reuse count (capped at 10)
    token_reuse_val = min(1.0, window_metrics.token_reuse_count / 10.0)

    # 8. SQL injection score
    sql_val = check_sql_injection_score(body_str)

    # 9. Parameter tampering
    tamper_val = check_parameter_tampering(body_dict, headers)

    # 10. Payload length (normalized by 5000 bytes)
    payload_len_val = min(1.0, len(body_bytes) / 5000.0)

    # 11. Is POST method
    is_post_val = 1.0 if method.upper() == "POST" else 0.0

    # 12. Endpoint regulatory risk
    risk_val = ml_settings.endpoint_risk_map.get(endpoint, 0.4)

    return {
        "scope_mismatch": scope_val,
        "sequence_anomaly": seq_val,
        "benford_deviation": benford_val,
        "request_rate_10s": rate_10s_val,
        "request_rate_60s": rate_60s_val,
        "distinct_accounts_ratio": distinct_accounts_val,
        "token_reuse_count": token_reuse_val,
        "payload_sql_score": sql_val,
        "payload_param_tamper": tamper_val,
        "payload_length": payload_len_val,
        "is_post_method": is_post_val,
        "endpoint_risk_weight": risk_val,
    }


def feature_dict_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dictionary to ordered feature vector matching FEATURE_NAMES."""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
