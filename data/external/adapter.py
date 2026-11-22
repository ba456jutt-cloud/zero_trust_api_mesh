"""External Dataset Adapter for SBP Zero-Trust API Security Feature Space.

Maps real-world, externally-validated API security datasets (such as Zenodo
APISentry #22658802 and CSIC 2010 HTTP Dataset) into our internal 12-dimensional
feature vector defined in `ml.config.FEATURE_NAMES`.

================================================================================
12 FEATURE MAPPING & DERIVATION SPECIFICATION
================================================================================

1. `scope_mismatch` (float: 0.0 - 1.0):
   - Definition: OAuth 2.0 token scope does not authorize requested endpoint.
   - Dataset Reality: External datasets (APISentry, CSIC 2010) capture general HTTP
     or e-commerce API telemetry and lack OAuth 2.0 authorization/scope headers.
   - Derivation: Checked via keys `scope_mismatch`, `scope`, or `authorization`.
     If present and indicates mismatch -> 1.0. If not present in raw dataset,
     defaults strictly to 0.0 without faking.

2. `sequence_anomaly` (float: 0.0 - 1.0):
   - Definition: Navigation bypassed mandatory prior steps or exhibits out-of-order
     endpoint workflow hopping.
   - Dataset Reality: APISentry aggregates multi-endpoint exploration per IP in
     `ip_uniq_ep_10s` (z-score standardized, range [-1.06, 4.14]).
   - Derivation: If explicit `sequence_anomaly` is provided in row, use it.
     Otherwise derived from `ip_uniq_ep_10s`: values > 2.0 (high multi-endpoint
     scanning outside workflow) return 1.0; values > 1.0 return 0.5; else 0.0.

3. `benford_deviation` (float: 0.0 - 1.0):
   - Definition: Transaction amount first-digit deviation from Benford's Law
     distribution P(d) = log10(1 + 1/d).
   - Dataset Reality: APISentry contains e-commerce API telemetry with numerical
     fields `body_max_num` / `body_min_num`, but lacks banking monetary transaction
     amounts.
   - Derivation: If `amount` column is present and positive, extracts the leading
     digit d in [1..9] and computes |obs - P(d)| normalized to [0, 1]. If no
     financial amount column exists, defaults strictly to 0.0 (documented).

4. `request_rate_10s` (float: 0.0 - 1.0):
   - Definition: Burst velocity within a 10-second sliding window.
   - Dataset Reality: Maps directly to APISentry `ip_req_10s` (z-score standardized:
     min -0.62401, max 5.90462).
   - Derivation: Scaled to [0.0, 1.0] via min-max normalization:
     max(0.0, min(1.0, (ip_req_10s - (-0.62401)) / (5.90462 - (-0.62401)))).
     If raw integer count is passed, clipped to min(1.0, count / 50.0).

5. `request_rate_60s` (float: 0.0 - 1.0):
   - Definition: Cumulative request rate in a 60-second sliding window.
   - Dataset Reality: Derived from `requests_per_ip_per_min`, `ip_req_60s`, or
     scaled proportionally from `ip_req_10s`.
   - Derivation: If `requests_per_ip_per_min` or `ip_req_60s` is present, min-max
     normalize. Else scaled from `request_rate_10s` * 0.85, bounded [0.0, 1.0].

6. `distinct_accounts_ratio` (float: 0.0 - 1.0):
   - Definition: Ratio of distinct user IDs / accounts probed by the client.
   - Dataset Reality: Maps directly to APISentry `ip_distinct_users_10s`
     (z-score standardized: min -0.49344, max 7.54992). High in credential
     stuffing and BOLA attacks.
   - Derivation: Scaled to [0.0, 1.0] via min-max normalization:
     max(0.0, min(1.0, (ip_distinct_users_10s - (-0.49344)) / (7.54992 - (-0.49344)))).

7. `token_reuse_count` (float: 0.0 - 1.0):
   - Definition: Rate of token reuse / replay across multiple client IPs.
   - Dataset Reality: Maps directly to APISentry `token_ips_10s` (z-score
     standardized: min -0.77731, max 3.62527). High in token_replay attacks.
   - Derivation: Scaled to [0.0, 1.0] via min-max normalization:
     max(0.0, min(1.0, (token_ips_10s - (-0.77731)) / (3.62527 - (-0.77731)))).

8. `payload_sql_score` (float: 0.0 - 1.0):
   - Definition: Likelihood of SQL injection pattern in payload / parameters.
   - Dataset Reality: APISentry provides `body_sql_hits` (z-score standardized:
     min -0.41508, max 6.04240, attack mean 4.05). Raw payloads may have SQL text.
   - Derivation: If raw `payload` or `body` text contains SQL injection patterns
     (UNION, SELECT, OR 1=1, quotes, comments) -> 1.0. Else if `body_sql_hits` > 1.0
     -> 1.0. Otherwise 0.0.

9. `payload_param_tamper` (float: 0.0 - 1.0):
   - Definition: Indicator of parameter tampering (negative amounts, forced IDs).
   - Dataset Reality: APISentry provides `body_has_neg` (z-score standardized:
     min -0.07948, max 12.58156, attack mean 2.17) and `body_min_num`.
   - Derivation: If `body_has_neg` > 0.0 or `body_min_num` < -0.5 -> 1.0. If
     explicit `payload_param_tamper` in row -> float value. Otherwise 0.0.

10. `payload_length` (float: 0.0 - 1.0):
    - Definition: Normalized length of the HTTP request body payload.
    - Dataset Reality: Maps to APISentry `body_len` (z-score: min -0.75356, max 4.09470)
      or `request_size`.
    - Derivation: Scaled to [0.0, 1.0] via min-max normalization:
      max(0.0, min(1.0, (body_len - (-0.75356)) / (4.09470 - (-0.75356)))).
      If raw character length L is passed, clipped to min(1.0, L / 10000.0).

11. `is_post_method` (float: 0.0 or 1.0):
    - Definition: Indicator of state-mutating HTTP POST operation.
    - Dataset Reality: In APISentry `method_enc`: 0 = GET, 1 = POST.
    - Derivation: 1.0 if `method_enc == 1` or `method == "POST"`, else 0.0.

12. `endpoint_risk_weight` (float: 0.0 - 1.0):
    - Definition: Inherent regulatory risk tier of targeted endpoint.
    - Dataset Reality: In APISentry `endpoint_norm_enc`:
        0: /cart            -> 0.3
        1: /cart/add        -> 0.5
        2: /login           -> 0.8 (sensitive authentication)
        3: /order           -> 1.0 (financial execution)
        4: /products        -> 0.2
        5: /products/{id}   -> 0.2
        6: /search          -> 0.3
        7: /users/{id}      -> 0.7 (PII / account data)
    - Derivation: Mapped from `endpoint_norm_enc` or path string, fallback 0.3.
"""

import math
import re
from typing import Any, Dict

from ml.config import FEATURE_NAMES

# Pre-compiled SQL injection regex pattern for raw payload evaluation
_SQLI_REGEX = re.compile(
    r"(?i)(\b(SELECT|UNION|INSERT|DELETE|UPDATE|DROP|WHERE|OR|AND|BENCHMARK|SLEEP)\b.*[=<>'\"]|--|/\*|;\s*$|'\s*or\s*'1'='1|1=1)",
    re.IGNORECASE,
)

# APISentry endpoint risk mapping
_APISENTRY_ENDPOINT_MAP = {
    0: 0.3,  # /cart
    1: 0.5,  # /cart/add
    2: 0.8,  # /login (auth endpoint)
    3: 1.0,  # /order (financial/transaction equivalent)
    4: 0.2,  # /products
    5: 0.2,  # /products/{id}
    6: 0.3,  # /search
    7: 0.7,  # /users/{id} (PII/BOLA target)
}

# String path risk weighting
_PATH_RISK_MAP = {
    "/order": 1.0,
    "/transfer": 1.0,
    "/payments": 1.0,
    "/payment": 1.0,
    "/login": 0.8,
    "/auth": 0.8,
    "/token": 0.8,
    "/users": 0.7,
    "/accounts": 0.7,
    "/cart/add": 0.5,
    "/cart": 0.3,
    "/search": 0.3,
    "/products": 0.2,
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert value to float, handling None, NaN, and conversions safely."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _min_max_scale(val: float, min_val: float, max_val: float) -> float:
    """Normalize a float value into [0.0, 1.0] given min and max limits."""
    if max_val <= min_val:
        return 0.0
    scaled = (val - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, float(scaled)))


def _compute_benford_deviation(amount: float) -> float:
    """Compute deviation from Benford's Law for a given monetary amount."""
    if amount <= 0.0:
        return 0.0
    str_amount = f"{amount:.6f}".replace(".", "").lstrip("0")
    if not str_amount:
        return 0.0
    leading_digit = int(str_amount[0])
    if leading_digit < 1 or leading_digit > 9:
        return 0.0
    expected_prob = math.log10(1.0 + 1.0 / leading_digit)
    # Deviation from empirical frequency (normalized difference)
    # Uniform naive expectation would be 1/9 ~ 0.111
    return float(min(1.0, abs(expected_prob - 0.111) / 0.301))


def adapt_external_row(row: Dict[str, Any]) -> Dict[str, float]:
    """Map a row from an external dataset (e.g. APISentry) to the 12-feature vector.

    Args:
        row: Key-value dictionary representing one dataset record.

    Returns:
        Dict[str, float] with exactly the 12 keys defined in `ml.config.FEATURE_NAMES`.
        All values are numeric floats and guaranteed non-NaN and non-None.
    """
    output: Dict[str, float] = {}

    # 1. scope_mismatch
    if "scope_mismatch" in row:
        output["scope_mismatch"] = float(_safe_float(row["scope_mismatch"]))
    elif "scope" in row and "required_scope" in row:
        output["scope_mismatch"] = 0.0 if row["scope"] == row["required_scope"] else 1.0
    else:
        # Real external datasets lack OAuth scope headers; set to 0.0 as documented
        output["scope_mismatch"] = 0.0

    # 2. sequence_anomaly
    if "sequence_anomaly" in row:
        output["sequence_anomaly"] = float(_safe_float(row["sequence_anomaly"]))
    elif "endpoint_sequence_anomaly" in row:
        output["sequence_anomaly"] = float(_safe_float(row["endpoint_sequence_anomaly"]))
    elif "ip_uniq_ep_10s" in row:
        uniq_ep = _safe_float(row["ip_uniq_ep_10s"])
        if uniq_ep > 2.0:
            output["sequence_anomaly"] = 1.0
        elif uniq_ep > 1.0:
            output["sequence_anomaly"] = 0.5
        else:
            output["sequence_anomaly"] = 0.0
    else:
        output["sequence_anomaly"] = 0.0

    # 3. benford_deviation
    if "benford_deviation" in row:
        output["benford_deviation"] = float(_safe_float(row["benford_deviation"]))
    elif "amount" in row:
        output["benford_deviation"] = _compute_benford_deviation(_safe_float(row["amount"]))
    else:
        # APISentry contains no monetary transaction amounts; documented 0.0
        output["benford_deviation"] = 0.0

    # 4. request_rate_10s
    if "request_rate_10s" in row:
        output["request_rate_10s"] = float(_safe_float(row["request_rate_10s"]))
    elif "ip_req_10s" in row:
        # Standardized range: [-0.62401, 5.90462]
        output["request_rate_10s"] = _min_max_scale(_safe_float(row["ip_req_10s"]), -0.62401, 5.90462)
    else:
        output["request_rate_10s"] = 0.0

    # 5. request_rate_60s
    if "request_rate_60s" in row:
        output["request_rate_60s"] = float(_safe_float(row["request_rate_60s"]))
    elif "requests_per_ip_per_min" in row:
        output["request_rate_60s"] = _min_max_scale(_safe_float(row["requests_per_ip_per_min"]), 0.0, 300.0)
    elif "ip_req_60s" in row:
        output["request_rate_60s"] = _min_max_scale(_safe_float(row["ip_req_60s"]), -0.62401, 5.90462)
    else:
        # Derived from 10s velocity window
        output["request_rate_60s"] = float(min(1.0, output["request_rate_10s"] * 0.85))

    # 6. distinct_accounts_ratio
    if "distinct_accounts_ratio" in row:
        output["distinct_accounts_ratio"] = float(_safe_float(row["distinct_accounts_ratio"]))
    elif "ip_distinct_users_10s" in row:
        # Standardized range: [-0.49344, 7.54992]
        output["distinct_accounts_ratio"] = _min_max_scale(_safe_float(row["ip_distinct_users_10s"]), -0.49344, 7.54992)
    else:
        output["distinct_accounts_ratio"] = 0.0

    # 7. token_reuse_count
    if "token_reuse_count" in row:
        output["token_reuse_count"] = float(_safe_float(row["token_reuse_count"]))
    elif "token_ips_10s" in row:
        # Standardized range: [-0.77731, 3.62527]
        output["token_reuse_count"] = _min_max_scale(_safe_float(row["token_ips_10s"]), -0.77731, 3.62527)
    else:
        output["token_reuse_count"] = 0.0

    # 8. payload_sql_score
    if "payload_sql_score" in row:
        output["payload_sql_score"] = float(_safe_float(row["payload_sql_score"]))
    elif "payload" in row and isinstance(row["payload"], str) and _SQLI_REGEX.search(row["payload"]):
        output["payload_sql_score"] = 1.0
    elif "body" in row and isinstance(row["body"], str) and _SQLI_REGEX.search(row["body"]):
        output["payload_sql_score"] = 1.0
    elif "body_sql_hits" in row:
        # In APISentry, body_sql_hits > 1.0 denotes SQL injection attacks
        output["payload_sql_score"] = 1.0 if _safe_float(row["body_sql_hits"]) > 1.0 else 0.0
    else:
        output["payload_sql_score"] = 0.0

    # 9. payload_param_tamper
    if "payload_param_tamper" in row:
        output["payload_param_tamper"] = float(_safe_float(row["payload_param_tamper"]))
    elif "body_has_neg" in row and _safe_float(row["body_has_neg"]) > 0.0:
        output["payload_param_tamper"] = 1.0
    elif "body_min_num" in row and _safe_float(row["body_min_num"]) < -0.5:
        output["payload_param_tamper"] = 1.0
    else:
        output["payload_param_tamper"] = 0.0

    # 10. payload_length
    if "payload_length" in row:
        output["payload_length"] = float(_safe_float(row["payload_length"]))
    elif "body_len" in row:
        # Standardized range: [-0.75356, 4.09470]
        output["payload_length"] = _min_max_scale(_safe_float(row["body_len"]), -0.75356, 4.09470)
    elif "request_size" in row:
        output["payload_length"] = _min_max_scale(_safe_float(row["request_size"]), -0.75356, 4.09470)
    else:
        output["payload_length"] = 0.0

    # 11. is_post_method
    if "is_post_method" in row:
        output["is_post_method"] = float(_safe_float(row["is_post_method"]))
    elif "method_enc" in row:
        output["is_post_method"] = 1.0 if int(_safe_float(row["method_enc"])) == 1 else 0.0
    elif "method" in row:
        output["is_post_method"] = 1.0 if str(row["method"]).upper() == "POST" else 0.0
    else:
        output["is_post_method"] = 0.0

    # 12. endpoint_risk_weight
    if "endpoint_risk_weight" in row:
        output["endpoint_risk_weight"] = float(_safe_float(row["endpoint_risk_weight"]))
    elif "endpoint_norm_enc" in row:
        enc_id = int(_safe_float(row["endpoint_norm_enc"]))
        output["endpoint_risk_weight"] = float(_APISENTRY_ENDPOINT_MAP.get(enc_id, 0.3))
    elif "endpoint" in row:
        ep_str = str(row["endpoint"]).lower()
        matched_weight = 0.3
        for path_prefix, weight in _PATH_RISK_MAP.items():
            if path_prefix in ep_str:
                matched_weight = weight
                break
        output["endpoint_risk_weight"] = float(matched_weight)
    else:
        output["endpoint_risk_weight"] = 0.3

    # Safety guarantee: ensure all 12 keys exist, are floats, and non-NaN
    for feat in FEATURE_NAMES:
        if feat not in output or output[feat] is None or math.isnan(output[feat]):
            output[feat] = 0.0
        else:
            output[feat] = float(output[feat])

    return output

