"""Unit tests for the external dataset adapter.

Verifies:
1. Adapter returns all 12 feature keys matching `ml.config.FEATURE_NAMES`.
2. All returned values are numeric (int or float).
3. No NaN or None values in output under normal, empty, or corrupted input rows.
"""

import math
import pandas as pd
import pytest

from data.external.adapter import adapt_external_row
from ml.config import FEATURE_NAMES


@pytest.fixture
def apisentry_sample_row():
    """Load the first row from the real APISentry dataset if present, else fallback sample."""
    csv_path = "data/external/apisentry/dataset_processed.csv"
    try:
        df = pd.read_csv(csv_path, nrows=1)
        return df.to_dict(orient="records")[0]
    except Exception:
        return {
            "status_code": -0.349,
            "response_time_ms": 5.576,
            "request_size": 1.757,
            "response_size": -0.259,
            "hour": 0.0,
            "body_len": 1.722,
            "body_special": 1.563,
            "body_sql_hits": -0.415,
            "body_max_num": -0.063,
            "body_min_num": -0.059,
            "body_has_neg": -0.079,
            "body_field_count": 1.501,
            "is_error": -0.349,
            "is_login": 2.054,
            "ip_req_10s": -0.624,
            "ip_fail_10s": -0.356,
            "ip_fail_ratio_10s": -0.428,
            "ip_login_10s": -0.008,
            "ip_uniq_ep_10s": -1.059,
            "ip_distinct_users_10s": -0.493,
            "token_ips_10s": -0.777,
            "method_enc": 1,
            "endpoint_norm_enc": 2,
            "country_enc": 9,
            "device_enc": 1,
            "label": "normal",
            "label_binary": 0,
        }


def test_adapter_returns_all_twelve_keys(apisentry_sample_row):
    """Test 1: Adapter returns all 12 keys for a sample row."""
    adapted = adapt_external_row(apisentry_sample_row)

    assert isinstance(adapted, dict)
    assert len(adapted) == 12
    assert set(adapted.keys()) == set(FEATURE_NAMES)
    for key in FEATURE_NAMES:
        assert key in adapted, f"Missing required feature key: {key}"


def test_adapter_keys_match_config(apisentry_sample_row):
    """Test 1b: Adapter output keys exactly match ml.config.FEATURE_NAMES."""
    adapted = adapt_external_row(apisentry_sample_row)
    assert set(adapted.keys()) == set(FEATURE_NAMES)
    assert len(adapted.keys()) == len(FEATURE_NAMES)



def test_adapter_values_are_numeric(apisentry_sample_row):
    """Test 2: All returned values are numeric (int/float)."""
    # Test on real dataset sample row
    adapted = adapt_external_row(apisentry_sample_row)
    for key, val in adapted.items():
        assert isinstance(val, (int, float)), f"Value for {key} is not numeric: {type(val)}"

    # Test with custom row containing SQLi and string parameters
    custom_row = {
        "payload": "SELECT * FROM users WHERE id=1 OR '1'='1'--",
        "method": "POST",
        "endpoint": "/order",
        "amount": 1500.50,
        "requests_per_ip_per_min": 120,
    }
    adapted_custom = adapt_external_row(custom_row)
    for key, val in adapted_custom.items():
        assert isinstance(val, (int, float)), f"Custom value for {key} is not numeric: {type(val)}"


def test_adapter_no_nan_or_none(apisentry_sample_row):
    """Test 3: No NaN or None values in output under clean, empty, or corrupted rows."""
    # Test 3a: Clean dataset sample row
    adapted = adapt_external_row(apisentry_sample_row)
    for key, val in adapted.items():
        assert val is not None, f"Feature {key} is None"
        assert not math.isnan(val), f"Feature {key} is NaN"
        assert not math.isinf(val), f"Feature {key} is infinite"

    # Test 3b: Completely empty row
    empty_adapted = adapt_external_row({})
    assert len(empty_adapted) == 12
    for key, val in empty_adapted.items():
        assert val is not None, f"Feature {key} is None for empty row"
        assert not math.isnan(val), f"Feature {key} is NaN for empty row"
        assert val >= 0.0, f"Feature {key} is negative for empty row"

    # Test 3c: Corrupted row with explicit None and float('nan')
    corrupted_row = {
        "ip_req_10s": float("nan"),
        "body_sql_hits": None,
        "method_enc": "invalid_string",
        "endpoint_norm_enc": float("nan"),
        "amount": float("nan"),
    }
    corrupted_adapted = adapt_external_row(corrupted_row)
    for key, val in corrupted_adapted.items():
        assert val is not None, f"Feature {key} is None for corrupted row"
        assert not math.isnan(val), f"Feature {key} is NaN for corrupted row"

