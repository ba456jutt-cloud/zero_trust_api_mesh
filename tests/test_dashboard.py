"""Unit tests for Streamlit SOC Analyst Dashboard data loading and helpers."""

import json
from pathlib import Path
import pandas as pd
import pytest
from frontend.dashboard import get_pattern_store, load_verdict_logs
from tools.seed_patterns import PatternMemoryStore


def test_load_verdict_logs_missing_file(tmp_path: Path) -> None:
    """Verify missing log file gracefully returns empty DataFrame."""
    non_existent = str(tmp_path / "missing_verdicts.jsonl")
    df = load_verdict_logs(non_existent)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_verdict_logs_valid_records(tmp_path: Path) -> None:
    """Verify parsing valid JSONL entries into structured DataFrame."""
    log_file = tmp_path / "test_verdicts.jsonl"
    sample_records = [
        {
            "verdict": "APPROVE",
            "anomaly_score": 0.15,
            "latency_ms": 22.4,
            "explanations": [],
            "client_id": "tpp_user_01",
            "endpoint": "/consent",
            "http_method": "POST",
            "timestamp": "2026-11-20T10:00:00+00:00",
            "request_id": "req-001",
        },
        {
            "verdict": "BLOCK",
            "anomaly_score": 0.95,
            "latency_ms": 31.8,
            "explanations": [{"feature": "payload_sql_score", "contribution": 1.0, "description": "SQL injection detected"}],
            "client_id": "tpp_attacker_01",
            "endpoint": "/transfer",
            "http_method": "POST",
            "timestamp": "2026-11-20T10:01:00+00:00",
            "request_id": "req-002",
        },
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for r in sample_records:
            f.write(json.dumps(r) + "\n")

    df = load_verdict_logs(str(log_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "verdict" in df.columns
    assert "anomaly_score" in df.columns
    assert "latency_ms" in df.columns
    assert df.iloc[0]["verdict"] == "BLOCK"  # Sorted newest first


def test_get_pattern_store_integration() -> None:
    """Verify PatternMemoryStore instance returned by dashboard helper."""
    store = get_pattern_store()
    assert isinstance(store, PatternMemoryStore)
    assert store.count_patterns() >= 0

