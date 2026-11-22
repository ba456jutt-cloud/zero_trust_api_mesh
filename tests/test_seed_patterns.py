"""Unit and Integration tests for Threat Pattern Memory Store (SQLite + LRU Eviction)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from tools.seed_patterns import PatternMemoryStore, seed_initial_threat_patterns


@pytest.fixture
def temp_pattern_store(tmp_path: Path) -> PatternMemoryStore:
    """Provide isolated PatternMemoryStore in a temporary directory."""
    db_file = tmp_path / "test_pattern_memory.db"
    return PatternMemoryStore(db_path=db_file, max_entries=500)


def test_pattern_store_initialization(temp_pattern_store: PatternMemoryStore) -> None:
    """Verify table creation and initial zero count."""
    assert temp_pattern_store.db_path.exists()
    assert temp_pattern_store.count_patterns() == 0


def test_pattern_record_and_lookup(temp_pattern_store: PatternMemoryStore) -> None:
    """Verify recording a pattern and looking it up by its SHA-256 hash."""
    features = {"payload_sql_score": 1.0, "endpoint_risk_weight": 1.0}
    explanations = [{"feature": "payload_sql_score", "contribution": 1.0, "description": "SQL injection"}]

    pat_hash = temp_pattern_store.record_pattern(
        client_id="tpp_attacker_01",
        endpoint="/transfer",
        attack_type="sql_injection",
        features=features,
        verdict="BLOCK",
        anomaly_score=0.95,
        explanations=explanations,
    )
    assert len(pat_hash) == 64  # SHA-256 length
    assert temp_pattern_store.count_patterns() == 1

    record = temp_pattern_store.lookup_pattern(pat_hash)
    assert record is not None
    assert record["client_id"] == "tpp_attacker_01"
    assert record["endpoint"] == "/transfer"
    assert record["attack_type"] == "sql_injection"
    assert record["verdict"] == "BLOCK"
    assert record["anomaly_score"] == 0.95
    assert record["hit_count"] == 1
    assert len(record["explanations"]) == 1


def test_pattern_upsert_hit_count_increment(temp_pattern_store: PatternMemoryStore) -> None:
    """Verify duplicate pattern recording increments hit_count without duplicating rows."""
    features = {"scope_mismatch": 1.0}

    h1 = temp_pattern_store.record_pattern(
        client_id="tpp_attacker_02",
        endpoint="/accounts",
        attack_type="brute_force_login",
        features=features,
        verdict="BLOCK",
        anomaly_score=0.88,
    )
    h2 = temp_pattern_store.record_pattern(
        client_id="tpp_attacker_02",
        endpoint="/accounts",
        attack_type="brute_force_login",
        features=features,
        verdict="BLOCK",
        anomaly_score=0.89,
    )
    assert h1 == h2
    assert temp_pattern_store.count_patterns() == 1

    record = temp_pattern_store.lookup_pattern(h1)
    assert record is not None
    assert record["hit_count"] == 2
    assert record["anomaly_score"] == 0.89


def test_strict_hard_constraint_lru_eviction(tmp_path: Path) -> None:
    """Verify bounded capacity and strict LRU eviction when row limit is reached."""
    bounded_cap = 25
    db_file = tmp_path / "bounded_lru_test.db"
    store = PatternMemoryStore(db_path=db_file, max_entries=bounded_cap)

    base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    inserted_hashes = []

    # Insert 40 distinct patterns (exceeding cap of 25 by 15)
    for i in range(40):
        t = base_time + timedelta(minutes=i)
        features = {"request_rate_10s": float(i + 1)}
        h = store.record_pattern(
            client_id=f"attacker_{i:03d}",
            endpoint="/accounts",
            attack_type="credential_stuffing",
            features=features,
            observed_time=t,
        )
        inserted_hashes.append(h)

    # 1. Total row count must strictly equal bounded_cap (25)
    total_stored = store.count_patterns()
    assert total_stored == bounded_cap, f"Expected {bounded_cap} rows, found {total_stored}"

    # 2. The oldest 15 entries (indices 0..14) must have been evicted
    for h in inserted_hashes[:15]:
        assert store.lookup_pattern(h) is None, "Oldest pattern was not evicted by LRU"

    # 3. The newest 25 entries (indices 15..39) must remain present
    for h in inserted_hashes[15:]:
        assert store.lookup_pattern(h) is not None, "Recent pattern was incorrectly evicted"


def test_seed_initial_threat_patterns(temp_pattern_store: PatternMemoryStore) -> None:
    """Verify seeding pipeline populates the pattern store with all 7 attack vectors."""
    seeded_count = seed_initial_threat_patterns(store=temp_pattern_store)
    assert seeded_count > 0
    assert temp_pattern_store.count_patterns() == seeded_count

    recent = temp_pattern_store.get_recent_patterns(limit=200)
    distinct_attack_types = {r["attack_type"] for r in recent}

    expected_vectors = {
        "sql_injection",
        "parameter_tampering",
        "brute_force_login",
        "credential_stuffing",
        "bola",
        "api_flooding",
        "token_replay",
    }
    assert expected_vectors.issubset(distinct_attack_types)
