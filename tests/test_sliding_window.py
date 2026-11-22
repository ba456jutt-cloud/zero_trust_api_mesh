"""Unit tests for Sliding Window State Tracker and LRU eviction."""

import asyncio
import pytest
from ml.sliding_window import ClientSessionHistory, SlidingWindowTracker


def test_window_counts_accuracy() -> None:
    """Verify 10s and 60s window counting over simulated time."""
    history = ClientSessionHistory()
    base_time = 1000.0

    # Record requests at simulated offsets
    history.record_request("/accounts", timestamp=base_time - 70.0)  # > 60s ago
    history.record_request("/accounts", timestamp=base_time - 40.0)  # in 60s, outside 10s
    history.record_request("/accounts", timestamp=base_time - 5.0)   # in 10s
    history.record_request("/transfer", timestamp=base_time)         # in 10s

    metrics = history.compute_metrics(current_time=base_time)
    assert metrics.count_10s == 2
    assert metrics.count_60s == 3
    assert metrics.requests_per_sec == 0.2


def test_flooding_burst_detection() -> None:
    """Verify rapid API flooding detection."""
    tracker = SlidingWindowTracker()
    client = "attacker_burst_bot"
    now = 5000.0

    for i in range(100):
        metrics = tracker.record_and_compute(client, "/accounts", timestamp=now + (i * 0.05))

    assert metrics.count_10s == 100
    assert metrics.requests_per_sec == 10.0


def test_sequence_anomaly_detection() -> None:
    """Verify normal sequence vs anomalous skip detection."""
    # Case 1: Normal sequence consent -> accounts -> transfer
    normal_history = ClientSessionHistory()
    normal_history.record_request("/consent")
    normal_history.record_request("/accounts")
    normal_history.record_request("/transfer")
    normal_metrics = normal_history.compute_metrics()
    assert normal_metrics.sequence_anomaly == 0.0

    # Case 2: Severe skip directly to /transfer
    anomalous_history = ClientSessionHistory()
    anomalous_history.record_request("/transfer")
    anomalous_metrics = anomalous_history.compute_metrics()
    assert anomalous_metrics.sequence_anomaly == 1.0

    # Case 3: Skip consent, call accounts directly
    skip_consent = ClientSessionHistory()
    skip_consent.record_request("/accounts")
    assert skip_consent.compute_metrics().sequence_anomaly == 0.5


def test_token_replay_tracking() -> None:
    """Verify token reuse counting for token replay detection."""
    tracker = SlidingWindowTracker()
    client = "client_replay_test"
    token = "jti_token_abc_123"

    for _ in range(4):
        metrics = tracker.record_and_compute(client, "/transfer", token_id=token)

    assert metrics.token_reuse_count == 4


def test_credential_stuffing_distinct_accounts() -> None:
    """Verify tracking of distinct targeted accounts in window."""
    tracker = SlidingWindowTracker()
    client = "stuffing_attacker"

    for i in range(20):
        metrics = tracker.record_and_compute(client, "/accounts", account_id=f"ACC-PK-{i:03d}")

    assert metrics.distinct_accounts_count == 20


def test_hard_constraint_lru_eviction() -> None:
    """Verify strict LRU eviction bounding to prevent memory leaks."""
    capacity = 5
    tracker = SlidingWindowTracker(max_entries=capacity)

    for i in range(10):
        tracker.record_and_compute(f"client_{i}", "/consent")

    assert tracker.active_key_count() == capacity
    # Earliest clients (0..4) must have been evicted
    assert tracker.get_metrics("client_0").count_10s == 0
    # Latest clients (5..9) must be retained
    assert tracker.get_metrics("client_9").count_10s == 1


def test_sliding_window_with_explicit_timestamps() -> None:
    """Verify window counting when explicit timestamps/as_of are passed."""
    tracker = SlidingWindowTracker()
    client = "client_sim_time_test"

    # Record 5 events at t=0, t=1, t=2, t=11, t=12
    tracker.record_and_compute(client, "/accounts", as_of=0.0)
    tracker.record_and_compute(client, "/accounts", as_of=1.0)
    m2 = tracker.record_and_compute(client, "/accounts", as_of=2.0)
    assert m2.count_10s == 3  # All 3 fall in first window

    tracker.record_and_compute(client, "/accounts", as_of=11.0)
    m12 = tracker.record_and_compute(client, "/accounts", as_of=12.0)
    assert m12.count_10s == 2  # Only the last 2 (t=11, t=12) fall in 10s window


@pytest.mark.asyncio
async def test_concurrent_updates_are_thread_safe() -> None:
    """Verify concurrent record_and_compute calls maintain thread-safe counts and LRU limits."""
    tracker = SlidingWindowTracker(max_entries=10)
    client = "client_concurrent"

    # 1. 100 concurrent updates for single client
    async def worker_single(i: int) -> None:
        await asyncio.to_thread(
            tracker.record_and_compute,
            client_key=client,
            endpoint="/accounts",
            as_of=1000.0 + i * 0.01,
        )

    await asyncio.gather(*[worker_single(i) for i in range(100)])
    final_count = tracker.count_10s(client, as_of=1001.5)
    assert final_count == 100, f"Expected 100, got {final_count}"

    # 2. 100 concurrent updates across 25 keys with max_entries=10
    async def worker_multi(i: int) -> None:
        await asyncio.to_thread(
            tracker.record_and_compute,
            client_key=f"client_multi_{i % 25}",
            endpoint="/consent",
            as_of=2000.0 + i * 0.01,
        )

    await asyncio.gather(*[worker_multi(i) for i in range(100)])
    assert tracker.active_key_count() <= 10, f"Expected <= 10, got {tracker.active_key_count()}"



