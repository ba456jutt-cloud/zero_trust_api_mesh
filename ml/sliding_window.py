"""Sliding Window State Tracker with Bounded LRU Eviction.

Maintains temporal and sequential request metrics per client/IP:
- Request frequency in 10s and 60s windows (flooding detection)
- Endpoint sequence progression (sequence anomaly detection)
- Distinct entities/accounts touched (credential stuffing / brute force)
- Token reuse counters (token replay detection)

Guarantees hard constraint: <= 10,000 active entries with LRU eviction.
"""

import asyncio
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class WindowMetrics:
    """Calculated sliding window behavioral metrics for a client session."""

    count_10s: int = 0
    count_60s: int = 0
    requests_per_sec: float = 0.0
    endpoint_sequence: List[str] = field(default_factory=list)
    sequence_anomaly: float = 0.0
    distinct_accounts_count: int = 0
    token_reuse_count: int = 0


class ClientSessionHistory:
    """Bounded history structure for an individual client or IP key."""

    def __init__(self, max_timestamps: int = 200, max_sequence: int = 20) -> None:
        self.timestamps: Deque[float] = deque(maxlen=max_timestamps)
        self.endpoints: Deque[str] = deque(maxlen=max_sequence)
        self.seen_accounts: Deque[str] = deque(maxlen=max_timestamps)
        self.seen_tokens: Dict[str, int] = {}
        self.last_updated: float = time.time()

    def record_request(
        self,
        endpoint: str,
        timestamp: Optional[float] = None,
        account_id: Optional[str] = None,
        token_id: Optional[str] = None,
    ) -> None:
        """Record an incoming request into temporal buffers."""
        now = timestamp if timestamp is not None else time.time()
        self.timestamps.append(now)
        self.endpoints.append(endpoint)
        self.last_updated = now

        if account_id:
            self.seen_accounts.append(account_id)

        if token_id:
            self.seen_tokens[token_id] = self.seen_tokens.get(token_id, 0) + 1

    def compute_metrics(
        self,
        current_time: Optional[float] = None,
        expected_sequence: Optional[List[str]] = None,
        token_id: Optional[str] = None,
    ) -> WindowMetrics:
        """Calculate real-time temporal and sequence metrics."""
        now = current_time if current_time is not None else time.time()
        expected = expected_sequence or settings.expected_endpoint_sequence

        # Calculate counts in 10s and 60s windows (strictly within window)
        cutoff_10s = now - 10.0
        cutoff_60s = now - 60.0

        count_10s = sum(1 for t in self.timestamps if t > cutoff_10s)
        count_60s = sum(1 for t in self.timestamps if t > cutoff_60s)
        requests_per_sec = count_10s / 10.0

        # Sequence anomaly evaluation
        seq_anomaly = self._evaluate_sequence_anomaly(list(self.endpoints), expected)

        # Distinct accounts targeted in active window
        distinct_accounts = len(set(self.seen_accounts))

        # Token reuse count
        reuse_count = self.seen_tokens.get(token_id, 0) if token_id else 0

        return WindowMetrics(
            count_10s=count_10s,
            count_60s=count_60s,
            requests_per_sec=requests_per_sec,
            endpoint_sequence=list(self.endpoints),
            sequence_anomaly=seq_anomaly,
            distinct_accounts_count=distinct_accounts,
            token_reuse_count=reuse_count,
        )

    def _evaluate_sequence_anomaly(self, recent_endpoints: List[str], expected: List[str]) -> float:
        """Score sequence anomaly: 0.0 = normal flow, 1.0 = severe workflow skip."""
        if not recent_endpoints:
            return 0.0

        current = recent_endpoints[-1]
        # Calling /transfer as first action without consent/accounts is a severe anomaly
        if current == "/transfer":
            if "/consent" not in recent_endpoints and "/accounts" not in recent_endpoints:
                return 1.0
            if "/accounts" not in recent_endpoints:
                return 0.7

        # Calling /accounts without /consent
        if current == "/accounts" and "/consent" not in recent_endpoints:
            return 0.5

        return 0.0


class SlidingWindowTracker:
    """Thread-safe bounded LRU sliding window tracker across all clients."""

    def __init__(self, max_entries: Optional[int] = None) -> None:
        self.max_entries: int = max_entries or settings.max_pattern_entries
        self._cache: OrderedDict[str, ClientSessionHistory] = OrderedDict()
        self._lock: threading.RLock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None

    def _get_async_lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def record_and_compute(
        self,
        client_key: str,
        endpoint: str,
        timestamp: Optional[float] = None,
        account_id: Optional[str] = None,
        token_id: Optional[str] = None,
        as_of: Optional[float] = None,
    ) -> WindowMetrics:
        """Atomic thread-safe record and retrieve metrics with O(1) LRU eviction.

        Args:
            client_key: Unique identifier for client or IP.
            endpoint: Target API endpoint.
            timestamp: Optional request timestamp.
            account_id: Optional target account.
            token_id: Optional authorization token/JTI.
            as_of: Optional simulated current timestamp (takes precedence over timestamp).
        """
        with self._lock:
            effective_time = as_of if as_of is not None else (timestamp if timestamp is not None else time.time())
            history = self._get_or_create(client_key)
            history.record_request(
                endpoint=endpoint,
                timestamp=effective_time,
                account_id=account_id,
                token_id=token_id,
            )
            return history.compute_metrics(current_time=effective_time, token_id=token_id)

    async def async_record_and_compute(
        self,
        client_key: str,
        endpoint: str,
        timestamp: Optional[float] = None,
        account_id: Optional[str] = None,
        token_id: Optional[str] = None,
        as_of: Optional[float] = None,
    ) -> WindowMetrics:
        """Asynchronous record and compute protected by both async and thread locks."""
        async with self._get_async_lock():
            return self.record_and_compute(
                client_key=client_key,
                endpoint=endpoint,
                timestamp=timestamp,
                account_id=account_id,
                token_id=token_id,
                as_of=as_of,
            )

    def count_10s(self, client_key: str, as_of: Optional[float] = None) -> int:
        """Query 10s window count for a client key."""
        with self._lock:
            history = self._cache.get(client_key)
            if not history:
                return 0
            return history.compute_metrics(current_time=as_of).count_10s

    def get_metrics(self, client_key: str, token_id: Optional[str] = None) -> WindowMetrics:
        """Fetch current metrics for a client key without mutating history."""
        with self._lock:
            history = self._cache.get(client_key)
            if not history:
                return WindowMetrics()
            return history.compute_metrics(token_id=token_id)

    def _get_or_create(self, client_key: str) -> ClientSessionHistory:
        """Retrieve existing history or insert new with LRU eviction."""
        if client_key in self._cache:
            self._cache.move_to_end(client_key)
            return self._cache[client_key]

        # Enforce hard memory cap
        if len(self._cache) >= self.max_entries:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("sliding_window_lru_eviction", evicted_key=evicted_key)

        new_history = ClientSessionHistory()
        self._cache[client_key] = new_history
        return new_history

    def active_key_count(self) -> int:
        """Current number of tracked keys in memory."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Reset in-memory cache."""
        with self._lock:
            self._cache.clear()


# Default global tracker instance
tracker = SlidingWindowTracker()

