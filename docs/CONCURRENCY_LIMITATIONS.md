# Concurrency & Scaling Limitations (v1)

## Single-Worker Assumption

The sliding-window state (`ml.sliding_window.SlidingWindowTracker`) is held in
Python process memory. This means:

- **Single worker (default)**: All requests share sliding window state. Flooding
  detection, credential stuffing detection, and token replay detection work
  correctly across concurrent requests — verified via `tests/test_concurrent_http.py`.

- **Multiple workers (`--workers N > 1`)**: Each worker process has its own isolated
  sliding window state. A client hitting worker A then worker B will appear as TWO
  separate low-velocity clients to each worker. Flooding detection would FAIL silently
  (attack goes undetected across partitioned memory).

## Why We Ship Single-Worker in the Sandbox

SBP Open Banking Regulations 2026 (draft) Phase 1 testing is limited-scale
(we propose max 50 concurrent test users). A single Uvicorn worker comfortably handles
high throughput (well above our 200ms latency SLA requirement of 5 req/sec
target). Multi-worker is not required for sandbox success.

## Production Path (Post-Sandbox)

For production, sliding window state must move to an external cache:
```text
Client → Gateway / FastAPI (N workers) → Redis (Shared Sliding Window) → Decision
```

Redis sorted sets (`ZADD` with timestamp scores, `ZCOUNT` for window counts, `ZREMRANGEBYSCORE` for eviction) map directly to our current sliding-window semantics. Migration is ~1 day of engineering work, deferred to post-sandbox production hardening.

## Test Coverage

- `tests/test_concurrent_http.py::test_50_concurrent_requests` — verifies
  50 concurrent legitimate requests all succeed.
- `tests/test_concurrent_http.py::test_50_concurrent_attacks_all_blocked` —
  verifies 50 concurrent SQLi attacks are all blocked (HTTP 403).
- `tests/test_sliding_window.py::test_concurrent_updates_are_thread_safe` —
  verifies in-memory tracker handles concurrent writes and bounds LRU capacity under multithreaded load.

## Known Limitations (Honest Disclosure)

1. **LRU eviction race**: If worker hits 10,000 entries while a request is
   mid-processing, the request's own entry may be evicted before completion.
   Impact: extreme edge case; no evidence of real-world failures at our target
   scale (< 50 concurrent users).

2. **Pattern-memory SQLite lock**: `logs/pattern_memory.db` uses SQLite WAL mode
   which handles concurrent reads well but serializes writes. Under very high
   concurrency (~1000 req/sec writes), SQLite write lock could become
   a bottleneck. Impact: none at sandbox scale.

3. **SHAP explainer model access**: SHAP `TreeExplainer` holds a reference to the
   RF model. If the model is reloaded at runtime, concurrent requests may see
   mixed state. Impact: models are loaded once at startup and never reloaded
   in v1. Documented for future hot-reload consideration.

