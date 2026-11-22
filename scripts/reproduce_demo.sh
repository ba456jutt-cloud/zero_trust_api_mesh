#!/usr/bin/env bash
set -euo pipefail

# Reproduces the full demo from a fresh checkout
# Requires: Python 3.11, curl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== 1. Set up virtualenv ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "=== 2. Verify external dataset present ==="
if [ ! -f "data/external/apisentry/dataset_processed.csv" ]; then
    echo "ERROR: External dataset missing. Download from Zenodo 22658802 into data/external/apisentry/"
    exit 1
fi

echo "=== 3. Train all synthetic models (IF, RF, LSTM, Scaler) ==="
python -m ml.train_all

echo "=== 4. Train external models & empirical calibrators ==="
python -m ml.train_rf_external
python -m ml.train_if_external

echo "=== 5. Run test suite ==="
pytest -v

echo "=== 6. Start server in background ==="
# Ensure port 8080 is clear
fuser -k 8080/tcp 2>/dev/null || true
python -m uvicorn backend.main:app --port 8080 --workers 1 &
UVICORN_PID=$!

cleanup() {
    echo "Cleaning up background Uvicorn server (PID ${UVICORN_PID})..."
    kill "${UVICORN_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be responsive
echo "Waiting for server on port 8080..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 0.5
done

echo "=== 6b. Warming up server ==="
for i in $(seq 1 20); do
    curl -s -o /dev/null http://127.0.0.1:8080/health
done
curl -s -o /dev/null -X POST http://127.0.0.1:8080/consent \
    -H "X-Consent-Scope: consent:write" \
    -H "X-Client-ID: tpp_warmup_probe_unique" \
    -H "Content-Type: application/json" \
    -d '{"client_id":"tpp_warmup_probe_unique","tpp_name":"Warmup","scopes":["consent:write"],"account_ids":["ACC-PK-WARMUP"],"valid_until":"2026-12-31T23:59:59Z"}'
echo "=== Warm-up complete ==="

echo "=== 7. Run canonical attack simulation ==="
ISO_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat << EOF > docs/LIVE_DEMO_CANONICAL.log
# ============================================================
# SBP AI API Security Mesh — Canonical Live Demo Run
# Generated: ${ISO_TS}
# Commit: $(git rev-parse HEAD 2>/dev/null || echo "no-git")
# Command: python -m tools.attack_simulator -t http://127.0.0.1:8080 -s all
# ============================================================

EOF
python -m tools.attack_simulator -t http://127.0.0.1:8080 -s all | tee -a docs/LIVE_DEMO_CANONICAL.log

echo "=== DONE — demo reproducible from scratch ==="
