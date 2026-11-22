"""Integration tests for Starlette reverse-proxy interceptor middleware and main app."""

import json
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
from backend.config import settings
from backend.main import app


@pytest.fixture(autouse=True)
def clean_test_log(tmp_path: Path) -> None:
    """Ensure clean temporary verdict log path for tests."""
    temp_log = tmp_path / "test_verdicts.jsonl"
    settings.verdicts_log_path = temp_log


@pytest.mark.asyncio
async def test_health_check_bypasses_mesh() -> None:
    """System endpoints like /health should bypass security evaluation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "UP"
        assert settings.header_verdict not in res.headers


@pytest.mark.asyncio
async def test_approve_flow_with_body_re_reading() -> None:
    """Valid request with required scope should pass through and downstream must read body."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {
            settings.header_consent_scope: "consent:write",
            settings.header_client_id: "tpp_approved_01",
        }
        payload = {
            "client_id": "tpp_approved_01",
            "tpp_name": "Verified Banking Client",
            "scopes": ["accounts:read", "transfer:write"],
            "account_ids": ["ACC-PK-001"],
            "valid_until": "2026-12-31T23:59:59Z",
        }
        res = await client.post("/consent", json=payload, headers=headers)
        assert res.status_code == 201
        data = res.json()
        assert data["client_id"] == "tpp_approved_01"
        assert "x-mesh-latency-ms" in res.headers


@pytest.mark.asyncio
async def test_block_flow_on_scope_mismatch() -> None:
    """Request missing required scope must be blocked (HTTP 403) with SHAP explanation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # /accounts requires 'accounts:read', send 'transfer:write' instead
        headers = {
            settings.header_consent_scope: "transfer:write",
            settings.header_client_id: "tpp_unauthorized_scope",
        }
        res = await client.get("/accounts", headers=headers)
        assert res.status_code == 403
        data = res.json()
        assert data["verdict"] == "BLOCK"
        assert data["anomaly_score"] >= 0.7
        assert len(data["explanations"]) > 0
        assert data["explanations"][0]["feature"] == "scope_mismatch"
        assert res.headers[settings.header_verdict] == "BLOCK"


@pytest.mark.asyncio
async def test_block_flow_on_sql_injection() -> None:
    """Payload containing SQL injection must trigger 403 BLOCK with explanation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {
            settings.header_consent_scope: "transfer:write",
            settings.header_client_id: "tpp_attacker",
        }
        payload = {
            "consent_id": "CONSENT-MOCK-001",
            "debtor_account": "PK36SCBL0000001123456701",
            "creditor_account": "PK36HABB0000009988776602",
            "amount": 1000.0,
            "currency": "PKR",
            "purpose_code": "SALA",
            "narrative": "' OR 1=1 -- malicious injection",
        }
        res = await client.post("/transfer", json=payload, headers=headers)
        assert res.status_code == 403
        data = res.json()
        assert data["verdict"] == "BLOCK"
        features = [e["feature"] for e in data["explanations"]]
        assert "sql_injection_signature" in features


@pytest.mark.asyncio
async def test_verdicts_are_logged_to_jsonl() -> None:
    """Intercepted requests must be appended to the verdicts.jsonl sink."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {settings.header_consent_scope: "accounts:read"}
        await client.get("/accounts", headers=headers)

        assert settings.verdicts_log_path.exists()
        with open(settings.verdicts_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["endpoint"] == "/accounts"
        assert record["verdict"] in ["APPROVE", "FLAG", "BLOCK"]

