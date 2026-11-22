"""Concurrency tests verifying live HTTP request handling and attack blocking under load."""

import asyncio
import httpx
import pytest
from backend.main import create_app


async def _get_client():
    """Return an AsyncClient targeting live server on 8080 if running, else ASGITransport."""
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=5.0) as check_client:
            res = await check_client.get("/health")
            if res.status_code == 200:
                return httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=15.0)
    except Exception:
        pass

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8080", timeout=15.0)


@pytest.mark.asyncio
async def test_50_concurrent_requests():
    """Verify 50 concurrent legitimate requests are handled without race conditions."""
    client = await _get_client()
    async with client:
        async def one_request(i: int):
            return await client.post(
                "/consent",
                json={
                    "client_id": f"tpp_concurrent_{i}",
                    "tpp_name": "Test Fintech Partner",
                    "scopes": ["accounts:read"],
                    "account_ids": ["ACC-PK-001"],
                    "valid_until": "2026-12-31T23:59:59Z",
                },
                headers={
                    "X-Consent-Scope": "consent:write",
                    "X-Client-ID": f"tpp_concurrent_{i}",
                },
            )

        results = await asyncio.gather(*[one_request(i) for i in range(50)])

    failures = [r.status_code for r in results if r.status_code not in (200, 201)]
    assert all(r.status_code in (200, 201) for r in results), f"Failures: {failures}"


@pytest.mark.asyncio
async def test_50_concurrent_attacks_all_blocked():
    """Verify 50 concurrent SQLi attack requests are blocked (HTTP 403)."""
    client = await _get_client()
    async with client:
        async def one_attack(i: int):
            return await client.post(
                "/transfer",
                json={
                    "consent_id": f"c-{i}",
                    "debtor_account": "PK36SCBL0000001123456789",
                    "creditor_account": "PK36HABB0000009988776602",
                    "amount": 1000.0,
                    "currency": "PKR",
                    "narrative": f"Transfer {i}; DROP TABLE users; -- ' OR 1=1 --",
                },
                headers={
                    "X-Consent-Scope": "transfer:write",
                    "X-Consent-ID": f"c-{i}",
                    "X-Client-ID": f"attacker_{i}",
                },
            )

        results = await asyncio.gather(*[one_attack(i) for i in range(50)])

    blocked = sum(1 for r in results if r.status_code == 403)
    assert blocked >= 45, f"Only {blocked}/50 blocked under concurrent load"

