"""Unit tests for FAPI-inspired Mock Open Banking Routers."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from backend.routers.accounts import router as accounts_router
from backend.routers.consent import router as consent_router
from backend.routers.transactions import router as transactions_router
from backend.routers.transfer import router as transfer_router


@pytest.fixture
def test_app() -> FastAPI:
    """Create isolated FastAPI app mounting mock banking routers."""
    app = FastAPI(title="Test Mock Banking Gateway")
    app.include_router(consent_router)
    app.include_router(accounts_router)
    app.include_router(transactions_router)
    app.include_router(transfer_router)
    return app


@pytest.mark.asyncio
async def test_consent_lifecycle(test_app: FastAPI) -> None:
    """Test creating and querying consent."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create consent
        create_res = await client.post(
            "/consent",
            json={
                "client_id": "tpp_habib_test",
                "tpp_name": "Sadapay Integration",
                "scopes": ["accounts:read", "transfer:write"],
                "account_ids": ["ACC-PK-001"],
                "valid_until": "2026-12-31T23:59:59Z",
            },
        )
        assert create_res.status_code == 201
        data = create_res.json()
        assert "consent_id" in data
        assert data["status"] == "authorised"
        consent_id = data["consent_id"]

        # Query created consent
        get_res = await client.get(f"/consent/{consent_id}")
        assert get_res.status_code == 200
        assert get_res.json()["consent_id"] == consent_id


@pytest.mark.asyncio
async def test_accounts_endpoints(test_app: FastAPI) -> None:
    """Test accounts list and single account query."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        list_res = await client.get("/accounts")
        assert list_res.status_code == 200
        data = list_res.json()
        assert data["total_count"] >= 3

        # Query existing account
        single_res = await client.get("/accounts/ACC-PK-001")
        assert single_res.status_code == 200
        assert single_res.json()["account_id"] == "ACC-PK-001"

        # Query non-existent account
        missing_res = await client.get("/accounts/ACC-NONEXISTENT")
        assert missing_res.status_code == 404


@pytest.mark.asyncio
async def test_transactions_endpoints(test_app: FastAPI) -> None:
    """Test retrieving transactions for an account."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/transactions/ACC-PK-001")
        assert res.status_code == 200
        data = res.json()
        assert data["account_id"] == "ACC-PK-001"
        assert len(data["transactions"]) > 0


@pytest.mark.asyncio
async def test_transfer_endpoints(test_app: FastAPI) -> None:
    """Test fund transfer initiation and settlement."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        transfer_payload = {
            "consent_id": "CONSENT-MOCK-001",
            "debtor_account": "PK36SCBL0000001123456701",
            "creditor_account": "PK36HABB0000009988776602",
            "amount": 25000.0,
            "currency": "PKR",
            "purpose_code": "SALA",
            "narrative": "Staff Salary Settlement",
        }
        res = await client.post("/transfer", json=transfer_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SETTLED"
        assert data["amount"] == 25000.0
        assert data["reference_number"].startswith("RST-PK-")

