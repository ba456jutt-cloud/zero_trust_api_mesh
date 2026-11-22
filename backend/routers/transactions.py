"""FAPI-inspired Mock Transactions Router.

Endpoints:
- GET /transactions/{account_id} : Query transaction statement for an account
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Header, status
import structlog
from backend.config import settings
from backend.models import TransactionItem, TransactionsResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/transactions", tags=["Transactions"])

# Seeded synthetic Pakistani transaction history
_SYNTHETIC_TRANSACTIONS: Dict[str, List[TransactionItem]] = {
    "ACC-PK-001": [
        TransactionItem(
            transaction_id="TXN-PK-90112",
            account_id="ACC-PK-001",
            amount=45_000.0,
            currency="PKR",
            timestamp="2026-11-18T10:14:22Z",
            counterparty_title="Lahore Supply Logistics",
            narrative="Vendor Invoice 4920 Payment",
            transaction_type="DEBIT",
        ),
        TransactionItem(
            transaction_id="TXN-PK-90113",
            account_id="ACC-PK-001",
            amount=120_000.0,
            currency="PKR",
            timestamp="2026-11-18T14:45:00Z",
            counterparty_title="Karachi Retail Client",
            narrative="Raast Instant Settlement",
            transaction_type="CREDIT",
        ),
    ],
    "ACC-PK-002": [
        TransactionItem(
            transaction_id="TXN-PK-88401",
            account_id="ACC-PK-002",
            amount=3_500.0,
            currency="PKR",
            timestamp="2026-11-19T08:30:15Z",
            counterparty_title="K-Electric Utility",
            narrative="Monthly Power Bill IBFT",
            transaction_type="DEBIT",
        ),
    ],
}


@router.get("/{account_id}", response_model=TransactionsResponse)
async def get_transactions(
    account_id: str,
    x_consent_id: Optional[str] = Header(default=None, alias=settings.header_consent_id),
) -> TransactionsResponse:
    """Fetch statement records for specified account ID."""
    transactions = _SYNTHETIC_TRANSACTIONS.get(account_id, [])
    if not transactions and account_id not in ["ACC-PK-001", "ACC-PK-002", "ACC-PK-003"]:
        await logger.awarning("transactions_account_not_found", account_id=account_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No account or transactions found for '{account_id}'",
        )

    await logger.ainfo(
        "transactions_retrieved",
        account_id=account_id,
        consent_id=x_consent_id,
        count=len(transactions),
    )
    return TransactionsResponse(
        account_id=account_id,
        transactions=transactions,
        total_records=len(transactions),
    )

