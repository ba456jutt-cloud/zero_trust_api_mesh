"""FAPI-inspired Mock Accounts Router.

Endpoints:
- GET /accounts : Retrieve authorized account list for authenticated consent
- GET /accounts/{account_id} : Query specific account details
"""

from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Header, status
import structlog
from backend.config import settings
from backend.models import AccountInfo, AccountsListResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/accounts", tags=["Accounts"])

# Pre-seeded synthetic Pakistani bank accounts (100% synthetic mock data)
_SYNTHETIC_ACCOUNTS: Dict[str, AccountInfo] = {
    "ACC-PK-001": AccountInfo(
        account_id="ACC-PK-001",
        iban="PK36SCBL0000001123456701",
        account_title="Ahmed Khan Business",
        currency="PKR",
        account_type="BUSINESS_CURRENT",
        balance=1_450_000.50,
        status="ACTIVE",
    ),
    "ACC-PK-002": AccountInfo(
        account_id="ACC-PK-002",
        iban="PK36HABB0000009988776602",
        account_title="Fatima Ali Personal",
        currency="PKR",
        account_type="SAVINGS",
        balance=285_400.00,
        status="ACTIVE",
    ),
    "ACC-PK-003": AccountInfo(
        account_id="ACC-PK-003",
        iban="PK36MCBB0000005544332203",
        account_title="Tariq Mehmood Asaan",
        currency="PKR",
        account_type="ASAAN_CURRENT",
        balance=45_120.75,
        status="ACTIVE",
    ),
}


@router.get("", response_model=AccountsListResponse)
async def list_accounts(
    x_consent_id: Optional[str] = Header(default=None, alias=settings.header_consent_id),
) -> AccountsListResponse:
    """Fetch authorized bank accounts for the active consent session."""
    accounts = list(_SYNTHETIC_ACCOUNTS.values())
    await logger.ainfo(
        "accounts_listed",
        consent_id=x_consent_id,
        count=len(accounts),
    )
    return AccountsListResponse(accounts=accounts, total_count=len(accounts))


@router.get("/{account_id}", response_model=AccountInfo)
async def get_account(account_id: str) -> AccountInfo:
    """Fetch details of a single account by ID."""
    account = _SYNTHETIC_ACCOUNTS.get(account_id)
    if not account:
        await logger.awarning("account_not_found", account_id=account_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account '{account_id}' not found",
        )
    return account

