"""FAPI-inspired Mock Payment Transfer Router.

Endpoints:
- POST /transfer : Initiate and settle fund transfer between accounts
"""

from datetime import datetime, timezone
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Header, status
import structlog
from backend.config import settings
from backend.models import TransferRequest, TransferResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/transfer", tags=["Transfer"])


@router.post("", response_model=TransferResponse, status_code=status.HTTP_200_OK)
async def initiate_transfer(
    payload: TransferRequest,
    x_consent_id: str | None = Header(default=None, alias=settings.header_consent_id),
) -> TransferResponse:
    """Execute synthetic Raast / 1LINK IBFT payment transfer."""
    if payload.amount <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Transfer amount must be strictly positive",
        )

    transfer_id = f"TXF-{uuid4().hex[:10].upper()}"
    reference_number = f"RST-PK-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid4().hex[:6].upper()}"

    await logger.ainfo(
        "transfer_settled",
        transfer_id=transfer_id,
        reference_number=reference_number,
        amount=payload.amount,
        debtor=payload.debtor_account,
        creditor=payload.creditor_account,
        consent_id=x_consent_id or payload.consent_id,
    )

    return TransferResponse(
        transfer_id=transfer_id,
        status="SETTLED",
        reference_number=reference_number,
        amount=payload.amount,
        currency=payload.currency,
        timestamp=datetime.now(timezone.utc).isoformat(),
        fee=0.0,
    )

