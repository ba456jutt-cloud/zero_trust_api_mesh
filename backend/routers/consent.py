"""FAPI-inspired Mock Consent Management Router.

Endpoints:
- POST /consent : Register user consent for third-party access
- GET /consent/{consent_id} : Query existing consent status
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Header, status
import structlog
from backend.config import settings
from backend.models import ConsentCreateRequest, ConsentResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/consent", tags=["Consent"])

# In-memory mock storage for synthetic consents (stateless simulation)
_SYNTHETIC_CONSENTS: Dict[str, ConsentResponse] = {
    "CONSENT-MOCK-001": ConsentResponse(
        consent_id="CONSENT-MOCK-001",
        status="authorised",
        client_id="tpp_sandbox_client_01",
        scopes=["accounts:read", "transfer:write"],
        created_at="2026-01-01T00:00:00Z",
        expires_at="2027-01-01T00:00:00Z",
    )
}


@router.post("", response_model=ConsentResponse, status_code=status.HTTP_201_CREATED)
async def create_consent(
    payload: ConsentCreateRequest,
    x_client_id: Optional[str] = Header(default=None, alias=settings.header_client_id),
) -> ConsentResponse:
    """Issue a new Open Banking consent for a verified TPP client."""
    client_id = x_client_id or payload.client_id
    consent_id = f"CONSENT-PK-{uuid4().hex[:12].upper()}"
    now_utc = datetime.now(timezone.utc)
    expires_at = (now_utc + timedelta(days=90)).isoformat()

    consent = ConsentResponse(
        consent_id=consent_id,
        status="authorised",
        client_id=client_id,
        scopes=payload.scopes,
        created_at=now_utc.isoformat(),
        expires_at=expires_at,
    )
    _SYNTHETIC_CONSENTS[consent_id] = consent

    await logger.ainfo(
        "consent_created",
        consent_id=consent_id,
        client_id=client_id,
        scopes=payload.scopes,
    )
    return consent


@router.get("/{consent_id}", response_model=ConsentResponse)
async def get_consent(consent_id: str) -> ConsentResponse:
    """Retrieve existing consent authorization details."""
    consent = _SYNTHETIC_CONSENTS.get(consent_id)
    if not consent:
        await logger.awarning("consent_not_found", consent_id=consent_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Consent record '{consent_id}' not found",
        )
    return consent

