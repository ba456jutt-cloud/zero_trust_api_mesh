"""Pydantic data schemas for Pakistan Open Banking API Security Mesh.

Covers:
1. FAPI-compliant domain models (/consent, /accounts, /transactions, /transfer)
2. Security interception, verdict, and SHAP explainability models
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from backend.config import VerdictEnum


# ---------------------------------------------------------------------------
# Security & Verdict Models
# ---------------------------------------------------------------------------

class SHAPExplanation(BaseModel):
    """Local feature attribution calculated via shap.Explainer."""

    feature: str = Field(description="Name of the evaluated feature")
    contribution: float = Field(description="SHAP attribution value (positive increases anomaly)")
    description: str = Field(default="", description="Human-readable explanation of the anomaly")


class SecurityEvaluationResult(BaseModel):
    """Comprehensive verdict result produced by the ML ensemble pipeline."""

    verdict: VerdictEnum
    anomaly_score: float = Field(ge=0.0, le=1.0, description="Normalized ensemble score")
    latency_ms: float = Field(description="Execution time for feature extraction and inference")
    explanations: List[SHAPExplanation] = Field(default_factory=list)
    client_id: Optional[str] = None
    endpoint: str
    http_method: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid4()))


class SecurityBlockResponse(BaseModel):
    """Payload returned when request is blocked (HTTP 403 Forbidden)."""

    status: str = "FORBIDDEN"
    message: str = "Request blocked by SBP Zero-Trust AI Security Mesh"
    verdict: VerdictEnum = VerdictEnum.BLOCK
    anomaly_score: float
    explanations: List[SHAPExplanation]
    request_id: str
    timestamp: str


# ---------------------------------------------------------------------------
# FAPI Open Banking Mock API Models
# ---------------------------------------------------------------------------

class ConsentCreateRequest(BaseModel):
    """Payload to authorize a new third-party banking consent."""

    client_id: str = Field(examples=["tpp_fintech_pk_01"])
    tpp_name: str = Field(examples=["NayaPay Partner App"])
    scopes: List[str] = Field(examples=[["accounts:read", "transfer:write"]])
    account_ids: List[str] = Field(examples=[["ACC-PK-0012948"]])
    valid_until: str = Field(examples=["2026-12-31T23:59:59Z"])


class ConsentResponse(BaseModel):
    """Active authorized consent details."""

    consent_id: str
    status: str = "authorised"
    client_id: str
    scopes: List[str]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str


class AccountInfo(BaseModel):
    """Open Banking account representation."""

    account_id: str
    iban: str = Field(description="Synthetic Pakistani IBAN (PKxx xxxx xxxx xxxx xxxx xxxx)")
    account_title: str
    currency: str = "PKR"
    account_type: str = "CURRENT"
    balance: float
    status: str = "ACTIVE"


class AccountsListResponse(BaseModel):
    """Response containing list of authorized accounts."""

    accounts: List[AccountInfo]
    total_count: int


class TransactionItem(BaseModel):
    """Historical account transaction record."""

    transaction_id: str
    account_id: str
    amount: float
    currency: str = "PKR"
    timestamp: str
    counterparty_title: str
    narrative: str
    transaction_type: str = Field(description="DEBIT or CREDIT")


class TransactionsResponse(BaseModel):
    """Response listing transactions for an account."""

    account_id: str
    transactions: List[TransactionItem]
    total_records: int


class TransferRequest(BaseModel):
    """Payload to execute an inter-bank or intra-bank payment transfer."""

    consent_id: str = Field(description="Authorized consent token ID")
    debtor_account: str = Field(description="Source account IBAN or ID")
    creditor_account: str = Field(description="Destination account IBAN or ID")
    amount: float = Field(gt=0.0, description="Transfer amount in PKR")
    currency: str = "PKR"
    purpose_code: str = Field(default="SALA", description="SBP Purpose Code (e.g. SALA, COMM, EDUC)")
    narrative: str = Field(default="Synthetic Raast transfer")


class TransferResponse(BaseModel):
    """Confirmation of an executed fund transfer."""

    transfer_id: str
    status: str = "SETTLED"
    reference_number: str
    amount: float
    currency: str = "PKR"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fee: float = 0.0

