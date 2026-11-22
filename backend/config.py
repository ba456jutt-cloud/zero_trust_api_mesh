"""Centralized configuration for Pakistan Open Banking API Security Mesh.

Adheres to SBP Sandbox Cohort 2 constraints:
- Inference latency budget < 200ms
- Bounded LRU pattern store <= 10,000 entries
- Standardized verdict score cutoffs
"""

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel, Field


class VerdictEnum(str, Enum):
    """Detection semantic verdicts for inspected API requests."""

    APPROVE = "APPROVE"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


class BackendSettings(BaseModel):
    """Global configuration settings for reverse proxy and security inspection."""

    app_name: str = "SBP-OpenBanking-SecurityMesh"
    environment: str = Field(default="simulation")
    host: str = "0.0.0.0"
    port: int = 8080

    # Latency constraint (ms)
    target_latency_ms: float = 200.0

    # ML Score Thresholds
    approve_threshold: float = 0.4
    block_threshold: float = 0.7

    # Storage & Logs Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")
    artifacts_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "ml" / "artifacts")
    verdicts_log_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs" / "verdicts.jsonl")
    pattern_db_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs" / "pattern_memory.db")

    # Hard Memory Cap for Sliding Window / Pattern LRU
    max_pattern_entries: int = 10000

    # Open Banking FAPI Simulation Headers
    header_consent_id: str = "x-consent-id"
    header_consent_scope: str = "x-consent-scope"
    header_client_id: str = "x-client-id"
    header_verdict: str = "x-security-verdict"

    # FAPI OAuth Scope Enforcement Mapping
    endpoint_scope_map: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "/consent": ["consent:write", "consent:read"],
            "/accounts": ["accounts:read"],
            "/transactions": ["accounts:read"],
            "/transfer": ["transfer:write"],
        }
    )

    # Expected Sequential Navigation Flow
    expected_endpoint_sequence: List[str] = Field(
        default_factory=lambda: ["/consent", "/accounts", "/transfer"]
    )


def get_settings() -> BackendSettings:
    """Retrieve application settings instance."""
    return BackendSettings()


settings = get_settings()

