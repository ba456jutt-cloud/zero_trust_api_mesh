"""Reverse-proxy interceptor middleware for API security mesh.

Intercepts requests, evaluates telemetry, checks deterministic rules,
and returns APPROVE, FLAG, or BLOCK verdicts with SHAP explanations.
Ensures body stream is preserved and re-readable by downstream handlers.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import structlog
from backend.config import VerdictEnum, settings
from backend.models import SecurityBlockResponse, SecurityEvaluationResult, SHAPExplanation

logger = structlog.get_logger(__name__)

# Type definition for pipeline evaluator: (Request, bytes) -> SecurityEvaluationResult
PipelineEvaluator = Callable[[Request, bytes], Awaitable[SecurityEvaluationResult]]

# Global registry for ML pipeline evaluation
_ACTIVE_PIPELINE: Optional[PipelineEvaluator] = None


def set_active_pipeline(evaluator: PipelineEvaluator) -> None:
    """Register active ML/ensemble pipeline evaluator."""
    global _ACTIVE_PIPELINE
    _ACTIVE_PIPELINE = evaluator


def get_active_pipeline() -> Optional[PipelineEvaluator]:
    """Retrieve currently registered pipeline evaluator."""
    return _ACTIVE_PIPELINE


class SecurityMeshInterceptor(BaseHTTPMiddleware):
    """Starlette middleware implementing reverse-proxy zero-trust inspection."""

    def __init__(self, app: Any, log_path: Optional[Path] = None) -> None:
        super().__init__(app)
        self._custom_log_path = log_path

    @property
    def log_path(self) -> Path:
        """Resolve log path dynamically from settings or override."""
        return self._custom_log_path or settings.verdicts_log_path

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Intercept, preserve body, evaluate security mesh, and render verdict."""
        # Bypass inspection for health & telemetry endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json", "/"]:
            return await call_next(request)

        start_time = time.perf_counter()

        # Re-inject body so downstream handlers can re-read it
        body_bytes = await request.body()

        async def receive() -> Dict[str, Any]:
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request._receive = receive  # type: ignore[assignment]

        evaluation = await self._evaluate_request(request, body_bytes, start_time)
        await self._log_verdict(evaluation)

        # Enforce security decisions
        if evaluation.verdict == VerdictEnum.BLOCK:
            return self._build_block_response(evaluation)

        response = await call_next(request)

        response.headers[settings.header_verdict] = evaluation.verdict.value
        response.headers["x-mesh-risk-score"] = f"{evaluation.anomaly_score:.4f}"
        response.headers["x-mesh-latency-ms"] = f"{evaluation.latency_ms:.2f}"
        return response

    async def _evaluate_request(
        self, request: Request, body_bytes: bytes, start_time: float
    ) -> SecurityEvaluationResult:
        """Run ML pipeline if present, else fallback to deterministic pre-ML checks."""
        pipeline = get_active_pipeline()
        if pipeline is not None:
            try:
                return await pipeline(request, body_bytes)
            except Exception as exc:
                await logger.aerror("pipeline_evaluation_failed", error=str(exc))

        # Baseline deterministic pre-ML evaluation
        return await self._deterministic_evaluation(request, body_bytes, start_time)

    async def _deterministic_evaluation(
        self, request: Request, body_bytes: bytes, start_time: float
    ) -> SecurityEvaluationResult:
        """Execute deterministic pre-ML rule checks."""
        path = request.url.path
        method = request.method
        headers = {k.lower(): v for k, v in request.headers.items()}
        explanations: List[SHAPExplanation] = []
        anomaly_score = 0.0

        # Scope enforcement check
        granted_scope = headers.get(settings.header_consent_scope, "")
        required_scopes = settings.endpoint_scope_map.get(path, [])
        if required_scopes and not any(s in granted_scope for s in required_scopes):
            anomaly_score = 0.85
            explanations.append(
                SHAPExplanation(
                    feature="scope_mismatch",
                    contribution=0.85,
                    description=f"Scope '{granted_scope}' does not satisfy required {required_scopes}",
                )
            )

        # Malformed SQL payload detection check
        body_str = body_bytes.decode("utf-8", errors="ignore").lower()
        sql_indicators = ["' or 1=1", "union select", "drop table", "--", "/*"]
        if any(ind in body_str for ind in sql_indicators):
            anomaly_score = max(anomaly_score, 0.95)
            explanations.append(
                SHAPExplanation(
                    feature="sql_injection_signature",
                    contribution=0.95,
                    description="SQL injection pattern identified in payload",
                )
            )

        verdict = self._score_to_verdict(anomaly_score)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SecurityEvaluationResult(
            verdict=verdict,
            anomaly_score=anomaly_score,
            latency_ms=latency_ms,
            explanations=explanations,
            client_id=headers.get(settings.header_client_id),
            endpoint=path,
            http_method=method,
        )

    def _score_to_verdict(self, score: float) -> VerdictEnum:
        """Map numeric score to APPROVE, FLAG, or BLOCK."""
        if score > settings.block_threshold:
            return VerdictEnum.BLOCK
        if score >= settings.approve_threshold:
            return VerdictEnum.FLAG
        return VerdictEnum.APPROVE

    def _build_block_response(self, evaluation: SecurityEvaluationResult) -> JSONResponse:
        """Generate 403 Forbidden response with explainability payload."""
        block_payload = SecurityBlockResponse(
            verdict=VerdictEnum.BLOCK,
            anomaly_score=evaluation.anomaly_score,
            explanations=evaluation.explanations,
            request_id=evaluation.request_id,
            timestamp=evaluation.timestamp,
        )
        return JSONResponse(
            status_code=403,
            content=block_payload.model_dump(),
            headers={
                settings.header_verdict: VerdictEnum.BLOCK.value,
                "x-mesh-latency-ms": f"{evaluation.latency_ms:.2f}",
            },
        )

    async def _log_verdict(self, evaluation: SecurityEvaluationResult) -> None:
        """Persist verdict telemetry to verdicts.jsonl (dashboard sink)."""
        try:
            target_path = self.log_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(evaluation.model_dump()) + "\n"
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            await logger.aerror("verdict_logging_failed", error=str(exc))
