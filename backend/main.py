"""FastAPI Main Application Entrypoint for SBP API Security Mesh.

Assembles Starlette reverse-proxy interceptor middleware,
Open Banking routers, structlog configuration, and health endpoints.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
import structlog
from backend.config import settings
from backend.middleware.interceptor import SecurityMeshInterceptor
from backend.routers.accounts import router as accounts_router
from backend.routers.consent import router as consent_router
from backend.routers.transactions import router as transactions_router
from backend.routers.transfer import router as transfer_router


def configure_logging() -> None:
    """Configure structured JSON logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


configure_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and graceful shutdown lifecycle."""
    await logger.ainfo(
        "security_mesh_starting",
        app_name=settings.app_name,
        target="SBP Regulatory Sandbox Cohort 2",
        latency_budget_ms=settings.target_latency_ms,
    )
    try:
        from ml.pipeline import register_pipeline_with_interceptor
        register_pipeline_with_interceptor()
    except Exception as exc:
        await logger.aerror("failed_to_initialize_ml_pipeline", error=str(exc))
    yield
    await logger.ainfo("security_mesh_stopped")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    mesh_app = FastAPI(
        title="AI-Powered API Security Mesh",
        description="Zero-Trust AI Reverse Proxy for Pakistan Open Banking (SBP Sandbox Cohort 2)",
        version="1.0.0",
        lifespan=lifespan,
    )

    mesh_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach reverse-proxy security interceptor
    mesh_app.add_middleware(SecurityMeshInterceptor)

    # Mount Open Banking FAPI routers
    mesh_app.include_router(consent_router)
    mesh_app.include_router(accounts_router)
    mesh_app.include_router(transactions_router)
    mesh_app.include_router(transfer_router)

    @mesh_app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint bypassing security mesh evaluation."""
        return {
            "status": "UP",
            "mesh": "ACTIVE",
            "environment": settings.environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @mesh_app.get("/", status_code=status.HTTP_200_OK, tags=["System"])
    async def root() -> Dict[str, str]:
        """Root API metadata."""
        return {
            "service": "AI-Powered API Security Mesh",
            "regulatory_sandbox": "State Bank of Pakistan Cohort 2",
            "status": "ONLINE",
        }

    return mesh_app


app = create_app()

