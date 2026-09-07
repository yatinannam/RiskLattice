"""RiskLattice API - FastAPI application.

Synthetic data / TEST MODE only. The API orchestrates and serializes the
prepared engine state; it never executes real payment actions and never
exposes ground-truth labels.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes import audit, campaigns, containment, investigations, overview

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("risklattice.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.time()
    from app.services.app_state import get_app_state

    app.state.risklattice = get_app_state()  # singleton build once
    logger.info(
        "API startup complete in %.2fs (dataset=%s)",
        time.time() - started, settings.dataset_name,
    )
    yield


app = FastAPI(
    title="RiskLattice API",
    description=(
        "AI-powered fraud containment intelligence. Synthetic data, TEST MODE. "
        "No real payment actions. No ground-truth labels exposed."
    ),
    version="0.7.0",
    lifespan=lifespan,
)

# Local frontend development CORS (no wildcard in production-style config).
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(overview.router)
app.include_router(campaigns.router)
app.include_router(investigations.router)
app.include_router(containment.router)
app.include_router(audit.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak Python tracebacks to clients."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Unexpected server error. Check the API logs."},
    )


@app.get(
    "/health",
    response_model=dict,
    summary="Health check",
    description="Service health, dataset, and TEST MODE indicator.",
)
def health() -> dict:
    return {
        "status": "ok",
        "service": "risklattice-api",
        "version": "0.7.0",
        "dataset": settings.dataset_name,
        "mode": "TEST_MODE",
    }