"""
RecoveryTwin API — FastAPI Application.

Exposes the RecoveryTwin ML/financial system through REST APIs.
All data originates from existing RecoveryTwin reports and model artifacts.
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from backend.app.config import settings
from backend.app.services.data_service import data_service
from backend.app.api.routes import health, overview, payments, decisions, analytics, scenarios


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup."""
    print("[RecoveryTwin API] Loading data and reports...")
    data_service.load()
    print(f"[RecoveryTwin API] Loaded: test_data={len(data_service.test_data) if data_service.test_data is not None else 0} rows")
    print(f"[RecoveryTwin API] Decision table: {len(data_service.decision_table) if data_service.decision_table is not None else 0} rows")
    print("[RecoveryTwin API] Ready.")
    yield
    print("[RecoveryTwin API] Shutting down.")


app = FastAPI(
    title="RecoveryTwin API",
    description="Revenue Recovery Intelligence — Counterfactual ML for Payment Recovery",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend origins from env or any *.up.railway.app
import os as _os
from starlette.middleware.base import BaseHTTPMiddleware

class _CORSMiddleware(BaseHTTPMiddleware):
    """Permissive CORS that allows any origin matching the allowlist."""
    _EXACT = {
        settings.FRONTEND_ORIGIN,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    }

    @classmethod
    def _extra_origins(cls):
        extra = _os.getenv("CORS_ORIGINS", "")
        return {o.strip() for o in extra.split(",") if o.strip()}

    @classmethod
    def _is_allowed(cls, origin: str) -> bool:
        if origin in cls._EXACT:
            return True
        if origin in cls._extra_origins():
            return True
        # Allow any *.up.railway.app origin
        if origin.endswith(".up.railway.app"):
            return True
        return False

    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin", "")
        response = await call_next(request)
        if self._is_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        return response

app.add_middleware(_CORSMiddleware)

# Register routes
app.include_router(health.router, prefix="/api")
app.include_router(overview.router, prefix="/api")
app.include_router(payments.router, prefix="/api")
app.include_router(decisions.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(scenarios.router, prefix="/api")


@app.get("/")
async def root():
    return {
        "service": "RecoveryTwin API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy" if data_service.is_loaded else "loading",
    }
