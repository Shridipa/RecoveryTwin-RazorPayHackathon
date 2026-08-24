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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
