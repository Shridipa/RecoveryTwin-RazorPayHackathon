"""Health check endpoint."""
from fastapi import APIRouter
from datetime import datetime
from backend.app.services.data_service import data_service

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy" if data_service.is_loaded else "loading",
        "service": "RecoveryTwin API",
        "version": "1.0.0",
        "models_loaded": data_service.is_loaded,
        "timestamp": datetime.now().isoformat(),
    }
