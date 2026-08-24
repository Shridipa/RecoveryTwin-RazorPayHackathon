"""Analytics endpoints."""
from fastapi import APIRouter
from backend.app.services.data_service import data_service

router = APIRouter()


@router.get("/analytics/models")
async def get_model_metrics():
    return data_service.get_model_metrics()


@router.get("/analytics/segments")
async def get_segments():
    return data_service.get_segments()


@router.get("/analytics/stress-tests")
async def get_stress_tests():
    return data_service.get_stress_test_results()
