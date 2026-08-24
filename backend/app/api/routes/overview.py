"""Overview and dashboard endpoints."""
from fastapi import APIRouter
from backend.app.services.data_service import data_service

router = APIRouter()


@router.get("/overview")
async def get_overview():
    return data_service.get_overview()


@router.get("/policies")
async def get_policies():
    return data_service.get_policies()


@router.get("/actions")
async def get_actions():
    return data_service.get_action_distribution()
