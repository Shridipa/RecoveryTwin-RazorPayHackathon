"""Scenario and financial endpoints."""
from fastapi import APIRouter
from backend.app.services.data_service import data_service

router = APIRouter()


@router.get("/scenarios")
async def get_scenarios():
    return data_service.get_scenarios()


@router.get("/financial/monte-carlo")
async def get_monte_carlo():
    return data_service.get_monte_carlo()


@router.get("/financial/breakeven")
async def get_breakeven():
    return data_service.get_breakeven()


@router.get("/financial/robustness")
async def get_robustness():
    return data_service.get_robustness()
