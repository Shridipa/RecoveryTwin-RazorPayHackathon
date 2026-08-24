"""Payment endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.app.services.data_service import data_service

router = APIRouter()


@router.get("/payments")
async def get_payments(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    search: Optional[str] = None,
    failure_reason: Optional[str] = None,
    recommended_action: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
):
    return data_service.get_payments(
        page=page,
        page_size=page_size,
        search=search,
        failure_reason=failure_reason,
        recommended_action=recommended_action,
        min_amount=min_amount,
        max_amount=max_amount,
    )


@router.get("/payments/{payment_id}")
async def get_payment(payment_id: str):
    result = data_service.get_payment(payment_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Payment {payment_id} not found")
    return result
