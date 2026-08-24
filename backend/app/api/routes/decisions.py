"""Decision endpoints."""
from fastapi import APIRouter, HTTPException
from backend.app.services.data_service import data_service

router = APIRouter()


@router.get("/decisions/{payment_id}")
async def get_decision(payment_id: str):
    result = data_service.get_payment(payment_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Payment {payment_id} not found")

    # Extract decision-specific information
    return {
        "payment_id": result["payment_id"],
        "amount": result["amount"],
        "failure_reason": result["failure_reason"],
        "recommended_action": result.get("recommended_action", "unknown"),
        "recommended_value": result.get("recommended_value", 0),
        "actions": result.get("actions", []),
        "explanation": result.get("explanation", {}),
        "cate": {
            "retry": result.get("retry_cate"),
            "reminder": result.get("reminder_cate"),
            "alternative_method": result.get("alternative_method_cate"),
        },
    }
