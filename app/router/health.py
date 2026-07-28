from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/health", tags=["Health"])

class HealthResponse(BaseModel):
    status: str = "ok"

@router.get("", response_model=HealthResponse)
async def get_health():
    return HealthResponse(status="ok")
