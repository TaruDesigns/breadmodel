from fastapi import APIRouter

from .admin import router as admin_router
from .inference import router as inference_router

api_router = APIRouter()
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
api_router.include_router(inference_router, prefix="/predict", tags=["predict"])
