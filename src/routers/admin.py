
from fastapi import APIRouter
from pydantic import BaseModel

from settings import SETTINGS

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


class UpdateSettingsRequest(BaseModel):
    MIN_BREAD_LABEL_CONFIDENCE: float | None = None
    MIN_BREAD_SEG_CONFIDENCE: float | None = None
    FILTER_BREAD_LABEL_CONFIDENCE: float | None = None
    FILTER_BREAD_SEG_CONFIDENCE: float | None = None
    BREAD_DETECTION_CONFIDENCE: float | None = None
    OVERRIDE_DETECTION_CONFIDENCE: float | None = None


@router.post("/update-settings")
async def set_infer_confidence(req: UpdateSettingsRequest):
    SETTINGS.min_bread_label_confidence = (
        req.MIN_BREAD_LABEL_CONFIDENCE
        if req.MIN_BREAD_LABEL_CONFIDENCE is not None
        else SETTINGS.min_bread_label_confidence
    )
    SETTINGS.min_bread_seg_confidence = (
        req.MIN_BREAD_SEG_CONFIDENCE
        if req.MIN_BREAD_SEG_CONFIDENCE is not None
        else SETTINGS.min_bread_seg_confidence
    )
    SETTINGS.filter_bread_label_confidence = (
        req.FILTER_BREAD_LABEL_CONFIDENCE
        if req.FILTER_BREAD_LABEL_CONFIDENCE is not None
        else SETTINGS.filter_bread_label_confidence
    )
    SETTINGS.filter_bread_seg_confidence = (
        req.FILTER_BREAD_SEG_CONFIDENCE
        if req.FILTER_BREAD_SEG_CONFIDENCE is not None
        else SETTINGS.filter_bread_seg_confidence
    )
    SETTINGS.bread_detection_confidence = (
        req.BREAD_DETECTION_CONFIDENCE
        if req.BREAD_DETECTION_CONFIDENCE is not None
        else SETTINGS.bread_detection_confidence
    )
    SETTINGS.override_detection_confidence = (
        req.OVERRIDE_DETECTION_CONFIDENCE
        if req.OVERRIDE_DETECTION_CONFIDENCE is not None
        else SETTINGS.override_detection_confidence
    )
    return


@router.get("/check-cuda")
async def check_cuda():
    # Reinit the inference model with new parameters
    import torch

    try:
        results = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.current_device(),
            "cuda_device_count": torch.cuda.device_count(),
        }
    except:
        results = {"cuda": "no"}
    return results
