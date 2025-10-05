import base64
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends
from loguru import logger
from pydantic import BaseModel

from inference.inference import InferenceHandler, NoDetections, NoSegmentation
from registry import REGISTRY

router = APIRouter()


class ImageData(BaseModel):
    image: str  # base64 encoded image


class PredictResponse(BaseModel):
    image: str | None  # base64 encoded image
    roundness: float | None
    labels: dict[str, float] | None  # Labels with confidences


def get_inference_handler() -> InferenceHandler:
    return REGISTRY.inference_handler


@router.post("/predict")
async def predict(
    req: ImageData,
    handler: InferenceHandler = Depends(get_inference_handler),
) -> PredictResponse:
    image_bytes = base64.b64decode(req.image)
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)
        try:
            seg_result = handler.segmentation_from_imgpath(tmp_path)
            logger.info("Segmentation successful")
            segmented_img = seg_result[0]
            with open(segmented_img, "rb") as img_file:
                image = base64.b64encode(img_file.read()).decode("utf-8")
            try:
                roundness = handler.estimate_roundness_from_mask(seg_result[1])
            except Exception as e:
                logger.warning(str(e))
                roundness = None
        except NoSegmentation:
            image = None
            roundness = None
        try:
            logger.info("Detection successful")
            labels = handler.detection_from_imgpath(tmp_path)
        except NoDetections:
            labels = None
    return PredictResponse(image=image, roundness=roundness, labels=labels)
