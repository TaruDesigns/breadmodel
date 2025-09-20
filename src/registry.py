from inference import InferenceHandler
from settings import SETTINGS


class Registry:
    def __init__(self) -> None:
        self.inference_handler: InferenceHandler = InferenceHandler(
            SETTINGS.detection_model_path, SETTINGS.segmentation_model_path
        )
        self.settings = SETTINGS


REGISTRY = Registry()
