from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Example fields based on typical .env.example contents
    min_bread_label_confidence: float = 0.05
    min_bread_seg_confidence: float = 0.05
    filter_bread_label_confidence: float = 0.5
    filter_bread_seg_confidence: float = 0.4
    bread_detection_confidence: float = 0.5
    override_detection_confidence: float = 0.1
    detection_model_path: Path = Path("weights/breadv7m-det.pt")
    segmentation_model_path: Path = Path("weights/breadsegv4m-seg.pt")
    debug: bool = False

    model_config = SettingsConfigDict(env_prefix="__", env_file=".env")


SETTINGS = Settings()
