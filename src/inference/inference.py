import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO
from ultralytics.engine.results import Results

from settings import SETTINGS


class NoDetections(Exception): ...


class NoSegmentation(Exception): ...


class InferenceHandler:
    """Main Inference class, works for both local and http requests (roboflow) based on the input parameter. Default is local model"""

    def __init__(
        self,
        det_model: Path = Path("../weights/breadv7m-det.pt"),
        seg_model: Path = Path("../weights/breadsegv4m-seg.pt"),
    ):
        logger.info("Loading inference models: Local")

        self.local_det_model = YOLO(det_model)
        self.local_seg_model = YOLO(seg_model)

    def segmentation_from_imgpath(
        self, input_img_path: Path, confidence: float | None = None
    ) -> tuple[Path, list[Results]]:
        """Creates segmentation image
        Uses model from https://universe.roboflow.com/school-dkgod/bread-segmentation-hfhm8/model/4
        Creates a new image that has the segmented bread drawn over it. Returns image path and the results (json object)
        Args:
            input_img (str, optional): Input image path. Defaults to None.
            output_img (str, optional): Output image path to be written to. Defaults to None: If None, it will be saved on default location.

        Raises:
            ValueError: Raise error if input image is not provided

        Returns:
            output_img_path: path of the saved file (will match output_img_path if it's not null)
            result: json object with the results from the API
        """

        if confidence is None:
            confidence = SETTINGS.min_bread_seg_confidence
            # Default to standard output folder
        outputfolder = Path(os.path.join(os.getcwd(), "output", "segmented"))
        os.makedirs(outputfolder, exist_ok=True)
        output_img_path = outputfolder / input_img_path.name

        # Get inference and file from computing with local YOLOv8 model
        logger.info(f"Computing local inference for segmentation: {input_img_path}")
        results: list[Results] = self.local_seg_model.predict(
            input_img_path,
            save=True,
            device="cpu",
            conf=confidence,
            project="output",
            name="segmented",
            exist_ok=True,
            show_labels=False,
            show_conf=False,
            show_boxes=False,
        )
        if results and results[0].masks is not None:
            # It's saved when running predict but we can't set the path directly so we retrieve it here and moved to destination path
            temp_filepath = os.path.join(
                str(results[0].save_dir), os.path.basename(results[0].path)
            )
            shutil.move(temp_filepath, output_img_path)
            return output_img_path, results
        else:
            raise NoSegmentation()

    def detection_from_imgpath(
        self, input_img_path: Path, confidence: float | None = None
    ) -> dict[str, float]:
        """Use detection model to get labels

        Args:
            input_img_path (Path): Input image path
            confidence (float | None, optional): Confidence filter. Defaults to None.

        Returns:
            Dict[str, float]: Dict of labels with confidences
        """
        if confidence is None:
            confidence = SETTINGS.min_bread_label_confidence
        logger.info(f"Computing label predictions for: {input_img_path}")
        results = self.local_det_model.predict(
            input_img_path,
            save=False,
            device="cpu",
            conf=confidence,
        )
        if results:
            resulttojson = json.loads(
                results[0].to_json()
            )  # We already check if there's any and we only expect one image when we call this
            predictions = {
                prediction["name"]: prediction["confidence"]
                for prediction in resulttojson
            }
        else:
            raise NoDetections()
        logger.info(f"Label Predictions: {predictions}")
        return predictions

    def map_confidence_to_sentiment(self, confidence: float, label: str) -> str:
        # TODO: This is responsbility of the bot
        """Translate a confidence percentage to a text to indicate how accurate the element is

        Args:
            confidence (float): Confidence value
            label (str): Label for the confidence

        Returns:
            str: Value for the confidence specified
        """
        label = label.replace("_", " ")
        if confidence < 0.5:
            return f"{label}, H E L P, "
        elif confidence < 0.6:
            return f", just a bit {label}"
        elif confidence < 0.7:
            return f"reasonably {label}"
        elif confidence < 0.8:
            return f"probably {label}"
        elif confidence < 0.9:
            return f"fairly confident that it's {label}"
        elif confidence < 1.0:
            return f"pretty sure it is {label}"
        else:
            return f"Confirmed that it's {label}"

    def get_message_content_from_labels(
        self, predictions: dict[str, float], min_confidence: float | None = None
    ) -> str:
        # TODO: This is responsbility of the bot
        """Generate a message based on the input labels

        Args:
            labels (list[str], optional): Input labels. Defaults to None.

        Returns:
            str: Generated message to be used when sending
        """
        if min_confidence is None:
            min_confidence = SETTINGS.filter_bread_label_confidence
        labeltext = "This is certainly bread! "
        for label, confidence in predictions.items():
            if confidence >= min_confidence:
                labeltext = (
                    labeltext
                    + self.map_confidence_to_sentiment(
                        confidence=confidence, label=label
                    )
                    + " "
                )
        logger.debug(labeltext)
        return labeltext

    def estimate_roundness_from_mask(self, results: list[Results]) -> float:
        """Estimates roundness from mask, based on how close it is to a perfect circunscribed circle"""
        if results and results[0].masks is not None:
            orig_shape = results[0].masks.orig_shape
            mask = results[0].masks.xy
        else:
            raise ValueError("No masks to read")
        if orig_shape is None:
            raise ValueError("Orig Shape is mandatory")
        if mask is None:
            raise ValueError("Mask is mandatory")
        black_image = np.zeros(
            (orig_shape[0], orig_shape[1], 3), dtype=np.uint8
        )  # Empty image to draw lines and use to estimate contours
        cv2.polylines(black_image, np.int32([mask]), True, (255, 255, 255), 2)
        gray_image = cv2.cvtColor(
            black_image, cv2.COLOR_BGR2GRAY
        )  # Polylines turns the image to BGR so we need to turn it back to grayscale
        # Find contours in the mask
        contours, _ = cv2.findContours(
            gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Assuming there's only one closed contour
        if contours:
            contour = contours[0]
            # Calculate area of the contour
            area = cv2.contourArea(contour)
            # Calculate minimum enclosing circle and its area
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius**2
            # Calculate roundness (ratio of contour area to area of minimum enclosing circle)
            roundness = area / circle_area
            return roundness
        else:
            return None

    def get_message_from_roundness(self, roundness: float):
        if roundness is None:
            return "I don't think this bread is round at all..."
        messagecontent = f"This bread seems {round(roundness * 100, 2):.2f}% round. Anything over an 80% is pretty close to a sphere!"
        logger.debug(messagecontent)
        return messagecontent


if __name__ == "__main__":
    inferhandler = InferenceHandler()

    input_img = Path("downloads/IMG_3904.png")
    output_img = "output/segmented/roboresult.png"

    labels = inferhandler.detection_from_imgpath(input_img_path=input_img)
    if "bread" in labels.keys():
        _, result = inferhandler.segmentation_from_imgpath(input_img_path=input_img)
        print("A")
