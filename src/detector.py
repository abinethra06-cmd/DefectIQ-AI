
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# Project root is the parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Load the trained model once, not on every image.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained YOLO weights not found: {MODEL_PATH}"
    )

_model = YOLO(str(MODEL_PATH))


def detect(image_bgr, min_conf=0.25):
    """
    Detect steel surface defects and calculate bounding-box dimensions.

    Returns:
        defects: list of defect dictionaries
        annotated: image with defect labels, boxes and pixel dimensions
        inference_ms: inference time in milliseconds
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty.")

    if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    height, width = image_bgr.shape[:2]

    start_time = time.perf_counter()
    result = _model.predict(
        source=image_bgr,
        conf=float(min_conf),
        verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - start_time) * 1000

    defects = []

    if result.boxes is not None:
        for detection in result.boxes:
            class_id = int(detection.cls.item())

            # Get the predicted defect class and confidence.
            defect_type = _model.names[class_id]
            confidence = float(detection.conf.item())

            x1, y1, x2, y2 = (
                float(value)
                for value in detection.xyxy[0].tolist()
            )

            # Keep coordinates inside the image boundaries.
            x1 = max(0.0, min(x1, width))
            x2 = max(0.0, min(x2, width))
            y1 = max(0.0, min(y1, height))
            y2 = max(0.0, min(y2, height))

            # Calculate bounding-box dimensions in pixels.
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)

            width_px = round(box_width, 1)
            height_px = round(box_height, 1)

            area_ratio = (
                (box_width * box_height) / (width * height)
                if width > 0 and height > 0
                else 0.0
            )

            # Return defect information to the severity pipeline.
            defects.append({
                "type": defect_type,
                "confidence": round(confidence, 3),
                "box": [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ],
                "box_norm": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
                "area_ratio": area_ratio,
                "width_px": width_px,
                "height_px": height_px,
            })

    # Draw YOLO's original detection boxes and class labels.
    annotated = result.plot()

    # Add bounding-box width and height to the annotated image.
    for defect in defects:
        x1, y1, x2, y2 = defect["box"]

        label = (
            f'{defect["type"]} '
            f'W:{defect["width_px"]:.0f}px '
            f'H:{defect["height_px"]:.0f}px'
        )

        # Place the measurement label above the box when possible.
        cv2.putText(
            annotated,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return defects, annotated, round(inference_ms, 1)
