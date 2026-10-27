
"""Connect image -> detector -> severity engine -> SQLite."""

import sys
from pathlib import Path
import cv2

# Supports importing src.pipeline or running from the repository root.
SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from severity import evaluate, load_config
from storage import save_result


def draw_zone(image_bgr, zone):
    """Draw the critical inspection zone."""
    height, width = image_bgr.shape[:2]
    output = image_bgr.copy()

    cv2.rectangle(
        output,
        (int(zone[0] * width), int(zone[1] * height)),
        (int(zone[2] * width), int(zone[3] * height)),
        (0, 165, 255),
        1,
    )
    return output


def inspect(
    image_bgr,
    image_name,
    cfg=None,
    profile="Standard",
    save=True,
    detector_fn=None,
    pixels_per_mm=None,
    specimen=None,
):
    """
    Inspect a steel image.

    pixels_per_mm:
        Supply only a valid calibration for this imaging setup.
        If omitted, physical defect dimensions remain unavailable.

    specimen:
        Optional dictionary of known steel specifications.
    """
    if detector_fn is None:
        from detector import detect as detector_fn

    cfg = cfg or load_config(profile=profile)

    defects, annotated, inference_ms = detector_fn(
        image_bgr,
        min_conf=cfg["min_confidence"],
    )

    # Ensure every detection contains pixel dimensions.
    # This also supports detectors that return only box coordinates.
    for defect in defects:
        box = defect.get("box", [0, 0, 0, 0])
        x1, y1, x2, y2 = box

        width_px = defect.get("width_px", max(0, x2 - x1))
        height_px = defect.get("height_px", max(0, y2 - y1))

        defect["width_px"] = round(float(width_px), 1)
        defect["height_px"] = round(float(height_px), 1)

        # Physical measurements require a valid calibration.
        defect["width_mm"] = None
        defect["height_mm"] = None

        if pixels_per_mm is not None and pixels_per_mm > 0:
            defect["width_mm"] = round(
                defect["width_px"] / pixels_per_mm, 2
            )
            defect["height_mm"] = round(
                defect["height_px"] / pixels_per_mm, 2
            )

    result = evaluate(defects, cfg)

    # Preserve the dimensions in the severity results.
    result["annotated"] = draw_zone(
        annotated, cfg["critical_zone"]
    )
    result["inference_ms"] = round(float(inference_ms), 1)
    result["specimen"] = specimen or {}

    result["id"] = (
        save_result(image_name, result, profile)
        if save
        else None
    )

    return result
