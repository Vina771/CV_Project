from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 960


def run_image_detection(
    model: Any,
    image: Image.Image | str | Path,
    confidence: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Any:
    """Run YOLO prediction on one image and return the first result."""
    results = model.predict(source=image, conf=confidence, imgsz=image_size, verbose=False)
    if not results:
        raise RuntimeError("Le modele n'a retourne aucun resultat.")
    return results[0]


def count_detections(result: Any) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    return len(boxes)
