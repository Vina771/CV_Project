from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def result_to_image(result: Any) -> Image.Image:
    """Convert an Ultralytics result with plotted boxes to a PIL image."""
    plotted = result.plot()
    rgb_image = np.asarray(plotted)[..., ::-1]
    return Image.fromarray(rgb_image)


def format_detection_summary(count: int) -> str:
    if count == 0:
        return "Aucune detection"
    if count == 1:
        return "1 detection"
    return f"{count} detections"
