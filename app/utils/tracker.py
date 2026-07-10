from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 960


def run_video_tracking(
    model: Any,
    video_path: str | Path,
    output_dir: str | Path,
    confidence: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    tracker: str = DEFAULT_TRACKER,
) -> Path:
    """Track objects in a video and return the rendered video path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.track(
        source=str(video_path),
        conf=confidence,
        imgsz=image_size,
        tracker=tracker,
        persist=True,
        save=True,
        project=str(output_dir),
        name="tracking",
        exist_ok=True,
        verbose=False,
    )

    saved_dir = output_dir / "tracking"
    candidates = sorted(saved_dir.glob("*.mp4")) + sorted(saved_dir.glob("*.avi"))
    if candidates:
        return candidates[0]

    # Ultralytics can preserve the source extension. Fall back to a broad search.
    rendered = [path for path in saved_dir.iterdir() if path.is_file()]
    if rendered:
        return rendered[0]

    raise RuntimeError("Aucune video de tracking n'a ete generee.")


def probe_video(video_path: str | Path) -> dict[str, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video: {video_path}")

    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration = frame_count / fps if fps else 0
    return {"frames": frame_count, "fps": fps, "width": width, "height": height, "duration": duration}
