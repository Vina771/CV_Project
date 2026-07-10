from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import requests
from ultralytics import YOLO

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
DEFAULT_WEIGHTS: Final[tuple[str, ...]] = ("best.pt", "last.pt")


class ModelNotFoundError(FileNotFoundError):
    """Raised when no local model exists and no download URL is configured."""


def _url_for_weight(weight_name: str) -> str | None:
    normalized = weight_name.upper().replace(".", "_")
    return (
        os.getenv(f"CV_{normalized}_MODEL_URL")
        or os.getenv(f"CV_{Path(weight_name).stem.upper()}_MODEL_URL")
        or os.getenv("CV_MODEL_URL")
    )


def download_model(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as model_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    model_file.write(chunk)
    return destination


def resolve_model_path(preferred_weight: str = "best.pt") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = (preferred_weight, *[name for name in DEFAULT_WEIGHTS if name != preferred_weight])

    for name in candidates:
        path = MODELS_DIR / name
        if path.exists():
            return path

    for name in candidates:
        url = _url_for_weight(name)
        if url:
            return download_model(url, MODELS_DIR / name)

    raise ModelNotFoundError(
        "Aucun modele YOLO trouve. Ajoute models/best.pt ou models/last.pt, "
        "ou configure CV_BEST_MODEL_URL / CV_LAST_MODEL_URL / CV_MODEL_URL."
    )


def load_yolo_model(preferred_weight: str = "best.pt") -> YOLO:
    return YOLO(str(resolve_model_path(preferred_weight)))
