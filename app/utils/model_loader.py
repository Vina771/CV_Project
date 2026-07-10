from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlparse

import requests

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
DEFAULT_WEIGHTS: Final[tuple[str, ...]] = ("best.pt", "last.pt")
DEFAULT_MODEL_URLS: Final[dict[str, str]] = {
    "best.pt": "https://drive.google.com/file/d/1zOpySTI7k6Rcwc13aUw271D1V22-sEe6/view?usp=sharing",
    "last.pt": "https://drive.google.com/file/d/1DWn9li9PkObgXZzO1vWIVsLzAQ2XP6IN/view?usp=sharing",
}


class ModelNotFoundError(FileNotFoundError):
    """Raised when no local model exists and no download URL is configured."""


def _url_for_weight(weight_name: str) -> str | None:
    normalized = weight_name.upper().replace(".", "_")
    return (
        os.getenv(f"CV_{normalized}_MODEL_URL")
        or os.getenv(f"CV_{Path(weight_name).stem.upper()}_MODEL_URL")
        or os.getenv("CV_MODEL_URL")
        or DEFAULT_MODEL_URLS.get(weight_name)
    )


def _to_direct_google_drive_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc not in {"drive.google.com", "www.drive.google.com"}:
        return url

    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "file" and parts[1] == "d":
        return f"https://drive.google.com/uc?export=download&id={parts[2]}"

    query = parse_qs(parsed.query)
    file_ids = query.get("id")
    if file_ids:
        return f"https://drive.google.com/uc?export=download&id={file_ids[0]}"

    return url


def _is_google_drive_url(url: str) -> bool:
    return urlparse(url).netloc in {"drive.google.com", "www.drive.google.com"}


def download_model(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_google_drive_url(url):
        import gdown

        gdown.download(url=_to_direct_google_drive_url(url), output=str(destination), quiet=False, fuzzy=True)
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        raise RuntimeError(f"Le telechargement Google Drive a echoue: {url}")

    url = _to_direct_google_drive_url(url)
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


def load_yolo_model(preferred_weight: str = "best.pt"):
    from ultralytics import YOLO

    return YOLO(str(resolve_model_path(preferred_weight)))
