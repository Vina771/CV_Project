from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_suffix(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return file_suffix(filename) in allowed_extensions


def save_uploaded_file(uploaded_file: BinaryIO, destination_dir: Path) -> Path:
    ensure_dir(destination_dir)
    suffix = file_suffix(getattr(uploaded_file, "name", "")) or ".bin"
    destination = destination_dir / f"upload_{uuid4().hex}{suffix}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def readable_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024**2:.1f} MB"
