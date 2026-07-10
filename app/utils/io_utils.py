"""
Utilitaires pour la gestion des fichiers uploadés (images/vidéos) et fichiers temporaires.
"""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def is_video_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def uploaded_file_to_array(uploaded_file) -> np.ndarray:
    """Convertit un fichier uploadé Streamlit (image) en array numpy RGB."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def save_uploaded_file_to_temp(uploaded_file) -> str:
    """
    Sauvegarde un fichier uploadé Streamlit (typiquement une vidéo) dans un
    fichier temporaire sur disque, et retourne son chemin.
    """
    suffix = Path(uploaded_file.name).suffix
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded_file.read())
    tmp_file.close()
    return tmp_file.name


def make_temp_output_path(suffix: str = ".mp4") -> str:
    """Crée un chemin de fichier temporaire pour écrire une vidéo de sortie."""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.close()
    return tmp_file.name
