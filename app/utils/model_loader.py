"""
Téléchargement et mise en cache des poids YOLOv8 depuis Google Drive.
"""
from pathlib import Path

import gdown
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

# IDs extraits des liens Google Drive fournis
WEIGHTS = {
    "best.pt": "1zOpySTI7k6Rcwc13aUw271D1V22-sEe6",
    "last.pt": "1DWn9li9PkObgXZzO1vWIVsLzAQ2XP6IN",
}


@st.cache_resource(show_spinner=False)
def get_weights_path(weight_name: str = "best.pt") -> str:
    """
    Retourne le chemin local du fichier de poids demandé.
    Le télécharge depuis Google Drive s'il n'existe pas déjà.
    """
    if weight_name not in WEIGHTS:
        raise ValueError(
            f"Poids inconnu: '{weight_name}'. Choix possibles: {list(WEIGHTS.keys())}"
        )

    local_path = MODELS_DIR / weight_name

    if not local_path.exists():
        file_id = WEIGHTS[weight_name]
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Téléchargement de {weight_name} depuis Google Drive..."):
            gdown.download(url, str(local_path), quiet=False)

        if not local_path.exists():
            raise RuntimeError(
                f"Échec du téléchargement de {weight_name}. "
                "Vérifie que le lien Google Drive est bien partagé "
                "en mode 'Anyone with the link'."
            )

    return str(local_path)
