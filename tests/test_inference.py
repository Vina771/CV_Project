"""
Tests basiques pour vérifier le chargement du modèle et l'inférence.
Nécessite une connexion internet (téléchargement des poids depuis Google Drive).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from utils.inference import load_model, run_inference  # noqa: E402


def test_load_model():
    model = load_model("best.pt")
    assert model is not None


def test_run_inference_on_blank_image():
    model = load_model("best.pt")
    blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
    result = run_inference(model, blank_image, conf=0.25)
    assert result is not None


if __name__ == "__main__":
    test_load_model()
    test_run_inference_on_blank_image()
    print("Tous les tests sont passés.")
