"""
Chargement du modèle YOLOv8 et fonctions d'inférence sur image.
"""
from typing import List

import numpy as np
import streamlit as st
from ultralytics import YOLO

from .model_loader import get_weights_path

CLASS_NAMES = ["NoHelmet", "PlateNumber", "Rider", "WithHelmet"]


@st.cache_resource(show_spinner=False)
def load_model(weight_name: str = "best.pt") -> YOLO:
    """Charge (et met en cache) le modèle YOLOv8 à partir du poids demandé."""
    weights_path = get_weights_path(weight_name)
    return YOLO(weights_path)


def run_inference(model: YOLO, image: np.ndarray, conf: float = 0.25):
    """
    Lance l'inférence YOLOv8 sur une image (array BGR ou RGB numpy).
    Retourne le premier résultat (results[0]) de ultralytics.
    """
    results = model.predict(image, conf=conf, verbose=False)
    return results[0]


def run_inference_batch(model: YOLO, images: List[np.ndarray], conf: float = 0.25):
    """Lance l'inférence sur une liste d'images. Retourne la liste des résultats."""
    results = model.predict(images, conf=conf, verbose=False)
    return results
