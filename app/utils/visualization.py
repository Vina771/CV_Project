"""
Fonctions de visualisation des résultats de détection sur images.
"""
import numpy as np

# Couleurs BGR par classe (cohérent avec CLASS_NAMES dans inference.py)
CLASS_COLORS = {
    "NoHelmet": (0, 0, 255),      # rouge : infraction
    "WithHelmet": (0, 200, 0),    # vert : ok
    "PlateNumber": (255, 165, 0), # orange
    "Rider": (255, 0, 0),         # bleu
}


def draw_results(image: np.ndarray, result) -> np.ndarray:
    """
    Dessine les bboxes annotées par ultralytics sur l'image.
    Utilise directement result.plot() qui gère déjà couleurs/labels/scores,
    fourni pour compatibilité si un rendu custom est nécessaire plus tard.
    """
    return result.plot()


def summarize_detections(result) -> dict:
    """
    Retourne un résumé texte des détections : nombre d'objets par classe.
    Utile pour afficher des métriques rapides sous l'image dans Streamlit.
    """
    summary = {}
    if result.boxes is None:
        return summary

    names = result.names
    for cls_id in result.boxes.cls.tolist():
        cls_name = names[int(cls_id)]
        summary[cls_name] = summary.get(cls_name, 0) + 1

    return summary
