"""
Page d'accueil de l'application Streamlit.
"""
import streamlit as st

st.set_page_config(
    page_title="Motorcycle Helmet & Plate Tracking",
    page_icon="🏍️",
    layout="wide",
)

st.title("🏍️ Détection & Tracking de Motards")

st.markdown(
    """
    Bienvenue sur l'application de détection et de suivi de motards, basée sur
    **YOLOv8** et **ByteTrack**.

    ### Classes détectées
    - `NoHelmet` — motard sans casque
    - `WithHelmet` — motard avec casque
    - `PlateNumber` — plaque d'immatriculation détectée
    - `Rider` — motard détecté

    ### Comment utiliser l'application

    Utilise le menu à gauche pour choisir un mode :

    - **📷 Image Detection** : dépose une ou plusieurs images (jpg, png)
      pour lancer la détection.
    - **🎥 Video Tracking** : dépose une vidéo (mp4, mov, avi) pour lancer
      la détection + le tracking ByteTrack (IDs persistants sur la vidéo).

    Les poids du modèle (`best.pt`) sont téléchargés automatiquement depuis
    Google Drive au premier lancement, puis mis en cache localement.
    """
)

st.info(
    "👈 Choisis une page dans le menu de gauche pour commencer.",
    icon="ℹ️",
)
