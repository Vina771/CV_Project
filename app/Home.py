from __future__ import annotations

from pathlib import Path

import streamlit as st

APP_TITLE = "Motorcycle Detection & Tracking"

st.set_page_config(page_title=APP_TITLE, page_icon=":motorcycle:", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[1]

st.title(APP_TITLE)
st.caption("YOLO + Streamlit pour analyser des images et videos de motos.")

left, right = st.columns([2, 1])

with left:
    st.subheader("Flux de travail")
    st.markdown(
        """
        1. Ajoute `best.pt` ou `last.pt` dans `models/`.
        2. Ouvre la page **Image Detection** pour traiter une ou plusieurs images.
        3. Ouvre la page **Video Tracking** pour suivre les motos dans une video.
        """
    )

with right:
    st.subheader("Chemins utiles")
    st.code(f"Modeles: {ROOT_DIR / 'models'}")
    st.code("streamlit run app/Home.py")

st.info(
    "Si aucun modele local n'est trouve, l'application peut utiliser une URL definie via "
    "CV_BEST_MODEL_URL, CV_LAST_MODEL_URL ou CV_MODEL_URL."
)
