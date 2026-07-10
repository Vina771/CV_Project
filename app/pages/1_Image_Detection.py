"""
Page Streamlit : détection sur une ou plusieurs images.
"""
import sys
from pathlib import Path

import streamlit as st

# Permet d'importer app.utils quand la page est lancée par Streamlit multipage
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.inference import load_model, run_inference
from utils.io_utils import uploaded_file_to_array
from utils.visualization import summarize_detections

st.set_page_config(page_title="Image Detection", page_icon="📷", layout="wide")
st.title("📷 Détection sur images")

with st.sidebar:
    st.header("Paramètres")
    weight_choice = st.selectbox("Poids du modèle", ["best.pt", "last.pt"], index=0)
    conf_threshold = st.slider("Seuil de confiance", 0.05, 0.95, 0.25, 0.05)

uploaded_files = st.file_uploader(
    "Dépose une ou plusieurs images",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if uploaded_files:
    with st.spinner("Chargement du modèle..."):
        model = load_model(weight_choice)

    for uploaded_file in uploaded_files:
        st.subheader(uploaded_file.name)

        image_array = uploaded_file_to_array(uploaded_file)
        result = run_inference(model, image_array, conf=conf_threshold)
        annotated_image = result.plot()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_array, caption="Image originale", use_container_width=True)
        with col2:
            st.image(
                annotated_image,
                caption="Détections",
                use_container_width=True,
                channels="BGR",
            )

        summary = summarize_detections(result)
        if summary:
            st.write("**Résumé des détections :**")
            st.table(
                {"Classe": list(summary.keys()), "Nombre": list(summary.values())}
            )
        else:
            st.write("Aucune détection.")

        st.divider()
else:
    st.info("Dépose une ou plusieurs images ci-dessus pour lancer la détection.")
