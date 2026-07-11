"""
Page Streamlit : détection + tracking ByteTrack sur vidéo.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.inference import load_model
from utils.io_utils import make_temp_output_path, save_uploaded_file_to_temp
from utils.tracker import track_video, reencode_for_browser

st.set_page_config(page_title="Video Tracking", page_icon="🎥", layout="wide")
st.title("🎥 Tracking sur vidéo (ByteTrack)")

with st.sidebar:
    st.header("Paramètres")
    weight_choice = st.selectbox("Poids du modèle", ["best.pt", "last.pt"], index=0)
    conf_threshold = st.slider("Seuil de confiance", 0.05, 0.95, 0.25, 0.05)

uploaded_video = st.file_uploader(
    "Dépose une vidéo",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=False,
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Lancer la détection + tracking", type="primary"):
        with st.spinner("Chargement du modèle..."):
            model = load_model(weight_choice)

        input_path = save_uploaded_file_to_temp(uploaded_video)
        output_path = make_temp_output_path(suffix=".mp4")

        progress_bar = st.progress(0.0, text="Traitement de la vidéo...")

        def update_progress(fraction: float) -> None:
            progress_bar.progress(fraction, text=f"Traitement... {int(fraction * 100)}%")

        try:
            result_path, stats = track_video(
                model=model,
                input_path=input_path,
                output_path=output_path,
                conf=conf_threshold,
                progress_callback=update_progress,
            )

            with st.spinner("Conversion pour lecture dans le navigateur..."):
                playable_path = reencode_for_browser(result_path)

            progress_bar.progress(1.0, text="Terminé !")

            st.success("Tracking terminé.")
            st.subheader("Vidéo annotée")
            st.video(playable_path)

            with open(playable_path, "rb") as f:
                st.download_button(
                    "Télécharger la vidéo annotée",
                    data=f,
                    file_name="video_annotee.mp4",
                    mime="video/mp4",
                )

            st.subheader("Résumé du tracking")
            par_classe = stats["par_classe"]
            if par_classe:
                st.table(
                    {
                        "Classe": list(par_classe.keys()),
                        "Objets uniques trackés": [v["objets_uniques"] for v in par_classe.values()],
                        "Détections totales (cumulées)": [v["detections_totales"] for v in par_classe.values()],
                    }
                )
                st.caption(
                    f"{stats['nb_frames']} frames traitées. "
                    "« Objets uniques » = nombre d'IDs de tracks distincts. "
                    "« Détections totales » = somme des détections sur toutes les frames."
                )
            else:
                st.write("Aucune détection sur cette vidéo.")

        except Exception as e:
            st.error(f"Erreur pendant le traitement : {e}")
else:
    st.info("Dépose une vidéo ci-dessus pour lancer la détection + tracking.")