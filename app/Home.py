from __future__ import annotations

from pathlib import Path

import streamlit as st

APP_TITLE = "Motorcycle Detection & Tracking"
AUTHOR = "Vina Raharitsifa"
PROGRAM = "M1 I2AD, INSI"

st.set_page_config(page_title=APP_TITLE, page_icon=":motorcycle:", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[1]

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    .hero {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 1.4rem 1.5rem;
        background: #f8fafc;
    }
    .hero h1 {
        margin: 0 0 .35rem 0;
        font-size: 2.2rem;
        line-height: 1.15;
    }
    .hero p {
        margin: .2rem 0;
        color: #475569;
        font-size: 1.02rem;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .75rem;
        margin-top: 1rem;
    }
    .metric-box {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: .8rem;
        background: white;
    }
    .metric-box strong {
        display: block;
        color: #0f172a;
    }
    .metric-box span {
        color: #64748b;
        font-size: .9rem;
    }
    @media (max-width: 760px) {
        .metric-row {
            grid-template-columns: 1fr;
        }
        .hero h1 {
            font-size: 1.65rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <section class="hero">
        <h1>{APP_TITLE}</h1>
        <p>Detection de motos sur images et tracking video avec YOLO, ByteTrack et Streamlit.</p>
        <p>{AUTHOR} - {PROGRAM}</p>
        <div class="metric-row">
            <div class="metric-box"><strong>Images</strong><span>Upload multiple et boites annotees</span></div>
            <div class="metric-box"><strong>Videos</strong><span>Tracking avec rendu exporte</span></div>
            <div class="metric-box"><strong>Modeles</strong><span>best.pt / last.pt depuis Google Drive</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1])

with left:
    st.subheader("Demarrage")
    st.markdown(
        """
        1. Ouvre **Image Detection** pour traiter une ou plusieurs images.
        2. Ouvre **Video Tracking** pour suivre les motos dans une video.
        3. Si `models/best.pt` ou `models/last.pt` n'existe pas, l'app le telecharge automatiquement.
        """
    )

with right:
    st.subheader("Projet")
    st.write(f"**Auteur:** {AUTHOR}")
    st.write(f"**Formation:** {PROGRAM}")
    st.code(f"Modeles: {ROOT_DIR / 'models'}")
    st.code("streamlit run app/Home.py")

st.info(
    "Les liens Google Drive de best.pt et last.pt sont deja configures. "
    "Tu peux toujours les remplacer via CV_BEST_MODEL_URL, CV_LAST_MODEL_URL ou CV_MODEL_URL."
)
