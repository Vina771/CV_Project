from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.utils.io_utils import ALLOWED_VIDEO_EXTENSIONS, is_allowed_file, readable_size, save_uploaded_file
from app.utils.model_loader import ModelNotFoundError, load_yolo_model
from app.utils.tracker import probe_video, run_video_tracking

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = PROJECT_ROOT / "outputs" / "streamlit"
UPLOAD_DIR = WORK_DIR / "uploads"
TRACKING_DIR = WORK_DIR / "tracking"

st.set_page_config(page_title="Video Tracking", page_icon=":film_frames:", layout="wide")
st.title("Video Tracking")

confidence = st.sidebar.slider("Confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
image_size = st.sidebar.select_slider("Image size", options=[320, 480, 640, 800, 960, 1280], value=960)
weight_name = st.sidebar.selectbox("Poids", options=["best.pt", "last.pt"], index=0)
tracker_name = st.sidebar.selectbox("Tracker", options=["bytetrack.yaml", "botsort.yaml"], index=0)

upload = st.file_uploader(
    "Glisse une video",
    type=[ext.replace(".", "") for ext in sorted(ALLOWED_VIDEO_EXTENSIONS)],
    accept_multiple_files=False,
)

if upload:
    if not is_allowed_file(upload.name, ALLOWED_VIDEO_EXTENSIONS):
        st.error("Format video non supporte.")
        st.stop()

    video_path = save_uploaded_file(upload, UPLOAD_DIR)
    st.caption(f"{Path(upload.name).name} - {readable_size(upload.size)}")
    st.video(str(video_path))

    try:
        info = probe_video(video_path)
        st.write(
            f"Resolution: {int(info['width'])}x{int(info['height'])} | "
            f"FPS: {info['fps']:.1f} | Duree: {info['duration']:.1f}s"
        )
    except RuntimeError as exc:
        st.warning(str(exc))

    if st.button("Lancer le tracking", type="primary"):
        try:
            model = load_yolo_model(weight_name)
            with st.spinner("Tracking en cours..."):
                rendered_path = run_video_tracking(
                    model,
                    video_path,
                    TRACKING_DIR,
                    confidence=confidence,
                    image_size=image_size,
                    tracker=tracker_name,
                )
            st.success("Tracking termine")
            st.video(str(rendered_path))
        except ModelNotFoundError as exc:
            st.error(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))
else:
    st.info("Ajoute une video MP4, AVI, MOV, MKV ou WEBM pour lancer le tracking.")
