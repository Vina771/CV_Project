from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from app.utils.inference import count_detections, run_image_detection
from app.utils.io_utils import ALLOWED_IMAGE_EXTENSIONS, is_allowed_file, readable_size
from app.utils.model_loader import ModelNotFoundError, load_yolo_model
from app.utils.visualization import format_detection_summary, result_to_image

st.set_page_config(page_title="Image Detection", page_icon=":frame_with_picture:", layout="wide")
st.title("Image Detection")

confidence = st.sidebar.slider("Confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
image_size = st.sidebar.select_slider("Image size", options=[320, 480, 640, 800, 960, 1280], value=960)
weight_name = st.sidebar.selectbox("Poids", options=["best.pt", "last.pt"], index=0)

uploads = st.file_uploader(
    "Glisse une ou plusieurs images",
    type=[ext.replace(".", "") for ext in sorted(ALLOWED_IMAGE_EXTENSIONS)],
    accept_multiple_files=True,
)

if uploads:
    try:
        model = load_yolo_model(weight_name)
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    for upload in uploads:
        if not is_allowed_file(upload.name, ALLOWED_IMAGE_EXTENSIONS):
            st.warning(f"Format ignore: {upload.name}")
            continue

        image = Image.open(upload).convert("RGB")
        result = run_image_detection(model, image, confidence=confidence, image_size=image_size)
        annotated = result_to_image(result)
        count = count_detections(result)

        st.subheader(Path(upload.name).name)
        st.caption(f"{readable_size(upload.size)} - {format_detection_summary(count)}")
        before, after = st.columns(2)
        before.image(image, caption="Original", use_container_width=True)
        after.image(annotated, caption="Detection", use_container_width=True)
else:
    st.info("Ajoute des images JPG, PNG, BMP ou WEBP pour lancer la detection.")
