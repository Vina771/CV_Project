"""
Wrapper de tracking ByteTrack pour vidéos, basé sur ultralytics.model.track().
"""
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import imageio_ffmpeg
from ultralytics import YOLO


def track_video(
    model: YOLO,
    input_path: str,
    output_path: str,
    conf: float = 0.25,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """
    Applique la détection + tracking ByteTrack frame par frame sur une vidéo,
    et écrit la vidéo annotée (bboxes + IDs persistants) dans output_path.

    Note : cv2.VideoWriter avec le codec mp4v produit un fichier .mp4 valide
    mais souvent illisible directement dans un <video> HTML5 (navigateur).
    Utiliser reencode_for_browser() sur le résultat avant st.video().

    progress_callback: fonction optionnelle appelée avec un float 0-1
    pour mettre à jour une barre de progression Streamlit.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    results_generator = model.track(
        source=input_path,
        conf=conf,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
    )

    for result in results_generator:
        annotated_frame = result.plot()  # dessine bboxes + IDs + labels
        writer.write(annotated_frame)

        frame_idx += 1
        if progress_callback is not None:
            progress_callback(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()

    return output_path


def reencode_for_browser(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Ré-encode une vidéo (typiquement issue de cv2.VideoWriter, codec mp4v)
    en H.264 + faststart, lisible directement dans un <video> HTML5 / st.video().

    Utilise le binaire ffmpeg embarqué par imageio-ffmpeg (installé via pip),
    donc aucune dépendance système (apt) requise.
    """
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_name(p.stem + "_web" + p.suffix))

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y", 
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not Path(output_path).exists():
        raise RuntimeError(
            f"Échec du ré-encodage ffmpeg :\n{result.stderr[-2000:]}"
        )

    return output_path