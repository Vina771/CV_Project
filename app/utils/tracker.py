"""
Wrapper de tracking ByteTrack pour vidéos, basé sur ultralytics.model.track().
"""
from pathlib import Path
from typing import Callable, Optional

import cv2
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
    # model.track garde l'état du tracker en interne entre les appels
    # tant qu'on ne passe pas persist=False
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
