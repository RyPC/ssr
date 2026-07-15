"""Live camera capture: frames -> mouth ROI clips, for recording and inference."""

import time

import numpy as np

from ssr.capture.face_mesh import FaceMeshTracker
from ssr.capture.roi import crop_mouth_roi


def record_mouth_clip(
    camera_index: int = 0,
    duration_s: float = 3.0,
    roi_size: int = 96,
    quit_key: str = "q",
) -> list[np.ndarray]:
    """Open a camera, track the mouth, and collect cropped ROI frames.

    Stops after `duration_s` seconds or when `quit_key` is pressed in the
    preview window. Frames where no face is detected are skipped.
    """
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")

    tracker = FaceMeshTracker()
    roi_frames: list[np.ndarray] = []

    try:
        start = time.monotonic()
        while time.monotonic() - start < duration_s:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            landmarks = tracker.process(frame_bgr)
            if landmarks is not None:
                roi = crop_mouth_roi(frame_bgr, landmarks, size=roi_size)
                roi_frames.append(roi)

                x, y, w, h = landmarks.bbox
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.imshow("SSR capture (press q to stop)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

    return roi_frames
