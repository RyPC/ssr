"""Crop and normalize the mouth region-of-interest (ROI) from a frame."""

import numpy as np

from ssr.capture.face_mesh import MouthLandmarks


def crop_mouth_roi(frame_bgr: np.ndarray, landmarks: MouthLandmarks, size: int = 96) -> np.ndarray:
    """Crop, square, and resize the mouth region to (size, size, 3).

    Returns RGB (not BGR) since downstream model tensors expect RGB.
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    x, y, box_w, box_h = landmarks.bbox

    cx, cy = x + box_w / 2, y + box_h / 2
    half = max(box_w, box_h) / 2

    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(w, cx + half))
    y1 = int(min(h, cy + half))

    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame_bgr

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.uint8)
