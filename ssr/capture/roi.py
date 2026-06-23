"""Crop and normalize the mouth region-of-interest (ROI) from a frame."""

import numpy as np

from ssr.capture.face_mesh import MouthLandmarks


def crop_mouth_roi(frame_bgr: np.ndarray, landmarks: MouthLandmarks, size: int = 96) -> np.ndarray:
    """Crop, square, and resize the mouth region to (size, size, 3)."""
    raise NotImplementedError
