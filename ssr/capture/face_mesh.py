"""Stage 1: face + mouth tracking via MediaPipe Face Mesh."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MouthLandmarks:
    points: np.ndarray  # (N, 2) or (N, 3) lip landmark coordinates
    openness: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


class FaceMeshTracker:
    """Wraps MediaPipe Face Mesh to extract mouth-region landmarks per frame."""

    def __init__(self, max_num_faces: int = 1):
        self.max_num_faces = max_num_faces
        self._face_mesh = None

    def _lazy_init(self):
        if self._face_mesh is None:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self.max_num_faces,
                refine_landmarks=True,
            )

    def process(self, frame_bgr: np.ndarray) -> MouthLandmarks | None:
        self._lazy_init()
        raise NotImplementedError("extract mouth landmarks from a BGR frame")
