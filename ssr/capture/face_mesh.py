"""Stage 1: face + mouth tracking via MediaPipe Face Mesh."""

from dataclasses import dataclass

import numpy as np

# MediaPipe Face Mesh landmark indices for the lips (outer + inner ring),
# using refine_landmarks=True topology. Order is not contiguous; we list
# them explicitly since they're what downstream geometry math depends on.
OUTER_LIP_IDXS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
INNER_LIP_IDXS = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]
LIP_IDXS = OUTER_LIP_IDXS + INNER_LIP_IDXS

# Inner-lip top/bottom center and inner-lip left/right corners, used for
# the openness ratio (vertical gap normalized by mouth width).
INNER_TOP_CENTER = 13
INNER_BOTTOM_CENTER = 14
INNER_LEFT_CORNER = 78
INNER_RIGHT_CORNER = 308


@dataclass
class MouthLandmarks:
    points: np.ndarray  # (N, 2) pixel-space lip landmark coordinates
    openness: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


class FaceMeshTracker:
    """Wraps MediaPipe Face Mesh to extract mouth-region landmarks per frame."""

    def __init__(self, max_num_faces: int = 1, bbox_padding_ratio: float = 0.4):
        self.max_num_faces = max_num_faces
        self.bbox_padding_ratio = bbox_padding_ratio
        self._face_mesh = None

    def _lazy_init(self):
        if self._face_mesh is None:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def process(self, frame_bgr: np.ndarray) -> MouthLandmarks | None:
        self._lazy_init()
        import cv2

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0].landmark
        lip_points = np.array(
            [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in LIP_IDXS],
            dtype=np.float32,
        )

        top = face_landmarks[INNER_TOP_CENTER]
        bottom = face_landmarks[INNER_BOTTOM_CENTER]
        left = face_landmarks[INNER_LEFT_CORNER]
        right = face_landmarks[INNER_RIGHT_CORNER]

        mouth_width = np.hypot((right.x - left.x) * w, (right.y - left.y) * h)
        mouth_gap = np.hypot((top.x - bottom.x) * w, (top.y - bottom.y) * h)
        openness = float(mouth_gap / mouth_width) if mouth_width > 1e-6 else 0.0

        x_min, y_min = lip_points.min(axis=0)
        x_max, y_max = lip_points.max(axis=0)
        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_x = box_w * self.bbox_padding_ratio
        pad_y = box_h * self.bbox_padding_ratio

        x0 = max(0, int(x_min - pad_x))
        y0 = max(0, int(y_min - pad_y))
        x1 = min(w, int(x_max + pad_x))
        y1 = min(h, int(y_max + pad_y))

        bbox = (x0, y0, x1 - x0, y1 - y0)

        return MouthLandmarks(points=lip_points, openness=openness, bbox=bbox)

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
