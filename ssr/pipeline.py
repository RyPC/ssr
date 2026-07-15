"""End-to-end pipeline orchestration: camera -> text output."""

from dataclasses import dataclass

import numpy as np
import torch

from ssr.capture.face_mesh import FaceMeshTracker
from ssr.capture.roi import crop_mouth_roi
from ssr.correction.reranker import LanguageCorrector
from ssr.decoding.beam_search import beam_search_decode
from ssr.models.sequence_model import CNNLSTMLipReader
from ssr.personalization.profile import UserProfile
from ssr.utils.config import PipelineConfig


@dataclass
class SilentSpeechPipeline:
    config: PipelineConfig
    tracker: FaceMeshTracker
    model: CNNLSTMLipReader
    corrector: LanguageCorrector
    vocab: list[str]
    profile: UserProfile | None = None

    def _frames_to_rois(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Track + crop raw BGR frames into mouth ROIs, dropping frames with no face."""
        rois = []
        for frame_bgr in frames:
            landmarks = self.tracker.process(frame_bgr)
            if landmarks is None:
                continue
            rois.append(crop_mouth_roi(frame_bgr, landmarks, size=self.config.capture.mouth_roi_size))
        return rois

    def process_clip(self, frames: list[np.ndarray]) -> str:
        """Run one mouthed-phrase clip (raw BGR camera frames) through the full
        pipeline and return the final, personalized text."""
        rois = self._frames_to_rois(frames)
        if not rois:
            return ""
        return self.process_clip_rois(rois)

    def process_clip_rois(self, rois: list[np.ndarray]) -> str:
        """Same as process_clip, but skips face tracking — for callers (like
        ssr.capture.camera.record_mouth_clip) that already produced cropped,
        RGB mouth ROI frames."""
        if not rois:
            return ""

        clip = torch.from_numpy(np.stack(rois)).float() / 255.0  # (T, H, W, 3)
        clip = clip.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(clip)[0]  # (T, vocab_size)

        candidates = beam_search_decode(
            logits,
            self.vocab,
            beam_width=self.config.decoding.beam_width,
            top_k=self.config.decoding.top_k,
        )
        text = self.corrector.correct(candidates)

        if self.profile is not None:
            text = self.profile.apply(text)
            self.profile.phrase_history.append(text)

        return text
