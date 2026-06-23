"""End-to-end pipeline orchestration: camera -> text output."""

from dataclasses import dataclass

import numpy as np

from ssr.capture.face_mesh import FaceMeshTracker
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
    profile: UserProfile | None = None

    def process_clip(self, frames: list[np.ndarray]) -> str:
        """Run one mouthed-phrase clip through the full pipeline and return text."""
        raise NotImplementedError
