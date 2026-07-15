import numpy as np

from ssr.correction.reranker import LanguageCorrector
from ssr.models.sequence_model import CNNLSTMLipReader
from ssr.personalization.profile import UserProfile
from ssr.pipeline import SilentSpeechPipeline
from ssr.utils.config import PipelineConfig


def test_process_clip_rois_end_to_end():
    vocab = [""] + sorted(set("can you pick up some milk "))
    model = CNNLSTMLipReader(feature_dim=32, hidden_size=16, vocab_size=len(vocab))

    pipeline = SilentSpeechPipeline(
        config=PipelineConfig(),
        tracker=None,
        model=model,
        corrector=LanguageCorrector(),
        vocab=vocab,
        profile=UserProfile(user_id="test"),
    )

    rois = [np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8) for _ in range(12)]
    text = pipeline.process_clip_rois(rois)

    assert isinstance(text, str)
    assert pipeline.profile.phrase_history == [text]


def test_process_clip_rois_empty_returns_empty_string():
    vocab = [""] + list("ab")
    model = CNNLSTMLipReader(feature_dim=16, hidden_size=8, vocab_size=len(vocab))
    pipeline = SilentSpeechPipeline(
        config=PipelineConfig(),
        tracker=None,
        model=model,
        corrector=LanguageCorrector(),
        vocab=vocab,
    )
    assert pipeline.process_clip_rois([]) == ""
