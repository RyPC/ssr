#!/usr/bin/env python
"""CLI entry point: run the silent speech pipeline against a live camera feed.

Press 'q' in the preview window (or wait out --duration) to end a clip;
the recognized text is printed, then recording starts again for the next
clip. Ctrl+C to exit.
"""

import argparse
import json
from pathlib import Path

import torch

from ssr.capture.camera import record_mouth_clip
from ssr.capture.face_mesh import FaceMeshTracker
from ssr.correction.reranker import LanguageCorrector
from ssr.models.sequence_model import CNNLSTMLipReader
from ssr.personalization.profile import UserProfile
from ssr.pipeline import SilentSpeechPipeline
from ssr.utils.config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="model.pt", help="trained model checkpoint")
    parser.add_argument("--vocab", default=None, help="vocab json (default: <model>.vocab.json)")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds per clip")
    parser.add_argument("--user-id", default="default")
    args = parser.parse_args()

    config = PipelineConfig()
    config.capture.camera_index = args.camera

    vocab_path = Path(args.vocab) if args.vocab else Path(args.model).with_suffix(".vocab.json")
    vocab = json.loads(vocab_path.read_text())

    model = CNNLSTMLipReader(vocab_size=len(vocab))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    pipeline = SilentSpeechPipeline(
        config=config,
        tracker=FaceMeshTracker(),
        model=model,
        corrector=LanguageCorrector(),
        vocab=vocab,
        profile=UserProfile(user_id=args.user_id),
    )

    print("Mouth your phrase when recording starts. Ctrl+C to exit.")
    try:
        while True:
            input("Press Enter to record a clip...")
            raw_frames = record_mouth_clip(
                camera_index=args.camera,
                duration_s=args.duration,
                roi_size=config.capture.mouth_roi_size,
            )
            if not raw_frames:
                print("No mouth detected — try again.")
                continue
            # record_mouth_clip already returns cropped ROI frames, so feed
            # them straight to the model rather than re-tracking raw frames.
            text = pipeline.process_clip_rois(raw_frames)
            print(f">> {text}")
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
