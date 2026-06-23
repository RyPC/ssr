#!/usr/bin/env python
"""CLI entry point: run the silent speech pipeline against a live camera feed."""

import argparse

from ssr.utils.config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    config = PipelineConfig()
    config.capture.camera_index = args.camera

    raise NotImplementedError("wire up SilentSpeechPipeline and camera loop")


if __name__ == "__main__":
    main()
