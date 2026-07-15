#!/usr/bin/env python
"""CLI: record self-labeled phrase clips for the Phase 1 custom dataset."""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from ssr.capture.camera import record_mouth_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="ssr/data/raw")
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=3.0, help="seconds per clip")
    parser.add_argument("--roi-size", type=int, default=96)
    args = parser.parse_args()

    print(f"Recording '{args.phrase}' for {args.duration}s — mouth the phrase now...")
    roi_frames = record_mouth_clip(
        camera_index=args.camera,
        duration_s=args.duration,
        roi_size=args.roi_size,
    )

    if not roi_frames:
        print("No mouth detected in any frame — nothing saved.")
        return

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_dir = Path(args.out_dir) / args.phrase / timestamp
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_path = clip_dir / "frames.npy"
    np.save(clip_path, np.stack(roi_frames))

    print(f"Saved {len(roi_frames)} frames to {clip_path}")


if __name__ == "__main__":
    main()
