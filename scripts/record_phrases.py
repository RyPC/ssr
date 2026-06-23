#!/usr/bin/env python
"""CLI: record self-labeled phrase clips for the Phase 1 custom dataset."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="ssr/data/raw")
    parser.add_argument("--phrase", required=True)
    args = parser.parse_args()

    raise NotImplementedError("capture frames for a single phrase and save to out-dir")


if __name__ == "__main__":
    main()
