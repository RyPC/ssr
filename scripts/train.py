#!/usr/bin/env python
"""CLI: train the visual sequence model (CNN+LSTM MVP) on a phrase dataset."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    raise NotImplementedError("training loop for CNNLSTMLipReader")


if __name__ == "__main__":
    main()
