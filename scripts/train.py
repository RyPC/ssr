#!/usr/bin/env python
"""CLI: train the visual sequence model (CNN+LSTM MVP) on a phrase dataset."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ssr.data.datasets import CustomPhraseDataset, pad_collate
from ssr.models.sequence_model import CNNLSTMLipReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="model.pt")
    args = parser.parse_args()

    dataset = CustomPhraseDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)

    model = CNNLSTMLipReader(vocab_size=len(dataset.vocab))
    criterion = torch.nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            logits = model(batch["frames"])  # (B, T, vocab_size)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, B, vocab_size)

            loss = criterion(
                log_probs,
                batch["labels"],
                batch["frame_lengths"],
                batch["label_lengths"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}")

    torch.save(model.state_dict(), args.out)
    vocab_path = Path(args.out).with_suffix(".vocab.json")
    vocab_path.write_text(json.dumps(dataset.vocab))
    print(f"saved model to {args.out}")
    print(f"saved vocab to {vocab_path}")


if __name__ == "__main__":
    main()
