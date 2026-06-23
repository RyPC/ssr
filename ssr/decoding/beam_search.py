"""Stage 3: beam search decoder over token probabilities -> top-K candidates."""

from dataclasses import dataclass

import torch


@dataclass
class Candidate:
    text: str
    score: float


def beam_search_decode(
    token_logits: torch.Tensor,
    vocab: list[str],
    beam_width: int = 10,
    top_k: int = 5,
) -> list[Candidate]:
    """Decode (T, vocab_size) token logits into the top-K text candidates."""
    raise NotImplementedError
