"""Stage 3: beam search decoder over token probabilities -> top-K candidates.

CTC prefix beam search. Convention: vocab[0] is always the CTC blank token
(matches ssr/data/datasets.py's _build_char_vocab and the model's output
vocab_size). Decoded text is the literal join of surviving vocab tokens in
order -- if vocab is char-level and includes a " " token, that's how spaces
appear in output; no separator is inserted by this function.
"""

import math
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
    """Decode (T, vocab_size) token logits into the top-K text candidates.

    Standard CTC prefix beam search: at each timestep, every beam prefix is
    extended by every vocab symbol; extensions are merged according to CTC
    collapsing rules (blank is dropped, immediate repeats of the same
    non-blank symbol collapse to one occurrence unless separated by a
    blank), and the beam is pruned to `beam_width` highest-log-prob prefixes.
    """
    blank_idx = 0
    log_probs = torch.log_softmax(token_logits, dim=-1)
    t_steps, vocab_size = log_probs.shape

    # beam: dict mapping (prefix, last_symbol) -> log-probability. last_symbol
    # is the most recent non-blank symbol appended (or "" after a blank /
    # at the start), since that determines whether a repeat collapses.
    beam: dict[tuple[str, str], float] = {("", ""): 0.0}

    for t in range(t_steps):
        frame_log_probs = log_probs[t]
        next_beam: dict[tuple[str, str], float] = {}

        for (prefix, last_symbol), prefix_log_prob in beam.items():
            for v in range(vocab_size):
                symbol = vocab[v]
                step_log_prob = float(frame_log_probs[v])
                new_log_prob = prefix_log_prob + step_log_prob

                if v == blank_idx:
                    key = (prefix, "")
                elif symbol == last_symbol:
                    # repeat of the previous non-blank symbol without an
                    # intervening blank: collapses, prefix unchanged.
                    key = (prefix, symbol)
                else:
                    key = (prefix + symbol, symbol)

                if key in next_beam:
                    next_beam[key] = _log_add(next_beam[key], new_log_prob)
                else:
                    next_beam[key] = new_log_prob

        # prune to beam_width highest-probability (prefix, last_symbol) entries
        pruned = sorted(next_beam.items(), key=lambda kv: kv[1], reverse=True)[:beam_width]
        beam = dict(pruned)

    # merge entries that share the same final prefix (different last_symbol)
    merged: dict[str, float] = {}
    for (prefix, _last_symbol), log_prob in beam.items():
        if prefix in merged:
            merged[prefix] = _log_add(merged[prefix], log_prob)
        else:
            merged[prefix] = log_prob

    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [Candidate(text=text, score=score) for text, score in ranked]


def _log_add(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) computed in a numerically stable way."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))
