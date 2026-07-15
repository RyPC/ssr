import torch

from ssr.decoding.beam_search import beam_search_decode

# vocab[0] is the CTC blank, per convention.
VOCAB = ["_", "a", "b"]


def _peaky_logits(symbol_indices: list[int], high: float = 10.0) -> torch.Tensor:
    """Build (T, len(VOCAB)) logits that strongly favor symbol_indices[t] at each t."""
    t_steps = len(symbol_indices)
    logits = torch.zeros(t_steps, len(VOCAB))
    for t, idx in enumerate(symbol_indices):
        logits[t, idx] = high
    return logits


def test_beam_search_recovers_top1_after_ctc_collapse():
    # symbols: a a b b _  -> collapse repeats -> "ab"
    logits = _peaky_logits([1, 1, 2, 2, 0])
    candidates = beam_search_decode(logits, VOCAB, beam_width=10, top_k=5)

    assert candidates[0].text == "ab"


def test_beam_search_returns_top_k_in_descending_score_order():
    logits = _peaky_logits([1, 1, 2, 2, 0])
    candidates = beam_search_decode(logits, VOCAB, beam_width=10, top_k=5)

    assert len(candidates) <= 5
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_blank_only_decodes_to_empty_string():
    logits = _peaky_logits([0, 0, 0])
    candidates = beam_search_decode(logits, VOCAB, beam_width=5, top_k=3)

    assert candidates[0].text == ""
