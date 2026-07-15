"""Stage 4: language correction model.

Selects/lightly corrects among top-K beam search candidates without
inventing new meaning.
"""

from ssr.correction.corpus import Corpus, DEFAULT_CORPUS
from ssr.decoding.beam_search import Candidate


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]


class LanguageCorrector:
    def __init__(self, corpus: Corpus | None = None):
        self.corpus = corpus if corpus is not None else DEFAULT_CORPUS

    def _plausibility(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        score = sum(self.corpus.word_log_freq(w) for w in words)
        score += sum(self.corpus.bigram_bonus(a, b) for a, b in zip(words, words[1:]))
        return score

    def _closest_vocab_word(self, word: str, max_distance: int = 2) -> str | None:
        candidates = []
        for vocab_word in self.corpus.vocab:
            if abs(len(vocab_word) - len(word)) > max_distance:
                continue
            dist = _levenshtein(word, vocab_word)
            if dist <= max_distance:
                candidates.append((dist, vocab_word))
        if not candidates:
            return None
        min_dist = min(d for d, _ in candidates)
        tied = [w for d, w in candidates if d == min_dist]
        # tie-break: prefer same starting letter, then higher corpus frequency
        tied.sort(key=lambda w: (w[0] != word[0], -self.corpus.word_freq[w]))
        return tied[0]

    def correct(self, candidates: list[Candidate]) -> str:
        """Return the best, lightly-corrected sentence from candidates."""
        if not candidates:
            return ""

        best_candidate = max(
            candidates,
            key=lambda c: c.score + self._plausibility(c.text),
        )

        corrected_words = []
        for word in best_candidate.text.split():
            lower = word.lower()
            if lower in self.corpus.vocab:
                corrected_words.append(word)
                continue
            replacement = self._closest_vocab_word(lower)
            corrected_words.append(replacement if replacement is not None else word)

        return " ".join(corrected_words)
