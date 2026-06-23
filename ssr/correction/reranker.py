"""Stage 4: language correction model.

Selects/lightly corrects among top-K beam search candidates without
inventing new meaning.
"""

from ssr.decoding.beam_search import Candidate


class LanguageCorrector:
    def correct(self, candidates: list[Candidate]) -> str:
        """Return the best, lightly-corrected sentence from candidates."""
        raise NotImplementedError
