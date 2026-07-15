"""Stage 5: per-user personalization (vocabulary, n-gram boosting, corrections)."""

import re
from dataclasses import dataclass, field


def _replace_phrase_ci(text: str, phrase: str, replacement: str) -> str:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    return pattern.sub(replacement, text)


@dataclass
class UserProfile:
    user_id: str
    custom_vocab: dict[str, str] = field(default_factory=dict)  # e.g. "grb bb" -> "grab boba"
    phrase_history: list[str] = field(default_factory=list)

    def apply(self, text: str) -> str:
        """Apply user-specific vocabulary substitutions to a corrected sentence."""
        result = text
        # multi-word phrases first (whole-phrase, case-insensitive match)
        for raw, corrected in self.custom_vocab.items():
            if " " not in raw:
                continue
            result = _replace_phrase_ci(result, raw, corrected)

        # then single-word substitutions, word by word
        words = result.split()
        single_word_vocab = {k.lower(): v for k, v in self.custom_vocab.items() if " " not in k}
        if single_word_vocab:
            words = [single_word_vocab.get(w.lower(), w) for w in words]
        return " ".join(words)

    def record_correction(self, raw: str, corrected: str) -> None:
        self.custom_vocab[raw] = corrected
