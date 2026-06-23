"""Stage 5: per-user personalization (vocabulary, n-gram boosting, corrections)."""

from dataclasses import dataclass, field


@dataclass
class UserProfile:
    user_id: str
    custom_vocab: dict[str, str] = field(default_factory=dict)  # e.g. "grb bb" -> "grab boba"
    phrase_history: list[str] = field(default_factory=list)

    def apply(self, text: str) -> str:
        """Apply user-specific vocabulary substitutions to a corrected sentence."""
        raise NotImplementedError

    def record_correction(self, raw: str, corrected: str) -> None:
        self.custom_vocab[raw] = corrected
