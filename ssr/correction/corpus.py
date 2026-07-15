"""Built-in offline language corpus for plausibility scoring.

This is a small, hand-written toy corpus (not a real dataset) used to
bootstrap word/bigram frequency statistics without network access or
pretrained weights. Drop a text file at CORPUS_FILE_PATH (one sentence
per line) to use a real corpus instead — it takes priority if present.
"""

from collections import Counter
from pathlib import Path

CORPUS_FILE_PATH = Path(__file__).parent / "corpus.txt"

_SENTENCES = [
    "can you pick up some milk",
    "can you pick up some bread",
    "please pick up the kids",
    "can you grab some coffee",
    "thank you for the help",
    "thank you very much",
    "can you pass the salt",
    "i need some water please",
    "can you open the door",
    "please close the window",
    "i would like some tea",
    "can we talk later",
    "see you tomorrow morning",
    "good morning to you",
    "good night and see you soon",
    "can you turn off the light",
    "please turn on the music",
    "i am going to the store",
    "can you call me back",
    "please send me the file",
    "thank you so much for everything",
    "i will be there soon",
    "can you help me with this",
    "please wait for me here",
    "i love you very much",
    "see you later today",
    "can you bring some snacks",
    "please bring your laptop",
    "i need to go now",
    "can you drive me home",
    "thank you for waiting",
    "please give me a minute",
    "can you hear me now",
    "i can see you clearly",
    "please come over here",
    "can you stop the car",
    "thank you for coming",
    "i will pick up the milk",
    "please pick up some milk and bread",
    "can you get some milk please",
]


def _load_sentences() -> list[str]:
    if CORPUS_FILE_PATH.exists():
        lines = CORPUS_FILE_PATH.read_text().splitlines()
        return [line.strip() for line in lines if line.strip()]
    return _SENTENCES


class Corpus:
    def __init__(self, sentences: list[str] | None = None):
        self.sentences = sentences if sentences is not None else _load_sentences()
        self.word_freq: Counter = Counter()
        self.bigram_freq: Counter = Counter()
        for sentence in self.sentences:
            words = sentence.lower().split()
            self.word_freq.update(words)
            self.bigram_freq.update(zip(words, words[1:]))
        self.vocab: set[str] = set(self.word_freq)
        self._total_words = sum(self.word_freq.values())

    def word_log_freq(self, word: str) -> float:
        import math

        count = self.word_freq.get(word.lower(), 0)
        # add-one smoothing so unknown words get a small but nonzero score
        return math.log((count + 1) / (self._total_words + len(self.vocab) + 1))

    def bigram_bonus(self, word_a: str, word_b: str) -> float:
        return 0.5 if (word_a.lower(), word_b.lower()) in self.bigram_freq else 0.0


DEFAULT_CORPUS = Corpus()
