from ssr.correction.reranker import LanguageCorrector
from ssr.decoding.beam_search import Candidate


def test_correct_prefers_more_plausible_candidate():
    candidates = [
        Candidate(text="cam you pig up sum silk", score=0.0),
        Candidate(text="can you pick up some milk", score=0.0),
        Candidate(text="can you big up some milk", score=0.0),
    ]
    corrector = LanguageCorrector()
    result = corrector.correct(candidates)
    assert result == "can you pick up some milk"


def test_correct_lightly_fixes_unknown_words_via_edit_distance():
    candidates = [Candidate(text="cn you pik up som milk", score=10.0)]
    corrector = LanguageCorrector()
    result = corrector.correct(candidates)
    assert result == "can you pick up some milk"


def test_correct_empty_candidates_returns_empty_string():
    corrector = LanguageCorrector()
    assert corrector.correct([]) == ""
