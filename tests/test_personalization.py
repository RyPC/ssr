from ssr.personalization.profile import UserProfile


def test_apply_substitutes_known_phrase():
    profile = UserProfile(user_id="u1", custom_vocab={"grb bb": "grab boba"})
    assert profile.apply("can you grb bb please") == "can you grab boba please"


def test_apply_substitutes_single_word():
    profile = UserProfile(user_id="u1", custom_vocab={"milk": "oat milk"})
    assert profile.apply("can you pick up some milk") == "can you pick up some oat milk"


def test_apply_leaves_unknown_text_unchanged():
    profile = UserProfile(user_id="u1")
    text = "can you pick up some milk"
    assert profile.apply(text) == text
