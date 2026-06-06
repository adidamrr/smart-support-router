from src.templates import DEFAULT_REPLY, get_suggested_reply


def test_get_suggested_reply_known_intent_returns_string() -> None:
    reply = get_suggested_reply("atm_support")

    assert isinstance(reply, str)
    assert reply.strip()


def test_get_suggested_reply_unknown_intent_returns_default_string() -> None:
    reply = get_suggested_reply("unknown_intent")

    assert isinstance(reply, str)
    assert reply == DEFAULT_REPLY
