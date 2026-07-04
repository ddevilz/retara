from unittest.mock import patch

from magenta.chat.perceive import perceive, Perception
from magenta.chat.state import DialogueState


def test_perceive_makes_one_cheap_structured_call():
    fake = Perception(
        intents=["cancel"], sentiment=-0.5, entities={"competitor": "Vodafone"},
        understanding_confidence=0.9,
    )
    with patch("magenta.chat.perceive.chat_structured", return_value=fake) as m:
        st = DialogueState(customer_id="C1")
        out = perceive("I want to cancel, Vodafone is cheaper", st)
    assert m.call_count == 1
    args, kwargs = m.call_args
    role = kwargs.get("role", args[0] if args else None)
    assert role == "cheap"
    assert kwargs.get("model_cls", None) is Perception or Perception in args
    assert out.intents == ["cancel"]
    assert out.sentiment == -0.5
    assert out.understanding_confidence == 0.9


def test_perceive_passes_user_text_into_messages():
    fake = Perception(intents=[], sentiment=0.0, entities={}, understanding_confidence=0.5)
    with patch("magenta.chat.perceive.chat_structured", return_value=fake) as m:
        perceive("hello there", DialogueState(customer_id="C1"))
    _, kwargs = m.call_args
    messages = kwargs.get("messages")
    joined = " ".join(msg["content"] for msg in messages)
    assert "hello there" in joined
