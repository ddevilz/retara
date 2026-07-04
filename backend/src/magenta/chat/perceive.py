from __future__ import annotations

from pydantic import BaseModel, Field

from magenta.llm import chat_structured
from magenta.chat.state import DialogueState

_SYSTEM = (
    "You parse one customer turn in a telecom retention chat. "
    "Return intents (short verbs/nouns like 'cancel', 'bill_question', "
    "'offer_response', 'chitchat'), sentiment in [-1, 1] (negative = "
    "angry/frustrated), any named entities (competitor, amount, plan), and "
    "understanding_confidence in [0, 1] for how sure you are you understood."
)


class Perception(BaseModel):
    intents: list[str] = Field(default_factory=list)
    sentiment: float = 0.0
    entities: dict = Field(default_factory=dict)
    understanding_confidence: float = 0.0


def perceive(text: str, state: DialogueState) -> Perception:
    history = "\n".join(f"{t.speaker}: {t.text}" for t in state.turns[-4:])
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Recent turns:\n{history}\n\nCustomer now says: {text}"},
    ]
    return chat_structured(role="cheap", messages=messages, model_cls=Perception)
