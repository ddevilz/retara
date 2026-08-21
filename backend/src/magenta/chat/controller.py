from __future__ import annotations

from enum import StrEnum

from magenta.chat.perceive import Perception
from magenta.chat.state import DialogueState

_NEGOTIATE_INTENTS = {"cancel", "offer_response"}


class DialogueAct(StrEnum):
    CLARIFY = "CLARIFY"
    ANSWER = "ANSWER"
    EMPATHIZE = "EMPATHIZE"
    NEGOTIATE = "NEGOTIATE"
    CONFIRM_ACT = "CONFIRM_ACT"
    HANDOFF = "HANDOFF"


def next_act(state: DialogueState, perception: Perception) -> DialogueAct:
    if perception.understanding_confidence < 0.6:
        return DialogueAct.CLARIFY
    if perception.sentiment < -0.6:
        return DialogueAct.EMPATHIZE
    if len(state.turns) > 8:
        return DialogueAct.HANDOFF
    if _NEGOTIATE_INTENTS & set(perception.intents):
        return DialogueAct.NEGOTIATE
    return DialogueAct.ANSWER
