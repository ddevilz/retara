import pytest

from magenta.chat.controller import next_act, DialogueAct
from magenta.chat.perceive import Perception
from magenta.chat.state import DialogueState, Turn


def _state(n_turns=0):
    return DialogueState(
        customer_id="C1",
        turns=[Turn(speaker="customer", text="x") for _ in range(n_turns)],
    )


CASES = [
    # (confidence, sentiment, intents, n_turns, expected)
    (0.4, 0.0, ["cancel"], 1, DialogueAct.CLARIFY),      # low conf wins over all
    (0.9, -0.8, ["cancel"], 1, DialogueAct.EMPATHIZE),   # anger before negotiate
    (0.9, 0.0, ["bill_question"], 9, DialogueAct.HANDOFF),  # turn cap
    (0.9, 0.0, ["cancel"], 3, DialogueAct.NEGOTIATE),
    (0.9, 0.0, ["offer_response"], 3, DialogueAct.NEGOTIATE),
    (0.9, 0.0, ["chitchat"], 3, DialogueAct.ANSWER),
    (0.9, -0.61, ["cancel"], 1, DialogueAct.EMPATHIZE),  # boundary just under -0.6
    (0.59, 0.0, ["chitchat"], 1, DialogueAct.CLARIFY),   # boundary just under 0.6
]


@pytest.mark.parametrize("conf,sent,intents,n,expected", CASES)
def test_controller_rule_table(conf, sent, intents, n, expected):
    p = Perception(intents=intents, sentiment=sent, entities={}, understanding_confidence=conf)
    assert next_act(_state(n), p) is expected


def test_confirm_act_never_emitted_by_rules():
    p = Perception(intents=["cancel"], sentiment=0.0, entities={}, understanding_confidence=1.0)
    for n in range(0, 9):
        assert next_act(_state(n), p) is not DialogueAct.CONFIRM_ACT
