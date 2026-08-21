from magenta.chat.state import ChatStatus, DialogueState, Turn
from magenta.offers import Arm


def test_chatstatus_values():
    assert ChatStatus.ACTIVE == "ACTIVE"
    assert {s.value for s in ChatStatus} == {
        "ACTIVE", "ACCEPTED", "REJECTED", "ESCALATED", "HANDOFF"
    }


def test_dialogue_state_defaults_are_empty_and_active():
    st = DialogueState(customer_id="C1")
    assert st.status is ChatStatus.ACTIVE
    assert st.turns == []
    assert st.intent_stack == []
    assert st.sentiment == 0.0
    assert st.ladder_position == 0
    assert st.concessions_made == []
    assert st.understanding_confidence == 0.0


def test_dialogue_state_roundtrip_and_arm_typing():
    st = DialogueState(
        customer_id="C1",
        turns=[Turn(speaker="customer", text="hi"), Turn(speaker="agent", text="hello")],
        intent_stack=["cancel"],
        sentiment=-0.3,
        concessions_made=[Arm.BILL_CREDIT],
    )
    dumped = st.model_dump()
    assert dumped["concessions_made"] == ["BILL_CREDIT"]
    reloaded = DialogueState.model_validate(dumped)
    assert reloaded.concessions_made == [Arm.BILL_CREDIT]
    assert len(reloaded.turns) == 2
