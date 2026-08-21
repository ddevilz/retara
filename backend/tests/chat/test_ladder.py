from magenta.chat.ladder import OfferLadder
from magenta.chat.state import DialogueState
from magenta.graph.state import Diagnosis
from magenta.offers import Arm, Offer, OfferCatalog, OfferDecision


def _catalog(offers):
    return OfferCatalog.model_construct(offers=offers)


def _diag(tags):
    return Diagnosis(root_cause_tags=tags, narrative="", eligible_offer_ids=[], confidence=0.8)


def _fixture_catalog():
    return _catalog([
        Offer(arm=Arm.ACKNOWLEDGE_AND_FIX, cost=0.0, min_margin=0.0, eligibility_note="", fits_causes=["bill_shock"]),
        Offer(arm=Arm.BILL_CREDIT, cost=20.0, min_margin=0.0, eligibility_note="", fits_causes=["bill_shock"]),
        Offer(arm=Arm.PLAN_DOWNSELL, cost=50.0, min_margin=0.0, eligibility_note="", fits_causes=["bill_shock"]),
        Offer(arm=Arm.DEVICE_UPGRADE, cost=120.0, min_margin=0.0, eligibility_note="", fits_causes=["device"]),
    ])


def test_open_returns_cheapest_fitting_arm():
    ladder = OfferLadder(_fixture_catalog(), _diag(["bill_shock"]), authority_cap_eur=80.0)
    d = ladder.open()
    assert isinstance(d, OfferDecision)
    assert d.arm is Arm.ACKNOWLEDGE_AND_FIX
    assert ladder.position == 1


def test_concede_walks_cost_order():
    ladder = OfferLadder(_fixture_catalog(), _diag(["bill_shock"]), authority_cap_eur=80.0)
    ladder.open()                       # ACKNOWLEDGE_AND_FIX (0)
    assert ladder.concede(DialogueState(customer_id="C")).arm is Arm.BILL_CREDIT   # 20
    assert ladder.position == 2
    assert ladder.concede(DialogueState(customer_id="C")).arm is Arm.PLAN_DOWNSELL  # 50
    assert ladder.position == 3


def test_concede_returns_none_when_next_exceeds_cap():
    # cap 40 → after BILL_CREDIT(20), next is PLAN_DOWNSELL(50) > 40 → None
    ladder = OfferLadder(_fixture_catalog(), _diag(["bill_shock"]), authority_cap_eur=40.0)
    ladder.open()
    assert ladder.concede(DialogueState(customer_id="C")).arm is Arm.BILL_CREDIT
    assert ladder.concede(DialogueState(customer_id="C")) is None


def test_concede_never_exceeds_cap_and_none_when_exhausted():
    ladder = OfferLadder(_fixture_catalog(), _diag(["bill_shock"]), authority_cap_eur=1000.0)
    ladder.open()
    seen = []
    while (d := ladder.concede(DialogueState(customer_id="C"))) is not None:
        seen.append(d.arm)
        assert d.cost <= 1000.0
    # device offer does not fit bill_shock, so ladder = ack/credit/downsell only
    assert seen == [Arm.BILL_CREDIT, Arm.PLAN_DOWNSELL]


def test_position_zero_before_open():
    ladder = OfferLadder(_fixture_catalog(), _diag(["bill_shock"]), authority_cap_eur=80.0)
    assert ladder.position == 0
