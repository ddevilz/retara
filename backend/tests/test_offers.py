from magenta.config import configs_dir
from magenta.offers import Arm, Offer, OfferCatalog, OfferDecision
from magenta.sim.population import generate_population


def _cat() -> OfferCatalog:
    return OfferCatalog.load(configs_dir() / "offers.yaml")


def test_all_eight_arms_present():
    cat = _cat()
    for arm in Arm:
        assert isinstance(cat.get(arm), Offer)
    assert len(list(Arm)) == 8


def test_no_action_is_free():
    off = _cat().get(Arm.NO_ACTION)
    assert off.cost == 0.0
    assert off.fits_causes == []


def test_eligible_always_includes_no_action():
    cat = _cat()
    pop, _ = generate_population(50, seed=1)
    for c in pop[:10]:
        assert Arm.NO_ACTION in cat.eligible(c)


def test_device_upgrade_only_eligible_near_contract_end():
    cat = _cat()
    pop, _ = generate_population(3000, seed=2)
    for c in pop:
        if Arm.DEVICE_UPGRADE in cat.eligible(c):
            assert c.contract_end_days <= 90


def test_offer_decision_defaults():
    d = OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0)
    assert d.rationale == ""
    assert d.propensity == 1.0
