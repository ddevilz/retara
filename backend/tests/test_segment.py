from magenta.brain.uplift import classify_segment
from magenta.sim.population import Segment


def test_low_risk_is_sure_thing():
    assert classify_segment(p_churn=0.10, tau=0.30) is Segment.SURE_THING


def test_negative_tau_high_risk_is_lost_cause():
    # tau <= 0 with high risk -> lost cause (offer won't help).
    assert classify_segment(p_churn=0.80, tau=-0.001) is Segment.LOST_CAUSE


def test_clearly_negative_tau_is_sleeping_dog():
    # strongly negative uplift -> contacting backfires.
    assert classify_segment(p_churn=0.60, tau=-0.15) is Segment.SLEEPING_DOG


def test_high_risk_positive_tau_is_persuadable():
    assert classify_segment(p_churn=0.70, tau=0.20) is Segment.PERSUADABLE
