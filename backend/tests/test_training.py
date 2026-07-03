from magenta.brain.training import build_training_data, TrainingData


def test_shapes_and_types():
    td = build_training_data(n=300, seed=11)
    assert isinstance(td, TrainingData)
    k = len(td.customers)
    assert k == 300
    assert len(td.churned) == k
    assert len(td.treated) == k
    assert len(td.retained) == k
    assert len(td.offers) == k
    assert all(isinstance(b, bool) for b in td.churned)


def test_treatment_is_roughly_half():
    td = build_training_data(n=1000, seed=5)
    share = sum(td.treated) / len(td.treated)
    assert 0.4 <= share <= 0.6


def test_control_rows_have_no_offer():
    td = build_training_data(n=400, seed=2)
    for treated, offer in zip(td.treated, td.offers):
        if not treated:
            assert offer is None
        else:
            assert offer is not None


def test_deterministic_given_seed():
    a = build_training_data(n=200, seed=7)
    b = build_training_data(n=200, seed=7)
    assert a.churned == b.churned
    assert a.treated == b.treated
