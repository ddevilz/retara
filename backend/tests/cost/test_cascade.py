from magenta.cost.cascade import cascade


def test_high_confidence_stays_cheap():
    calls = []

    def chat_fn(role, msgs):
        calls.append(role)
        return f"ans-{role}"

    r = cascade([{"role": "user", "content": "x"}], chat_fn, confidence_fn=lambda a: 0.9, tau=0.6)
    assert r.role_used == "cheap" and r.escalated is False and calls == ["cheap"]


def test_low_confidence_escalates():
    calls = []

    def chat_fn(role, msgs):
        calls.append(role)
        return f"ans-{role}"

    r = cascade([{"role": "user", "content": "x"}], chat_fn, confidence_fn=lambda a: 0.2, tau=0.6)
    assert r.role_used == "large" and r.escalated is True and calls == ["cheap", "large"]
