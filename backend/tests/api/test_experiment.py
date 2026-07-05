import json

import pytest

from magenta.api import routes_stream as rs


class _FakeScorecard:
    def model_dump(self):
        return {
            "churn_treatment": 0.2, "churn_holdout": 0.25, "ate": -0.05,
            "ci_low": -0.08, "ci_high": -0.02, "wasted_offer_rate": 0.1,
            "sleeping_dogs_contacted": 0, "euros_retained": 9000.0,
            "offer_spend": 1500.0, "acceptance_rate": 0.4,
            "n_treatment": 100, "n_holdout": 100, "offers_made": 60,
        }


def _parse_sse(text: str):
    events, cur = [], {}
    for line in text.splitlines():
        if line.startswith("event:"):
            cur["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            cur["data"] = line.split(":", 1)[1].strip()
        elif line == "" and cur:
            events.append(cur); cur = {}
    if cur:
        events.append(cur)
    return events


@pytest.mark.asyncio
async def test_experiment_streams_scorecard(client, monkeypatch):
    monkeypatch.setattr(rs, "run_experiment",
                        lambda policy, n, seed: _FakeScorecard())
    resp = await client.post("/api/experiment",
                             json={"policy": "rules", "n": 50, "seed": 7})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert any(e["event"] == "progress" for e in events)
    sc = [e for e in events if e["event"] == "scorecard"]
    assert len(sc) == 1
    data = json.loads(sc[0]["data"])
    assert data["ate"] == -0.05 and data["n_treatment"] == 100
    assert events[-1]["event"] == "done"


@pytest.mark.asyncio
async def test_experiment_noaction_never_touches_deps(client, monkeypatch):
    """noaction/rules must not build a real GraphDeps (no model/db I/O)."""
    monkeypatch.setattr(rs, "run_experiment",
                        lambda policy, n, seed: _FakeScorecard())

    def _boom():
        raise AssertionError("get_graph_deps() should not be called for 'noaction'")

    monkeypatch.setattr(rs, "get_graph_deps", _boom)
    resp = await client.post("/api/experiment",
                             json={"policy": "noaction", "n": 10, "seed": 1})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[-1]["event"] == "done"
