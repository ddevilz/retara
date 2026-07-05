import json

import pytest

from magenta.api import data_access as da

_SCHEMA_KEYS = {
    "churn_treatment", "churn_holdout", "ate", "ci_low", "ci_high",
    "wasted_offer_rate", "sleeping_dogs_contacted", "euros_retained",
    "offer_spend", "acceptance_rate", "n_treatment", "n_holdout", "offers_made",
}


@pytest.fixture
def fake_scorecards(tmp_path, monkeypatch):
    payload = {
        "rungs": [
            {
                "policy": p,
                "scorecard": {
                    "churn_treatment": 0.20, "churn_holdout": 0.25, "ate": -0.05,
                    "ci_low": -0.08, "ci_high": -0.02, "wasted_offer_rate": 0.10,
                    "sleeping_dogs_contacted": 0, "euros_retained": 12345.0,
                    "offer_spend": 2000.0, "acceptance_rate": 0.42,
                    "n_treatment": 500, "n_holdout": 500, "offers_made": 300,
                },
            }
            for p in ("noaction", "rules", "risk_rules", "agent_s1", "agent")
        ]
    }
    path = tmp_path / "scorecards.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(da, "SCORECARDS_PATH", path)
    return path


@pytest.mark.asyncio
async def test_scorecards_schema(client, fake_scorecards):
    resp = await client.get("/api/scorecards")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["rungs"]) == 5
    assert {r["policy"] for r in body["rungs"]} == {
        "noaction", "rules", "risk_rules", "agent_s1", "agent"
    }
    for rung in body["rungs"]:
        assert set(rung["scorecard"].keys()) == _SCHEMA_KEYS
