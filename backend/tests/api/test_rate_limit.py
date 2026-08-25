import pytest


def test_rate_limit_key_is_the_tenant_not_the_ip():
    """A whole team shares one office IP; IP keying would throttle all of them."""
    from unittest.mock import MagicMock

    from magenta.api.rate_limit import tenant_rate_key
    from magenta.context import set_tenant

    set_tenant("org_abc")
    assert tenant_rate_key(MagicMock()) == "org_abc"


@pytest.mark.asyncio
async def test_experiment_is_rate_limited(client):
    """Six rapid cohort runs: the limit is five per minute."""
    codes = []
    for _ in range(6):
        resp = await client.post(
            "/api/experiment", json={"policy": "noaction", "n": 2, "seed": 7}
        )
        codes.append(resp.status_code)
    assert 429 in codes
