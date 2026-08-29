import asyncio

import pytest

from magenta.context import get_tenant, require_tenant, set_tenant


def test_tenant_roundtrip():
    set_tenant("org_abc")
    assert get_tenant() == "org_abc"


def test_require_tenant_raises_when_unset():
    from magenta.context import current_tenant_id

    current_tenant_id.set(None)
    with pytest.raises(RuntimeError, match="no tenant"):
        require_tenant()


@pytest.mark.asyncio
async def test_context_does_not_leak_between_tasks():
    """Two concurrent requests must not see each other's tenant."""
    seen = {}

    async def worker(tenant: str):
        set_tenant(tenant)
        await asyncio.sleep(0.01)
        seen[tenant] = get_tenant()

    await asyncio.gather(worker("org_a"), worker("org_b"))
    assert seen == {"org_a": "org_a", "org_b": "org_b"}
