import json

from magenta.logging_config import bind_tenant, configure_logging, get_logger


def test_logs_are_json(capsys):
    configure_logging()
    get_logger("test").info("hello", extra_field=42)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["extra_field"] == 42


def test_tenant_id_is_bound_to_subsequent_lines(capsys):
    configure_logging()
    bind_tenant("org_abc")
    get_logger("test").info("scoped")
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["tenant_id"] == "org_abc"
