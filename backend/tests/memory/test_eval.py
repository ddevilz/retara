import re

import pytest

from magenta.memory.eval import run_memory_eval

pytestmark = pytest.mark.slow


def test_memory_eval_reports_accuracy():
    out = run_memory_eval(n=20, seed=1)
    assert "temporal_accuracy" in out


def test_memory_eval_meets_target_accuracy():
    """Lab 12 exit criterion: temporal_accuracy >= 0.9 on the scripted
    supersede/current set -- consolidation makes this deterministic, so a
    correct implementation should land at 1.0."""
    out = run_memory_eval(n=30, seed=3)
    m = re.search(r"temporal_accuracy=([0-9.]+)", out)
    assert m is not None
    assert float(m.group(1)) >= 0.9
