"""Public surface of the retention decision graph package.

Downstream consumers outside `magenta.graph` (e.g. `magenta.chat.agent`) import
graph contracts from here rather than reaching into submodules, so the
package's external API stays stable as internals move. Nothing here changes
any Lab 6/7 behavior — this module only aggregates existing names.
"""
from __future__ import annotations

from magenta.graph.build import GraphDeps
from magenta.graph.nodes import act
from magenta.graph.state import (
    Diagnosis,
    GuardrailVerdict,
    OverallState,
    RiskUpliftReport,
    Timing,
)

__all__ = [
    "GraphDeps",
    "act",
    "Diagnosis",
    "GuardrailVerdict",
    "OverallState",
    "RiskUpliftReport",
    "Timing",
]
