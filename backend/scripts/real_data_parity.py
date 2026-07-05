#!/usr/bin/env python
"""Real-data parity check: IBM Telco Customer Churn CSV vs Magenta simulator.

Thin, directly-runnable entry point. All the substance (download/verify,
feature building, model fit/eval, report formatting) lives in
``magenta.brain.parity`` so it's importable from both here and the
``magenta parity`` CLI subcommand, and testable offline in
``tests/test_parity.py`` without duplicating logic.

Usage:
    cd backend && uv run python scripts/real_data_parity.py
    # or, equivalently:
    cd backend && uv run magenta parity
"""
from __future__ import annotations

from magenta.brain.parity import main

if __name__ == "__main__":
    main()
