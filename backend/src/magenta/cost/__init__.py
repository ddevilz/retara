"""Cost/rate orchestration (Lab 13): semantic cache, confidence cascade, cost
meter -- the levers that keep an LLM-in-the-loop cohort run inside a free-tier
rate budget (Groq: 30 RPM on the large role) without silently degrading
diagnosis quality (see `magenta cost report`, §0.5).
"""
