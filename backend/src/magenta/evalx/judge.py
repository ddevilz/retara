from __future__ import annotations

from pydantic import BaseModel, Field

from magenta.llm import chat_structured

_SYSTEM = (
    "You are an impartial judge of two telecom retention-agent messages. "
    "Pick the message that is more empathetic, grounded (no invented offers), "
    "on-brand and safe. Answer with winner 'A' or 'B' (or 'TIE') and short reasons."
)


class JudgeVerdict(BaseModel):
    winner: str  # "A" | "B" | "TIE"
    reasons: list[str] = Field(default_factory=list)


class JudgeReport(BaseModel):
    win_rate: float
    ties: int
    examples: list[dict] = Field(default_factory=list)


def _ask(first: str, second: str, context: str) -> JudgeVerdict:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Context: {context}\n\nMessage A:\n{first}\n\nMessage B:\n{second}"},
    ]
    return chat_structured(role="judge", messages=messages, model_cls=JudgeVerdict)


def judge_pair(msg_a: str, msg_b: str, context: str) -> JudgeVerdict:
    v1 = _ask(msg_a, msg_b, context)          # msg_a in slot A
    v2 = _ask(msg_b, msg_a, context)          # msg_a in slot B
    # canonicalize each verdict to which *message* (a or b) it favored
    fav1 = {"A": "a", "B": "b"}.get(v1.winner)
    fav2 = {"A": "b", "B": "a"}.get(v2.winner)  # slots swapped in v2
    reasons = v1.reasons + v2.reasons
    if fav1 is None or fav2 is None or fav1 != fav2:
        return JudgeVerdict(winner="TIE", reasons=reasons)
    return JudgeVerdict(winner="A" if fav1 == "a" else "B", reasons=reasons)


def judge_sample(transcripts, baseline_fn, k: int = 20) -> JudgeReport:
    sample = list(transcripts)[:k]
    a_wins = 0
    non_tie = 0
    ties = 0
    examples: list[dict] = []
    for ctx in sample:
        agent_msg = ctx if isinstance(ctx, str) else str(ctx)
        v = judge_pair(agent_msg, baseline_fn(ctx), str(ctx))
        if v.winner == "TIE":
            ties += 1
        else:
            non_tie += 1
            if v.winner == "A":
                a_wins += 1
        examples.append({"context": str(ctx), "winner": v.winner, "reasons": v.reasons})
    win_rate = (a_wins / non_tie) if non_tie else 0.0
    return JudgeReport(win_rate=win_rate, ties=ties, examples=examples)
