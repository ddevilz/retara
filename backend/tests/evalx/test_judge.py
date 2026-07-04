from unittest.mock import patch

from magenta.evalx.judge import judge_pair, judge_sample, JudgeVerdict, JudgeReport


class _Raw(JudgeVerdict):
    pass


def test_judge_pair_agrees_across_swap_gives_winner():
    # order1 (A,B) says "A" wins; order2 (B,A) also says the SAME message wins
    # in order2 the agent message sits in slot B, so a consistent judge returns "B"
    seq = [JudgeVerdict(winner="A", reasons=["clearer"]),
           JudgeVerdict(winner="B", reasons=["clearer"])]
    with patch("magenta.evalx.judge.chat_structured", side_effect=seq) as m:
        v = judge_pair("agent msg", "baseline msg", "context")
    assert m.call_count == 2
    assert v.winner == "A"  # canonicalized: the agent message won both orders


def test_judge_pair_position_disagreement_is_tie():
    # order1 says slot-A wins, order2 also says slot-A wins → the position, not the
    # message, is winning → position bias → TIE
    seq = [JudgeVerdict(winner="A", reasons=["x"]),
           JudgeVerdict(winner="A", reasons=["y"])]
    with patch("magenta.evalx.judge.chat_structured", side_effect=seq):
        v = judge_pair("agent msg", "baseline msg", "context")
    assert v.winner == "TIE"


def test_judge_sample_win_rate_excludes_ties():
    transcripts = ["t1", "t2", "t3"]
    # make judge_pair deterministic: 2 wins for agent, 1 tie
    verdicts = [JudgeVerdict(winner="A", reasons=[]),
                JudgeVerdict(winner="A", reasons=[]),
                JudgeVerdict(winner="TIE", reasons=[])]
    with patch("magenta.evalx.judge.judge_pair", side_effect=verdicts):
        rep = judge_sample(transcripts, baseline_fn=lambda c: "baseline", k=3)
    assert isinstance(rep, JudgeReport)
    assert rep.ties == 1
    assert rep.win_rate == 1.0  # 2 A-wins / 2 non-tie
