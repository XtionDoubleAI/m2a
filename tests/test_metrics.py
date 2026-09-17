"""Unit tests for metric definitions (deterministic, no LLM).

Calibrated definitions (see metrics.py docstring): token F1/BLEU-1 over
parameter VALUES only, case-folded; TSA is the paper-comparable "TA"; EM is
our strict tool+params+values exact match.
"""

from m2a.eval.metrics import score_sample, aggregate


def test_perfect_prediction():
    r = score_sample("qa_1", "tool_a", {"city": "Dallas", "days": 7},
                     "tool_a", {"city": "Dallas", "days": 7})
    assert r.tsa and r.em
    assert r.f1 == 1.0 and r.bleu1 == 1.0 and r.slot_acc == 1.0


def test_wrong_tool_blocks_em_but_not_f1():
    r = score_sample("qa_1", "tool_a", {"city": "Dallas"},
                     "tool_b", {"city": "Dallas"})
    assert not r.tsa and not r.em
    assert r.f1 == 1.0  # value stream still matches
    agg = aggregate([r])
    assert agg.tsa == 0.0 and agg.arg_n == 0 and agg.arg_f1 == 0.0


def test_partial_overlap_f1_values_only():
    r = score_sample("qa_1", "tool_a", {"city": "New York", "days": 7},
                     "tool_a", {"city": "NEW YORK", "days": "week"})
    assert not r.em
    # values "new york 7" vs "new york week": case-folded match on 2/2, miss 1
    assert 0.5 < r.f1 < 1.0
    assert r.slot_detail["city"] is True   # casefold comparison
    assert r.slot_detail["days"] is False


def test_canonicalization_of_bool_and_number():
    r = score_sample("qa_1", "tool_a", {"isCrypto": True, "limit": 50},
                     "tool_a", {"isCrypto": "true", "limit": "50"})
    assert r.em, "JSON bool/number must compare equal to their string forms"


def test_missing_and_extra_params_symmetric_f1():
    # token-level F1 is symmetric under precision/recall swap
    missing = score_sample("qa_1", "t", {"a": "x", "b": "y"}, "t", {"a": "x"})
    extra = score_sample("qa_1", "t", {"a": "x"}, "t", {"a": "x", "b": "zzz"})
    assert abs(missing.f1 - extra.f1) < 1e-9
    assert missing.f1 < 1.0 and extra.f1 < 1.0
    assert not missing.em and not extra.em
    # slot level: missing param is a false slot; extra param is ignored
    assert missing.slot_detail == {"a": True, "b": False}
    assert extra.slot_detail == {"a": True}


def test_extra_params_do_not_break_tsa():
    r = score_sample("qa_1", "t", {"a": "x"}, "t", {"a": "x", "b": "junk"})
    assert r.tsa, "tool selection is unaffected by argument quality"
    assert not r.em, "strict EM requires the exact parameter set"


def test_empty_prediction():
    r = score_sample("qa_1", "t", {"a": "x"}, "t", {})
    assert r.f1 == 0.0 and r.bleu1 == 0.0 and not r.em
