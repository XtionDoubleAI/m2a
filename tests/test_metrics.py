"""Unit tests for metric definitions (deterministic, no LLM)."""

from m2a.eval.metrics import score_sample, aggregate


def test_perfect_prediction():
    r = score_sample("qa_1", "tool_a", {"city": "Dallas", "days": 7},
                     "tool_a", {"city": "Dallas", "days": 7})
    assert r.ta and r.tool_correct
    assert r.f1 == 1.0 and r.bleu1 == 1.0 and r.slot_acc == 1.0


def test_wrong_tool_blocks_ta_even_with_right_args():
    r = score_sample("qa_1", "tool_a", {"city": "Dallas"},
                     "tool_b", {"city": "Dallas"})
    assert not r.ta
    assert not r.tool_correct
    assert r.f1 == 1.0  # args alone still match
    agg = aggregate([r])
    assert agg.tsa == 0.0 and agg.arg_n == 0 and agg.arg_f1 == 0.0


def test_partial_overlap_f1():
    r = score_sample("qa_1", "tool_a", {"city": "New York", "days": 7},
                     "tool_a", {"city": "new york", "days": "week"})
    assert not r.ta
    # "city new york days 7" vs "city new york days week": 4/5 vs 4/4 overlap
    assert 0.7 < r.f1 < 1.0
    assert r.slot_detail["city"] is False  # case-sensitive comparison
    assert r.slot_detail["days"] is False


def test_canonicalization_of_bool_and_number():
    r = score_sample("qa_1", "tool_a", {"isCrypto": True, "limit": 50},
                     "tool_a", {"isCrypto": "true", "limit": "50"})
    assert r.ta, "JSON bool/number must compare equal to their string forms"


def test_missing_and_extra_params_symmetric_f1():
    # token-level F1 is symmetric in precision/recall swap: both cases give 0.667
    missing = score_sample("qa_1", "t", {"a": "x", "b": "y"}, "t", {"a": "x"})
    extra = score_sample("qa_1", "t", {"a": "x"}, "t", {"a": "x", "b": "zzz"})
    assert abs(missing.f1 - extra.f1) < 1e-9
    assert missing.f1 < 1.0 and extra.f1 < 1.0
    assert not missing.ta and not extra.ta
    # strictness lives in TA / slot level: missing param is a false slot, extra is ignored
    assert missing.slot_detail == {"a": True, "b": False}
    assert extra.slot_detail == {"a": True}


def test_empty_prediction():
    r = score_sample("qa_1", "t", {"a": "x"}, "t", {})
    assert r.f1 == 0.0 and r.bleu1 == 0.0 and not r.ta
