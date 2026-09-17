"""Unit tests for the deterministic parts of state/ and act/ (no LLM)."""

from m2a.act.intent import _covered
from m2a.act.retrieve import SlotRetriever, card_text
from m2a.schema import ParamSpec, ToolSpec
from m2a.state.store import MemoryStore, normalize_attribute

CARD = {
    "attribute": "Platform Preference (finance)",
    "value": "Twitter",
    "source_text": "I usually check finance chatter on Twitter",
    "session_id": "session_00003",
    "turn_index": 5,
}


# ---------- write path ----------

def test_normalize_attribute_surface_only():
    # surface variation collapses (case/parens/space)
    assert (normalize_attribute("Platform Preference (finance)")
            == normalize_attribute("platform preference finance"))
    # semantic variation must NOT collapse (different labels stay distinct)
    assert (normalize_attribute("Stock Price Inquiry (AAPL)")
            != normalize_attribute("Price Inquiry (AAPL)"))


def test_version_chain_latest_wins():
    store = MemoryStore()
    store.add_session("session_00001", [{
        "attribute": "Package Manager (Python)", "value": "pip",
        "source_text": "use pip", "session_id": "session_00001", "turn_index": 2}])
    store.add_session("session_00005", [{
        "attribute": "package manager python", "value": "uv",
        "source_text": "switched to uv", "session_id": "session_00005", "turn_index": 1}])
    latest = store.latest_by_attribute(store.cards)
    assert len(latest) == 1                      # same normalized attribute collapses
    assert latest["package manager python"]["value"] == "uv"   # newest session wins


def test_cards_scoped_to_evidence_sessions():
    store = MemoryStore()
    store.add_session("s1", [dict(CARD, session_id="s1")])
    store.add_session("s2", [dict(CARD, session_id="s2", value="Reddit")])
    visible = store.cards_for_sessions(["s2"])
    assert len(visible) == 1 and visible[0]["value"] == "Reddit"


# ---------- read path ----------

def test_demand_coverage_check():
    spec = ToolSpec(name="t", params=[
        ParamSpec(name="social", type="string", required=True),
        ParamSpec(name="limit", type="integer", required=False),
    ])
    assert _covered(spec, [{"param_name": "social", "query": "q"}])
    assert not _covered(spec, [{"param_name": "limit", "query": "q"}])


def test_slot_retriever_attribute_match_toggle():
    cards = [dict(CARD), dict(CARD, attribute="Cuisine (Italian)", value="pasta",
                             source_text="I love pasta")]
    demand_hit = {"param_name": "social", "query": "which platform",
                  "attribute_guess": normalize_attribute("Platform Preference (finance)")}
    demand_miss = {"param_name": "social", "query": "which platform",
                   "attribute_guess": "weather home city"}
    r_off = SlotRetriever(cards, k=1)
    r_on = SlotRetriever(cards, k=1, use_attribute_match=True)
    # without embedder only BM25 runs; attribute match must not crash and must
    # prefer the matching card when toggled on
    assert r_off.search(demand_hit)[0][0]["value"] in {"Twitter", "pasta"}
    assert r_on.search(demand_hit)[0][0]["value"] == "Twitter"
    assert r_on.search(demand_miss)[0][0]["value"] in {"Twitter", "pasta"}  # degrades safely


def test_card_text_includes_verbatim_source():
    t = card_text(CARD)
    assert CARD["source_text"] in t and CARD["value"] in t
