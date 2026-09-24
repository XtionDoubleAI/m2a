"""Unit tests for the deterministic parts of state/ and act/ (no LLM)."""

from forge.act.intent import _covered
from forge.act.retrieve import SlotRetriever, card_text
from forge.schema import ParamSpec, ToolSpec
from forge.state.store import MemoryStore, normalize_attribute

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


# ---------- deterministic binding (early C2) ----------

def test_deterministic_override_whitelist():
    from forge.act.binder import deterministic_override

    spec = ToolSpec(name="t", params=[
        ParamSpec(name="address", type="string", required=True),
        ParamSpec(name="days", type="integer"),
    ])
    hits_addr = {"param_name": "address"}, [
        (dict(CARD, attribute="Wallet", value="0x3f5CE5", source_text="wallet",
              session_id="session_00059", turn_index=3), 0.9)]
    hits_days = {"param_name": "days"}, [
        (dict(CARD, attribute="Horizon", value="upcoming week", source_text="week",
              session_id="session_00059", turn_index=4), 0.8)]
    # fabricated answer -> substituted with newest compatible candidate
    args = deterministic_override(spec, [hits_addr, hits_days],
                                  {"address": "madeup", "days": 16})
    assert args["address"] == "0x3f5CE5"
    # incompatible candidate never blocks the model ('upcoming week' vs integer)
    assert args["days"] == 16
    # model's answer already among candidates -> respected
    args2 = deterministic_override(spec, [hits_addr], {"address": "0x3f5ce5"})
    assert args2["address"] == "0x3f5ce5"


def test_deterministic_override_newest_wins_on_fabrication():
    from forge.act.binder import deterministic_override

    spec = ToolSpec(name="t", params=[ParamSpec(name="pm", type="string")])
    demand = {"param_name": "pm"}
    old = (dict(CARD, attribute="PM", value="pip", session_id="session_00010", turn_index=2), 0.9)
    new = (dict(CARD, attribute="PM", value="uv", session_id="session_00290", turn_index=1), 0.5)
    # model chose a real candidate -> kept even though it is the older one
    args = deterministic_override(spec, [(demand, [old, new])], {"pm": "pip"})
    assert args["pm"] == "pip"
    # model fabricated -> newest candidate substituted
    args2 = deterministic_override(spec, [(demand, [old, new])], {"pm": "npm"})
    assert args2["pm"] == "uv"


def test_demand_fallback_on_empty_llm_response():
    """Gap-A fix: empty LLM output must yield deterministic template demands
    covering every parameter, not an empty list."""
    from forge.act.intent import generate_demands
    from forge.schema import load_tool_spec

    class EmptyLLM:
        def chat(self, system, user):
            return "no json here"

    spec = load_tool_spec({
        "name": "T", "description": "d",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "the city"},
                "days": {"type": "int", "description": "forecast days"},
            },
            "required": ["city", "days"],
        }})
    out = generate_demands(EmptyLLM(), "book it", spec, fallback=True)
    assert [d["param_name"] for d in out] == ["city", "days"]
    assert "the city" in out[0]["query"] and "city" in out[0]["query"]

    # fallback disabled reproduces the old silent-empty behaviour
    out_old = generate_demands(EmptyLLM(), "book it", spec, fallback=False)
    assert out_old == []
