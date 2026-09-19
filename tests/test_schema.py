"""Unit tests for tool schema parsing (uses real corpus shapes)."""

from forge.schema import load_tool_spec

SCHEMA = {
    "description": "Retrieve trending tickers by impressions.",
    "name": "/get-social-trending/impressions",
    "parameters": {
        "properties": {
            "isCrypto": {"description": "Include crypto tickers.", "type": "boolean"},
            "limit": {"description": "Max tickers (1-50).", "type": "int", "default": 50},
            "social": {"description": "Platform, e.g. Twitter, StockTwits.", "type": "string",
                       "enum": ["Twitter", "StockTwits", "Reddit"]},
            "tags": {"description": "Filter tags.", "type": "dict",
                     "properties": {"verified": {"type": "boolean"}}},
        },
        "required": ["social"],
        "type": "dict",
    },
}


def test_type_alias_normalization():
    spec = load_tool_spec(SCHEMA)
    types = {p.name: p.type for p in spec.params}
    assert types == {
        "isCrypto": "boolean", "limit": "integer",
        "social": "string", "tags": "object",
    }


def test_required_and_default_split():
    spec = load_tool_spec(SCHEMA)
    assert [p.name for p in spec.required_params()] == ["social"]
    defaults = spec.params_with_default()
    assert [p.name for p in defaults] == ["limit"]
    assert defaults[0].deterministic_default is True
    assert defaults[0].default == 50


def test_enum_and_nested_flags():
    spec = load_tool_spec(SCHEMA)
    social = spec.param("social")
    assert social.enum == ["Twitter", "StockTwits", "Reddit"]
    assert spec.param("tags").nested is True
    assert spec.param("limit").nested is False


def test_param_lookup_miss():
    spec = load_tool_spec(SCHEMA)
    assert spec.param("nonexistent") is None


def test_empty_schema_tolerated():
    spec = load_tool_spec({})
    assert spec.name == "" and spec.params == []
