"""Tool schema loading and slot expansion.

Mem2ActBench schemas use non-standard type names (int/float/dict); we
normalize them to JSON-Schema-ish names and expose a typed view of each
parameter slot for the demand generator (C1) and the binder (C2).

Observed corpus stats (400 tools): string 897, int 162, boolean 146, float 75,
array 21, dict 6; 110 params carry an explicit default; 24 have enums;
27 are nested structures (binder treats nested values as opaque/verbatim).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# dataset type aliases -> canonical names
_TYPE_ALIASES = {
    "int": "integer", "integer": "integer", "float": "number", "number": "number",
    "str": "string", "string": "string", "bool": "boolean", "boolean": "boolean",
    "dict": "object", "object": "object", "list": "array", "array": "array",
}


@dataclass
class ParamSpec:
    name: str
    type: str                       # canonical: string/integer/number/boolean/object/array
    description: str = ""
    required: bool = False
    has_default: bool = False
    default: object = None
    enum: list | None = None
    nested: bool = False            # object/array with inner structure

    @property
    def deterministic_default(self) -> bool:
        """True when the schema itself supplies the value (C2 schema-default state)."""
        return self.has_default


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    params: list[ParamSpec] = field(default_factory=list)

    def required_params(self) -> list[ParamSpec]:
        return [p for p in self.params if p.required]

    def optional_params(self) -> list[ParamSpec]:
        return [p for p in self.params if not p.required]

    def params_with_default(self) -> list[ParamSpec]:
        return [p for p in self.params if p.has_default]

    def param(self, name: str) -> ParamSpec | None:
        for p in self.params:
            if p.name == name:
                return p
        return None


def _is_nested(spec: dict) -> bool:
    if isinstance(spec.get("properties"), dict):
        return True
    items = spec.get("items")
    return isinstance(items, dict) and ("properties" in items or "type" in items)


def load_tool_spec(schema: dict) -> ToolSpec:
    """Parse a Mem2ActBench `target_tool_schema` dict into a ToolSpec."""
    params_raw = (schema.get("parameters") or {}).get("properties") or {}
    required = set((schema.get("parameters") or {}).get("required") or [])
    if not required and isinstance(schema.get("required"), list):
        required = set(schema["required"])
    params = []
    for name, spec in params_raw.items():
        raw_type = str(spec.get("type", "string")).lower().strip()
        params.append(ParamSpec(
            name=name,
            type=_TYPE_ALIASES.get(raw_type, "string"),
            description=str(spec.get("description", "")),
            required=name in required,
            has_default="default" in spec,
            default=spec.get("default"),
            enum=spec.get("enum"),
            nested=_is_nested(spec),
        ))
    return ToolSpec(
        name=schema.get("name", ""),
        description=schema.get("description", ""),
        params=params,
    )
