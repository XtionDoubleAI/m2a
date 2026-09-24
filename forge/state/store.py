"""Fact-card store with attribute normalization and version chains.

Shared vocabulary (the "type system" of the memory-action interface): both the
write path (card attributes) and the read path (demand attribute guesses) go
through `normalize_attribute`, exported here and imported by act/intent.py and
act/retrieve.py. This is deliberate: one normalization
function for both sides of the interface.

Version chain (C3-light, toggled): cards sharing a normalized attribute keep
all versions ordered by (session order, turn); binding-side resolution of
"latest" happens here via `latest_by_attribute`.
"""

from __future__ import annotations

import re


def normalize_attribute(attr: str) -> str:
    """Canonical form used by BOTH card attributes and demand guesses.

    Lowercase; punctuation (including parentheses) becomes spaces; whitespace
    collapsed. Handles only SURFACE variation -- 'Platform Preference (finance)'
    and 'platform preference finance' collide, while semantically different
    labels ('Stock Price Inquiry' vs 'Price Inquiry') stay distinct. Semantic
    normalization is the extractor's/guesser's job, not a string function's.
    """
    s = attr.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


class MemoryStore:
    def __init__(self, cards: list[dict] | None = None):
        self.cards: list[dict] = list(cards or [])
        self._session_order: dict[str, int] = {}
        for c in self.cards:
            self._register(c)

    def _register(self, card: dict) -> None:
        sid = card.get("session_id", "")
        if sid not in self._session_order:
            self._session_order[sid] = len(self._session_order)

    def add_session(self, session_id: str, cards: list[dict]) -> None:
        self._session_order.setdefault(session_id, len(self._session_order))
        for c in cards:
            c.setdefault("session_id", session_id)
            self.cards.append(c)

    def cards_for_sessions(self, session_ids: list[str]) -> list[dict]:
        """Cards visible to a task: only from its evidence sessions."""
        wanted = set(session_ids)
        return [c for c in self.cards if c.get("session_id") in wanted]

    def latest_by_attribute(self, cards: list[dict]) -> dict[str, dict]:
        """Version-chain resolution: for each normalized attribute, the most
        recent card by (session insertion order, turn_index)."""
        best: dict[str, tuple[tuple[int, int], dict]] = {}
        for c in cards:
            key = normalize_attribute(c["attribute"])
            rank = (self._session_order.get(c.get("session_id", ""), 0),
                    int(c.get("turn_index", 0)))
            if key not in best or rank > best[key][0]:
                best[key] = (rank, c)
        return {k: v[1] for k, v in best.items()}
