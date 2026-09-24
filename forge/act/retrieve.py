"""Slot-level retrieval over fact cards.

Corpus unit = fact card, rendered as "attribute: value -- source_text".
Ranking: RRF fusion of
  path 1: BM25 over card text (always on)
  path 2: dense cosine over card text vs demand question (always on)
  path 3: attribute match -- normalized exact/containment between the demand's
          attribute_guess and card attributes (toggle, default off)

Failure-safe by design: with the toggle off, or when the guess matches nothing,
paths 1-2 alone rank -- never worse than pure semantic retrieval.
"""

from __future__ import annotations

import re

from forge.state.store import normalize_attribute


def card_text(card: dict) -> str:
    return f"{card['attribute']}: {card['value']} -- {card['source_text']}"


class SlotRetriever:
    def __init__(self, cards: list[dict], embedder=None, k: int = 5,
                 rrf_k: int = 60, use_attribute_match: bool = False):
        from rank_bm25 import BM25Okapi

        self.cards = list(cards)
        self.k = k
        self.rrf_k = rrf_k
        self.use_attribute_match = use_attribute_match
        self._texts = [card_text(c) for c in self.cards]
        self._attrs = [normalize_attribute(c["attribute"]) for c in self.cards]
        self._bm25 = BM25Okapi([_tok(t) for t in self._texts]) if self._texts else None
        self._vecs = embedder.encode(self._texts) if (embedder and self._texts) else None
        self._embedder = embedder

    def search(self, demand: dict, demand_vec=None) -> list[tuple[dict, float]]:
        """Return top-k cards for one demand."""
        if not self.cards:
            return []
        q = demand["query"]
        rrf: dict[int, float] = {}

        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tok(q))
            for rank, idx in enumerate(sorted(range(len(self.cards)), key=lambda i: -scores[i])[:50]):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        if self._vecs is not None:
            qv = demand_vec if demand_vec is not None else self._embedder.encode([q])[0]
            sims = self._vecs @ qv
            for rank, idx in enumerate(sorted(range(len(self.cards)), key=lambda i: -sims[i])[:50]):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        if self.use_attribute_match and demand.get("attribute_guess"):
            guess = demand["attribute_guess"]
            matched = [i for i, a in enumerate(self._attrs)
                       if guess and (a == guess or guess in a or a in guess)]
            for rank, idx in enumerate(matched):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        top = sorted(rrf.items(), key=lambda x: -x[1])[: self.k]
        return [(self.cards[i], s) for i, s in top]


def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
