"""Evidence rendering: pair each parameter with its retrieved cards.

The renderer is where "Retrieved-but-Unused" failures are attacked (teaching 01,
failure 5): evidence enters the prompt per parameter slot, not as an undirected
pile. Original source_text is always included verbatim (never compressed).
"""

from __future__ import annotations

from m2a.act.retrieve import card_text


def render_evidence(demand_evidence: list[tuple[dict, list[tuple[dict, float]]]],
                    chunk_texts: list[str] | None = None) -> str:
    """demand_evidence: [(demand, [(card, score), ...]), ...] -> prompt text.

    Format note (learned the hard way): small models do not parse values out of
    prose-like card lines -- they ignore them and hallucinate. The card's clean
    `value` field is therefore surfaced FIRST as an explicit candidate the model
    can confirm or override, with the verbatim evidence sentence after it.

    chunk_texts: lossless dialogue excerpts (hybrid store). Appended verbatim
    after the per-parameter blocks so values that never became cards remain
    reachable by the model.
    """
    parts = []
    for demand, hits in demand_evidence:
        block = [f"### Parameter `{demand['param_name']}`"]
        if not hits:
            block.append("(no evidence found in memory)")
        for j, (card, _score) in enumerate(hits, 1):
            block.append(
                f"{j}. candidate value: \"{card['value']}\""
                f"\n   supporting dialogue: \"{card['source_text']}\""
                f"\n   (category: {card['attribute']}; recorded in {card.get('session_id', '?')} "
                f"turn {card.get('turn_index', '?')})"
            )
        parts.append("\n".join(block))
    out = "\n\n".join(parts)
    if chunk_texts:
        out += "\n\n### Original dialogue excerpts (verbatim)\n" + "\n---\n".join(chunk_texts)
    return out
