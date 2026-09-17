"""Evidence rendering: pair each parameter with its retrieved cards.

The renderer is where "Retrieved-but-Unused" failures are attacked (teaching 01,
failure 5): evidence enters the prompt per parameter slot, not as an undirected
pile. Original source_text is always included verbatim (never compressed).
"""

from __future__ import annotations

from m2a.act.retrieve import card_text


def render_evidence(demand_evidence: list[tuple[dict, list[tuple[dict, float]]]]) -> str:
    """demand_evidence: [(demand, [(card, score), ...]), ...] -> prompt text."""
    parts = []
    for demand, hits in demand_evidence:
        block = [f"### Parameter `{demand['param_name']}`"]
        if not hits:
            block.append("(no evidence found in memory)")
        for j, (card, _score) in enumerate(hits, 1):
            block.append(
                f"{j}. {card_text(card)}  [recorded in {card.get('session_id', '?')} "
                f"turn {card.get('turn_index', '?')}]"
            )
        parts.append("\n".join(block))
    return "\n\n".join(parts)
