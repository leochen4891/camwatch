"""Decision rule: KNN neighbors -> (status, make?, model?).

Conservative by design — we'd rather leave a row NULL for Opus than guess
wrong. A label fires only when the top-K neighbors agree AND the closest
match is over the cosine threshold.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .index import Neighbor


@dataclass
class Decision:
    status: str            # 'high' | 'low' | 'no_match'
    make: str | None
    model: str | None
    color: str | None
    top_sim: float
    agree_count: int       # how many of the top k_agree neighbors voted for the chosen label
    topk: list[Neighbor]


def decide(neighbors: list[Neighbor], k_agree: int, tau_high: float) -> Decision:
    if not neighbors:
        return Decision("no_match", None, None, None, 0.0, 0, [])

    top_sim = neighbors[0].sim
    head = neighbors[:k_agree]
    votes: Counter[tuple[str | None, str | None]] = Counter(
        (n.make, n.model) for n in head
    )
    (best_make, best_model), agree = votes.most_common(1)[0]

    if (
        best_make is not None
        and best_model is not None
        and agree == len(head)            # unanimous within the head
        and top_sim >= tau_high
    ):
        # Pick the color from the closest agreeing neighbor — useful for
        # downstream filtering even though we don't claim color confidence.
        color = next(
            (n.color for n in head if n.make == best_make and n.model == best_model),
            None,
        )
        return Decision("high", best_make, best_model, color, top_sim, agree, neighbors)

    return Decision("low", None, None, None, top_sim, agree, neighbors)
