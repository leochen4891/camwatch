"""Decision rule: KNN neighbors -> (status, make?, model?).

Two confidence tiers per view, both anchored on the top-1 neighbor's label:

- HIGH: top-1's label is shared by at least `min_votes_high` of the top-K
  retrieved neighbors AND top-1 cosine >= `tau_high`. A single high view
  is enough to label a pass.
- MEDIUM: same form, looser bounds (`min_votes_medium`, `tau_medium`). A
  medium view doesn't label on its own — it only contributes when *every*
  available view of the same pass also lands at >= medium AND they all
  agree on (make, model). See _combine_views in server.py.

Anything that doesn't reach medium is LOW and falls through to the Opus
workflow.

Anchoring on the top-1 label (rather than picking the plurality across
the whole window) avoids the "top-1 is Tesla, plurality says Honda"
contradiction — the closest neighbor's label and the surrounding agreement
have to point the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

from .index import Neighbor


@dataclass
class Decision:
    status: str            # 'high' | 'medium' | 'low' | 'no_match'
    make: str | None
    model: str | None
    color: str | None
    top_sim: float
    agree_count: int       # how many of the top-K agreed with top-1's label
    topk: list[Neighbor]


def _pick_color(head: list[Neighbor], make: str | None, model: str | None) -> str | None:
    return next(
        (n.color for n in head if n.make == make and n.model == model),
        None,
    )


def decide(
    neighbors: list[Neighbor],
    k: int,
    min_votes_high: int, tau_high: float,
    min_votes_medium: int | None = None, tau_medium: float | None = None,
) -> Decision:
    """Return the strongest tier this neighbor list satisfies.

    `k` is the window size — only the top `k` neighbors are considered for
    vote counting (the caller still passes the full ranked list).

    `min_votes_medium` and `tau_medium` are optional; when both are None
    only the high tier is checked.
    """
    if not neighbors:
        return Decision("no_match", None, None, None, 0.0, 0, [])

    head = neighbors[:k]
    top = head[0]
    top_sim = top.sim

    if top.make is None or top.model is None:
        return Decision("low", None, None, None, top_sim, 0, neighbors)

    votes = sum(1 for n in head if n.make == top.make and n.model == top.model)
    color = _pick_color(head, top.make, top.model)

    if votes >= min_votes_high and top_sim >= tau_high:
        return Decision("high", top.make, top.model, color, top_sim, votes, neighbors)

    if (
        min_votes_medium is not None
        and tau_medium is not None
        and votes >= min_votes_medium
        and top_sim >= tau_medium
    ):
        return Decision("medium", top.make, top.model, color, top_sim, votes, neighbors)

    return Decision("low", None, None, None, top_sim, votes, neighbors)
