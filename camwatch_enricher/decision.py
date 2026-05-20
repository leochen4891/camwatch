"""Decision rule: KNN neighbors -> (status, make?, model?).

Two confidence tiers per view:

- HIGH: tighter rule (e.g. top-5 unanimous + cosine >= 0.85). On its own a
  single high view is enough to label a pass.
- MEDIUM: looser rule (e.g. top-3 unanimous + cosine >= 0.80). A medium
  view doesn't label on its own — it only contributes when *every*
  available view of the same pass also lands at medium-or-better AND they
  all agree on (make, model). See _combine_views in server.py.

Anything that doesn't reach medium is LOW, and the pass falls through to
the existing Opus workflow.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .index import Neighbor


@dataclass
class Decision:
    status: str            # 'high' | 'medium' | 'low' | 'no_match'
    make: str | None
    model: str | None
    color: str | None
    top_sim: float
    agree_count: int       # how many of the head neighbors voted for the chosen label
    topk: list[Neighbor]


def _pick_color(head: list[Neighbor], make: str | None, model: str | None) -> str | None:
    return next(
        (n.color for n in head if n.make == make and n.model == model),
        None,
    )


def _try_tier(
    neighbors: list[Neighbor], k_agree: int, tau: float
) -> tuple[str | None, str | None, int, str | None]:
    """Return (make, model, agree_count, color) if the top k_agree neighbors
    are unanimous AND top_sim >= tau. Otherwise (None, None, 0, None)."""
    head = neighbors[:k_agree]
    if not head:
        return (None, None, 0, None)
    votes: Counter[tuple[str | None, str | None]] = Counter(
        (n.make, n.model) for n in head
    )
    (mk, md), agree = votes.most_common(1)[0]
    if (
        mk is not None
        and md is not None
        and agree == len(head)
        and neighbors[0].sim >= tau
    ):
        return (mk, md, agree, _pick_color(head, mk, md))
    return (None, None, 0, None)


def decide(
    neighbors: list[Neighbor],
    k_agree_high: int, tau_high: float,
    k_agree_medium: int | None = None, tau_medium: float | None = None,
) -> Decision:
    """Return the strongest tier this neighbor list satisfies.

    k_agree_medium / tau_medium are optional. When both are None, only the
    high tier is checked (back-compat with the original single-tier rule).
    """
    if not neighbors:
        return Decision("no_match", None, None, None, 0.0, 0, [])

    top_sim = neighbors[0].sim
    mk, md, agree, color = _try_tier(neighbors, k_agree_high, tau_high)
    if mk:
        return Decision("high", mk, md, color, top_sim, agree, neighbors)

    if k_agree_medium is not None and tau_medium is not None:
        mk, md, agree, color = _try_tier(neighbors, k_agree_medium, tau_medium)
        if mk:
            return Decision("medium", mk, md, color, top_sim, agree, neighbors)

    return Decision("low", None, None, None, top_sim, 0, neighbors)
