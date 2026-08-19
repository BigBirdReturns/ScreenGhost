"""Taught wardrobe atlas and deterministic route-aware outfit planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

from experiments.dti.schema import AccessTier, OutfitPlan, ThemeCard, WardrobeItem

@dataclass(frozen=True)
class ScoredItem:
    item: WardrobeItem
    score: float
    reasons: Tuple[str, ...]


class WardrobeAtlas:
    def __init__(self, items: Iterable[WardrobeItem]) -> None:
        self.items = tuple(sorted(items, key=lambda item: item.item_id))
        if len({item.item_id for item in self.items}) != len(self.items):
            raise ValueError("wardrobe item ids must be unique")

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "WardrobeAtlas":
        items = []
        for record in records:
            items.append(
                WardrobeItem(
                    item_id=str(record["item_id"]),
                    slot=str(record["slot"]),
                    tags=tuple(record.get("tags") or ()),
                    palettes=tuple(record.get("palettes") or ()),
                    anchor=str(record["anchor"]),
                    access=AccessTier(str(record.get("access") or AccessTier.STANDARD.value)),
                    route_cost=float(record.get("route_cost", 1.0)),
                    enabled=bool(record.get("enabled", True)),
                    metadata=dict(record.get("metadata") or {}),
                )
            )
        return cls(items)

    def available(self, allowed_access: Sequence[AccessTier]) -> tuple[WardrobeItem, ...]:
        allowed = set(allowed_access)
        return tuple(
            item
            for item in self.items
            if item.enabled and item.access in allowed and item.access is not AccessTier.REMOVED
        )


class OutfitPlanner:
    """Deterministic theme-to-atlas resolver with route-cost pressure."""

    OPTIONAL_SLOTS = ("hair", "face", "shoes", "accessory")

    def __init__(self, *, route_penalty: float = 0.28) -> None:
        self.route_penalty = float(route_penalty)

    @staticmethod
    def _score_item(theme: ThemeCard, item: WardrobeItem) -> ScoredItem:
        theme_tags = set(theme.tags) | set(theme.silhouettes) | set(theme.motifs)
        item_tags = set(item.tags)
        matching_tags = sorted(theme_tags & item_tags)
        matching_palettes = sorted(set(theme.palettes) & set(item.palettes))
        avoid_matches = sorted(set(theme.avoid) & item_tags)

        score = 3.0 * len(matching_tags) + 1.4 * len(matching_palettes)
        score -= 4.0 * len(avoid_matches)
        if item.metadata.get("signature"):
            score += 0.35
        reasons = []
        if matching_tags:
            reasons.append("tags=" + ",".join(matching_tags))
        if matching_palettes:
            reasons.append("palette=" + ",".join(matching_palettes))
        if avoid_matches:
            reasons.append("avoid=" + ",".join(avoid_matches))
        if not reasons:
            reasons.append("generic slot coverage")
        return ScoredItem(item=item, score=score, reasons=tuple(reasons))

    def plan(
        self,
        theme: ThemeCard,
        atlas: WardrobeAtlas,
        *,
        allowed_access: Sequence[AccessTier] = (
            AccessTier.STANDARD,
            AccessTier.CODE,
            AccessTier.EVENT,
        ),
    ) -> OutfitPlan:
        available = atlas.available(allowed_access)
        by_slot: dict[str, list[ScoredItem]] = {}
        for item in available:
            scored = self._score_item(theme, item)
            by_slot.setdefault(item.slot.casefold(), []).append(scored)
        for values in by_slot.values():
            values.sort(
                key=lambda row: (
                    -(row.score - self.route_penalty * row.item.route_cost),
                    row.item.route_cost,
                    row.item.item_id,
                )
            )

        selected: list[ScoredItem] = []
        unresolved: list[str] = []

        dress = by_slot.get("dress", [None])[0] if by_slot.get("dress") else None
        top = by_slot.get("top", [None])[0] if by_slot.get("top") else None
        bottom = by_slot.get("bottom", [None])[0] if by_slot.get("bottom") else None
        dress_value = (
            dress.score - self.route_penalty * dress.item.route_cost if dress is not None else float("-inf")
        )
        separates = [value for value in (top, bottom) if value is not None]
        separates_value = sum(
            value.score - self.route_penalty * value.item.route_cost for value in separates
        )
        if dress is not None and (len(separates) < 2 or dress_value >= separates_value):
            selected.append(dress)
        elif len(separates) == 2:
            selected.extend(separates)
        elif dress is not None:
            selected.append(dress)
        else:
            unresolved.append("base")

        for slot in self.OPTIONAL_SLOTS:
            values = by_slot.get(slot) or []
            if values:
                selected.append(values[0])
            elif slot in {"hair", "shoes"}:
                unresolved.append(slot)

        # Route cost is charged once per taught anchor.  Freeplay quick teleport
        # can therefore make a whole station effectively one bounded transition.
        anchor_costs: dict[str, float] = {}
        for value in selected:
            anchor_costs[value.item.anchor] = min(
                anchor_costs.get(value.item.anchor, float("inf")), value.item.route_cost
            )
        route_cost = sum(anchor_costs.values())
        raw_score = sum(value.score for value in selected)
        final_score = raw_score - self.route_penalty * route_cost - 2.0 * len(unresolved)

        rationale = tuple(
            f"{value.item.slot}:{value.item.item_id} ({'; '.join(value.reasons)})"
            for value in selected
        )
        return OutfitPlan.build(
            theme=theme,
            items=[value.item for value in selected],
            score=final_score,
            route_cost=route_cost,
            rationale=rationale,
            unresolved=unresolved,
        )

