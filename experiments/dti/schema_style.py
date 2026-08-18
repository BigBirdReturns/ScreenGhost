"""Theme, wardrobe, and outfit-plan contracts."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from experiments.dti.schema_base import AccessTier, clean_text, content_id

@dataclass(frozen=True)
class ThemeCard:
    canonical_name: str
    aliases: Tuple[str, ...]
    tags: Tuple[str, ...]
    palettes: Tuple[str, ...] = ()
    silhouettes: Tuple[str, ...] = ()
    motifs: Tuple[str, ...] = ()
    avoid: Tuple[str, ...] = ()
    provenance: str = "screen_ghost_seed"

    def __post_init__(self) -> None:
        if not clean_text(self.canonical_name):
            raise ValueError("canonical_name is required")
        object.__setattr__(self, "aliases", tuple(sorted({str(v) for v in self.aliases})))
        object.__setattr__(self, "tags", tuple(sorted({str(v).casefold() for v in self.tags})))
        object.__setattr__(self, "palettes", tuple(sorted({str(v).casefold() for v in self.palettes})))
        object.__setattr__(self, "silhouettes", tuple(sorted({str(v).casefold() for v in self.silhouettes})))
        object.__setattr__(self, "motifs", tuple(sorted({str(v).casefold() for v in self.motifs})))
        object.__setattr__(self, "avoid", tuple(sorted({str(v).casefold() for v in self.avoid})))

    @property
    def card_id(self) -> str:
        return content_id("dtitheme1", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "tags": list(self.tags),
            "palettes": list(self.palettes),
            "silhouettes": list(self.silhouettes),
            "motifs": list(self.motifs),
            "avoid": list(self.avoid),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class WardrobeItem:
    item_id: str
    slot: str
    tags: Tuple[str, ...]
    palettes: Tuple[str, ...]
    anchor: str
    access: AccessTier = AccessTier.STANDARD
    route_cost: float = 1.0
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("item_id", "slot", "anchor"):
            if not clean_text(getattr(self, field_name)):
                raise ValueError(f"{field_name} is required")
        if not math.isfinite(float(self.route_cost)) or float(self.route_cost) < 0:
            raise ValueError("route_cost must be finite and nonnegative")
        object.__setattr__(self, "tags", tuple(sorted({str(v).casefold() for v in self.tags})))
        object.__setattr__(self, "palettes", tuple(sorted({str(v).casefold() for v in self.palettes})))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "slot": self.slot,
            "tags": list(self.tags),
            "palettes": list(self.palettes),
            "anchor": self.anchor,
            "access": self.access.value,
            "route_cost": self.route_cost,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OutfitPlan:
    plan_id: str
    theme_card_id: str
    theme_name: str
    item_ids: Tuple[str, ...]
    score: float
    route_cost: float
    covered_slots: Tuple[str, ...]
    rationale: Tuple[str, ...]
    unresolved: Tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        theme: ThemeCard,
        items: Sequence[WardrobeItem],
        score: float,
        route_cost: float,
        rationale: Sequence[str],
        unresolved: Sequence[str] = (),
    ) -> "OutfitPlan":
        item_ids = tuple(item.item_id for item in items)
        slots = tuple(sorted({item.slot for item in items}))
        payload = {
            "theme_card_id": theme.card_id,
            "item_ids": list(item_ids),
            "score": round(float(score), 6),
            "route_cost": round(float(route_cost), 6),
            "covered_slots": list(slots),
            "rationale": list(rationale),
            "unresolved": list(unresolved),
        }
        return cls(
            plan_id=content_id("dtiplan1", payload),
            theme_card_id=theme.card_id,
            theme_name=theme.canonical_name,
            item_ids=item_ids,
            score=float(score),
            route_cost=float(route_cost),
            covered_slots=slots,
            rationale=tuple(rationale),
            unresolved=tuple(unresolved),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

