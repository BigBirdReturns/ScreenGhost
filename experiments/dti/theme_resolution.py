"""Theme resolution and bounded outfit planning for the DTI cartridge.

Community theme guides and item databases are treated as replaceable data
suppliers.  This module owns the stable semantic contract: normalize the visible
theme, resolve it to a compact style card, and choose a feasible outfit from the
locally taught wardrobe atlas under access and route constraints.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from experiments.dti.schema import (
    AccessTier,
    OutfitPlan,
    ThemeCard,
    WardrobeItem,
    clean_text,
)


_WORD = re.compile(r"[a-z0-9]+")


def normalize_theme(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    tokens = _WORD.findall(text)
    while tokens and tokens[0] in {"theme", "the", "your"}:
        tokens.pop(0)
    return " ".join(tokens)


def _tokens(value: str) -> frozenset[str]:
    return frozenset(normalize_theme(value).split())


def _similarity(left: str, right: str) -> float:
    left_norm = normalize_theme(left)
    right_norm = normalize_theme(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens = _tokens(left_norm)
    right_tokens = _tokens(right_norm)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(a=left_norm, b=right_norm).ratio()
    containment = 1.0 if left_norm in right_norm or right_norm in left_norm else 0.0
    return min(1.0, 0.75 * sequence + 0.20 * jaccard + 0.05 * containment)


@dataclass(frozen=True)
class ThemeCandidate:
    card: ThemeCard
    score: float
    matched_name: str


@dataclass(frozen=True)
class ThemeResolution:
    raw_text: str
    normalized_text: str
    resolved: bool
    card: Optional[ThemeCard]
    confidence: float
    margin: float
    alternatives: Tuple[Tuple[str, float], ...]
    reason: str


class ThemeCatalog:
    def __init__(
        self,
        cards: Iterable[ThemeCard],
        *,
        minimum_confidence: float = 0.64,
        minimum_margin: float = 0.07,
    ) -> None:
        self.cards = tuple(sorted(cards, key=lambda card: card.canonical_name.casefold()))
        if not self.cards:
            raise ValueError("theme catalog cannot be empty")
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_margin = float(minimum_margin)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        provenance: str,
        minimum_confidence: float = 0.64,
        minimum_margin: float = 0.07,
    ) -> "ThemeCatalog":
        cards = []
        for record in records:
            cards.append(
                ThemeCard(
                    canonical_name=str(record["canonical_name"]),
                    aliases=tuple(record.get("aliases") or ()),
                    tags=tuple(record.get("tags") or ()),
                    palettes=tuple(record.get("palettes") or ()),
                    silhouettes=tuple(record.get("silhouettes") or ()),
                    motifs=tuple(record.get("motifs") or ()),
                    avoid=tuple(record.get("avoid") or ()),
                    provenance=str(record.get("provenance") or provenance),
                )
            )
        return cls(
            cards,
            minimum_confidence=minimum_confidence,
            minimum_margin=minimum_margin,
        )

    def resolve(self, raw_text: str) -> ThemeResolution:
        normalized = normalize_theme(raw_text)
        if not normalized:
            return ThemeResolution(
                raw_text=raw_text,
                normalized_text="",
                resolved=False,
                card=None,
                confidence=0.0,
                margin=0.0,
                alternatives=(),
                reason="theme text was empty after normalization",
            )

        candidates: list[ThemeCandidate] = []
        for card in self.cards:
            names = (card.canonical_name, *card.aliases)
            scored = [(_similarity(normalized, name), name) for name in names]
            score, matched_name = max(scored, key=lambda row: (row[0], row[1].casefold()))
            candidates.append(ThemeCandidate(card=card, score=score, matched_name=matched_name))
        candidates.sort(key=lambda row: (-row.score, row.card.canonical_name.casefold()))

        best = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = best.score - second_score
        exact = normalize_theme(best.matched_name) == normalized
        resolved = exact or (
            best.score >= self.minimum_confidence and margin >= self.minimum_margin
        )
        reason = (
            f"exact match to {best.matched_name!r}"
            if exact
            else (
                f"resolved with confidence={best.score:.3f}, margin={margin:.3f}"
                if resolved
                else f"ambiguous or weak theme match: confidence={best.score:.3f}, margin={margin:.3f}"
            )
        )
        return ThemeResolution(
            raw_text=raw_text,
            normalized_text=normalized,
            resolved=resolved,
            card=best.card if resolved else None,
            confidence=best.score,
            margin=margin,
            alternatives=tuple(
                (candidate.card.canonical_name, round(candidate.score, 6))
                for candidate in candidates[:5]
            ),
            reason=reason,
        )


