"""Round perception, phase custody, and bounded action settlement for DTI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from experiments.dti.policy import DTISafetyPolicy, PolicyDecision
from experiments.dti.schema import (
    ActionReceipt,
    DTIObservation,
    GameAction,
    GameActionKind,
    RoundPhase,
    RunContext,
    clean_text,
)


@dataclass(frozen=True)
class RoundSignals:
    texts: Tuple[str, ...] = ()
    labels: Tuple[str, ...] = ()
    theme_text: Optional[str] = None
    timer_seconds: Optional[int] = None
    freeplay_marker: bool = False
    runway_active: bool = False
    results_marker: bool = False
    confidence: float = 1.0
    frame_sha256: Optional[str] = None
    evidence_sources: Tuple[str, ...] = ("pixels",)
    detail: Mapping[str, object] = field(default_factory=dict)


class PhaseClassifier:
    """Small deterministic classifier over already-extracted visible signals."""

    def classify(self, signals: RoundSignals) -> DTIObservation:
        corpus = " ".join((*signals.texts, *signals.labels)).casefold()
        labels = tuple(signals.labels)
        phase = RoundPhase.UNKNOWN
        unknown = False

        if signals.results_marker or any(
            marker in corpus for marker in ("voting complete", "results", "podium", "winners")
        ):
            phase = RoundPhase.RESULTS
        elif signals.runway_active:
            phase = RoundPhase.RUNWAY
        elif any(marker in corpus for marker in ("to vote", "vote now", "give stars")):
            phase = RoundPhase.VOTING
        elif any(marker in corpus for marker in ("walk the runway", "choose runway")):
            phase = RoundPhase.RUNWAY_READY
        elif signals.freeplay_marker or (
            "freeplay" in corpus and signals.timer_seconds is None
        ):
            phase = RoundPhase.FREEPLAY
        elif signals.theme_text and signals.timer_seconds is not None:
            phase = RoundPhase.DRESSING
        elif signals.theme_text:
            phase = RoundPhase.THEME_BRIEF
        elif any(marker in corpus for marker in ("intermission", "next round", "waiting for players")):
            phase = RoundPhase.LOBBY
        else:
            phase = RoundPhase.UNKNOWN
            unknown = True

        return DTIObservation.build(
            phase=phase,
            confidence=signals.confidence,
            theme_text=signals.theme_text,
            timer_seconds=signals.timer_seconds,
            visible_labels=labels,
            frame_sha256=signals.frame_sha256,
            unknown=unknown,
            evidence_sources=signals.evidence_sources,
            detail=signals.detail,
        )


