"""Run, action, and visible observation contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from experiments.dti.schema_base import (
    ControlMode, GameActionKind, PlayVenue, Point, RoundPhase, clean_text,
    content_id, normalized_point, sha256_json,
)

@dataclass(frozen=True)
class RunContext:
    venue: PlayVenue
    mode: ControlMode
    official_client: bool = True
    emulator: bool = False
    modified_client: bool = False
    process_injection: bool = False
    process_memory: bool = False
    multi_account: bool = False
    reward_farming: bool = False
    autonomous_voting: bool = False
    human_present: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["venue"] = self.venue.value
        value["mode"] = self.mode.value
        return value


@dataclass(frozen=True)
class GameAction:
    action_id: str
    kind: GameActionKind
    semantic_label: str
    keys: Tuple[str, ...] = ()
    duration_ms: int = 0
    point: Optional[Point] = None
    delta: Optional[Tuple[int, int]] = None
    wheel_delta: int = 0
    expected_phase: Optional[RoundPhase] = None
    expected_label: Optional[str] = None
    require_visible_change: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not clean_text(self.action_id):
            raise ValueError("action_id is required")
        if not clean_text(self.semantic_label):
            raise ValueError("semantic_label is required")
        object.__setattr__(self, "keys", tuple(str(v).lower() for v in self.keys))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.point is not None:
            object.__setattr__(self, "point", normalized_point(self.point))
        if self.kind is GameActionKind.PRESS:
            if not self.keys or len(self.keys) > 4:
                raise ValueError("press requires one to four keys")
            if self.duration_ms not in (0,):
                raise ValueError("press does not accept duration_ms")
        elif self.kind is GameActionKind.HOLD:
            if not self.keys or len(self.keys) > 4:
                raise ValueError("hold requires one to four keys")
            if not 1 <= int(self.duration_ms) <= 2500:
                raise ValueError("hold duration must be in [1,2500] ms")
        elif self.kind is GameActionKind.CLICK:
            if self.point is None:
                raise ValueError("click requires a normalized point")
        elif self.kind is GameActionKind.MOVE_MOUSE:
            if self.delta is None or len(self.delta) != 2:
                raise ValueError("move_mouse requires a two-value delta")
            dx, dy = (int(v) for v in self.delta)
            if abs(dx) > 1500 or abs(dy) > 1500:
                raise ValueError("mouse movement exceeds the bounded delta")
            object.__setattr__(self, "delta", (dx, dy))
        elif self.kind is GameActionKind.SCROLL:
            if self.wheel_delta == 0 or abs(int(self.wheel_delta)) > 1200:
                raise ValueError("scroll delta must be nonzero and bounded")
        elif self.kind is GameActionKind.WAIT:
            if not 1 <= int(self.duration_ms) <= 5000:
                raise ValueError("wait duration must be in [1,5000] ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "semantic_label": self.semantic_label,
            "keys": list(self.keys),
            "duration_ms": self.duration_ms,
            "point": list(self.point) if self.point is not None else None,
            "delta": list(self.delta) if self.delta is not None else None,
            "wheel_delta": self.wheel_delta,
            "expected_phase": self.expected_phase.value if self.expected_phase else None,
            "expected_label": self.expected_label,
            "require_visible_change": self.require_visible_change,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DTIObservation:
    observation_id: str
    phase: RoundPhase
    confidence: float
    theme_text: Optional[str]
    timer_seconds: Optional[int]
    visible_labels: Tuple[str, ...]
    semantic_signature: str
    frame_sha256: Optional[str] = None
    unknown: bool = False
    evidence_sources: Tuple[str, ...] = ("pixels",)
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("observation confidence must lie in [0,1]")
        if self.timer_seconds is not None and int(self.timer_seconds) < 0:
            raise ValueError("timer_seconds cannot be negative")
        if self.unknown and self.phase is not RoundPhase.UNKNOWN:
            raise ValueError("unknown observations must use phase=unknown")
        object.__setattr__(self, "visible_labels", tuple(sorted({str(v) for v in self.visible_labels})))
        object.__setattr__(self, "evidence_sources", tuple(str(v) for v in self.evidence_sources))
        object.__setattr__(self, "detail", dict(self.detail))

    @classmethod
    def build(
        cls,
        *,
        phase: RoundPhase,
        confidence: float,
        theme_text: Optional[str] = None,
        timer_seconds: Optional[int] = None,
        visible_labels: Sequence[str] = (),
        frame_sha256: Optional[str] = None,
        unknown: bool = False,
        evidence_sources: Sequence[str] = ("pixels",),
        detail: Optional[Mapping[str, Any]] = None,
    ) -> "DTIObservation":
        labels = tuple(sorted({clean_text(v) for v in visible_labels if clean_text(v)}))
        semantic_payload = {
            "phase": phase.value,
            "theme_text": clean_text(theme_text),
            "timer_bucket": None if timer_seconds is None else int(timer_seconds) // 5,
            "labels": list(labels),
            "unknown": bool(unknown),
        }
        signature = sha256_json(semantic_payload)
        payload = {
            **semantic_payload,
            "confidence": round(float(confidence), 6),
            "frame_sha256": frame_sha256,
            "evidence_sources": list(evidence_sources),
            "detail": dict(detail or {}),
        }
        return cls(
            observation_id=content_id("dtiobservation1", payload),
            phase=phase,
            confidence=float(confidence),
            theme_text=clean_text(theme_text),
            timer_seconds=(int(timer_seconds) if timer_seconds is not None else None),
            visible_labels=labels,
            semantic_signature=signature,
            frame_sha256=frame_sha256,
            unknown=bool(unknown),
            evidence_sources=tuple(evidence_sources),
            detail=dict(detail or {}),
        )

    def has_label(self, label: str) -> bool:
        wanted = (clean_text(label) or "").casefold()
        return any(value.casefold() == wanted for value in self.visible_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "phase": self.phase.value,
            "confidence": self.confidence,
            "theme_text": self.theme_text,
            "timer_seconds": self.timer_seconds,
            "visible_labels": list(self.visible_labels),
            "semantic_signature": self.semantic_signature,
            "frame_sha256": self.frame_sha256,
            "unknown": self.unknown,
            "evidence_sources": list(self.evidence_sources),
            "detail": dict(self.detail),
        }

