"""Shared contracts for the Generic Utility campaign.

The campaign is an experiment harness, not a second ScreenGhost runtime.  Every
record is deliberately explicit about observation source, action authority,
model use, settlement, and whether a metric is measured or simulated.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
Bounds = Tuple[float, float, float, float]


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json_bytes(value))


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact or None


def normalized_bounds(value: Sequence[Any]) -> Bounds:
    if len(value) != 4:
        raise ValueError(f"bounds require four values, got {value!r}")
    vals = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"bounds contain non-finite values: {vals!r}")
    x1, y1, x2, y2 = vals
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"normalized bounds must lie inside [0,1]: {vals!r}")
    return vals


class Operator(str, Enum):
    OPEN = "open"
    ACTIVATE = "activate"
    TOGGLE = "toggle"
    FILL = "fill"
    START = "start"
    STOP = "stop"
    BACK = "back"
    CHECK = "check"


class EvidenceSource(str, Enum):
    PIXELS = "pixels"
    VISUAL_INDEX = "visual_index"
    PHONE_GRAMMAR = "phone_grammar"
    APP_GRAPH = "app_graph"
    DECISION_CACHE = "decision_cache"
    SMALL_GROUNDER = "small_grounder"
    LARGE_VLM = "large_vlm"
    TEACHER = "teacher"
    HUMAN = "human"


class MetricKind(str, Enum):
    MEASURED = "measured"
    SIMULATED = "simulated"
    DERIVED = "derived"


@dataclass(frozen=True)
class SemanticGoal:
    goal_id: str
    operator: Operator
    target_label: Optional[str] = None
    target_role: Optional[str] = None
    value: Optional[str] = None
    expected_screen: Optional[str] = None
    expected_state_key: Optional[str] = None
    expected_state_value: Optional[str] = None

    def __post_init__(self) -> None:
        if not clean_text(self.goal_id):
            raise ValueError("goal_id is required")
        if self.operator is not Operator.BACK and not (
            clean_text(self.target_label) or clean_text(self.target_role)
        ):
            raise ValueError("non-back goals require target_label or target_role")

    @property
    def normalized_target(self) -> str:
        return clean_text(self.target_label) or clean_text(self.target_role) or "system"


@dataclass(frozen=True)
class VisibleElement:
    element_id: str
    role: str
    label: Optional[str]
    normalized_bounds: Bounds
    interactive: bool = False
    enabled: bool = True
    states: Mapping[str, str] = field(default_factory=dict)
    parent_element_id: Optional[str] = None
    sensitive: bool = False
    pixel_crop_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not clean_text(self.element_id):
            raise ValueError("element_id is required")
        if not clean_text(self.role):
            raise ValueError("role is required")
        object.__setattr__(self, "normalized_bounds", normalized_bounds(self.normalized_bounds))
        object.__setattr__(self, "states", dict(sorted((str(k), str(v)) for k, v in self.states.items())))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VisibleElement":
        raw_states = value.get("states") or {}
        if not isinstance(raw_states, Mapping):
            raw_states = dict(raw_states)
        return cls(
            element_id=str(value.get("element_id") or value.get("semantic_key") or ""),
            role=str(value.get("role") or "unknown"),
            label=clean_text(value.get("label")),
            normalized_bounds=normalized_bounds(value.get("normalized_bounds") or value.get("bounds")),
            interactive=bool(value.get("interactive", False)),
            enabled=bool(value.get("enabled", True)),
            states={str(k): str(v) for k, v in raw_states.items()},
            parent_element_id=(
                str(value["parent_element_id"])
                if value.get("parent_element_id") is not None
                else None
            ),
            sensitive=bool(value.get("sensitive", False)),
            pixel_crop_sha256=(
                str(value["pixel_crop_sha256"])
                if value.get("pixel_crop_sha256") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["normalized_bounds"] = list(self.normalized_bounds)
        value["states"] = dict(self.states)
        return value


@dataclass(frozen=True)
class StudentObservation:
    observation_id: str
    screen_key: Optional[str]
    surface_id: Optional[str]
    app_family: Optional[str]
    confidence: float
    unknown: bool
    elements: Tuple[VisibleElement, ...]
    evidence_sources: Tuple[str, ...]
    explanation: Optional[str] = None
    match_detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("observation confidence must be in [0,1]")
        if self.unknown and self.screen_key is not None:
            raise ValueError("unknown observation cannot carry a screen_key")
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "evidence_sources", tuple(str(v) for v in self.evidence_sources))
        object.__setattr__(self, "match_detail", dict(self.match_detail))

    def get_element(self, label: str) -> Optional[VisibleElement]:
        wanted = clean_text(label)
        if not wanted:
            return None
        wanted_cf = wanted.casefold()
        matches = [e for e in self.elements if (e.label or "").casefold() == wanted_cf]
        if not matches:
            return None
        matches.sort(key=lambda e: (not e.interactive, not e.enabled, e.role, e.element_id))
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "screen_key": self.screen_key,
            "surface_id": self.surface_id,
            "app_family": self.app_family,
            "confidence": self.confidence,
            "unknown": self.unknown,
            "elements": [e.to_dict() for e in self.elements],
            "evidence_sources": list(self.evidence_sources),
            "explanation": self.explanation,
            "match_detail": dict(self.match_detail),
        }


@dataclass(frozen=True)
class ResolvedAction:
    action_id: str
    operator: Operator
    target_element_id: Optional[str]
    target_role: Optional[str]
    target_label: Optional[str]
    normalized_point: Optional[Tuple[float, float]]
    text_value: Optional[str]
    expected_screen: Optional[str]
    expected_state_key: Optional[str]
    expected_state_value: Optional[str]
    confidence: float
    evidence_sources: Tuple[str, ...]
    decision_cache_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not clean_text(self.action_id):
            raise ValueError("action_id is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("action confidence must be in [0,1]")
        if self.normalized_point is not None:
            x, y = map(float, self.normalized_point)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("normalized action point must lie in [0,1]")
        object.__setattr__(self, "evidence_sources", tuple(str(v) for v in self.evidence_sources))


@dataclass(frozen=True)
class ControllerReceipt:
    receipt_id: str
    idempotency_key: str
    action_id: str
    status: str
    committed: bool
    injected: bool
    started_ms: float
    completed_ms: float
    settlement_ms: float
    before_observation_id: str
    after_observation_id: Optional[str]
    postcondition: Mapping[str, Any]
    reason: str
    action_authority: str = "external_controller"

    def __post_init__(self) -> None:
        if not clean_text(self.receipt_id):
            raise ValueError("receipt_id is required")
        if self.committed and not self.injected:
            raise ValueError("committed receipt must have injected=True")
        if self.settlement_ms < 0:
            raise ValueError("settlement_ms cannot be negative")
        object.__setattr__(self, "postcondition", dict(self.postcondition))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["postcondition"] = dict(self.postcondition)
        return value


@dataclass
class StepMetrics:
    teacher_reads: int = 0
    accessibility_events: int = 0
    screenshots_captured: int = 0
    stable_pair_attempts: int = 0
    stable_pair_refusals: int = 0
    visual_index_hits: int = 0
    generic_grammar_resolutions: int = 0
    app_atlas_resolutions: int = 0
    decision_cache_hits: int = 0
    small_grounder_calls: int = 0
    large_vlm_calls: int = 0
    gpu_load_count: int = 0
    gpu_active_ms: float = 0.0
    peak_vram_mb: float = 0.0
    planning_ms: float = 0.0
    grounding_ms: float = 0.0
    settlement_ms: float = 0.0
    total_ms: float = 0.0
    graph_edges_reused: int = 0
    graph_edges_discovered: int = 0
    unknown_true_positive: int = 0
    unknown_false_positive: int = 0
    actions_injected: int = 0
    duplicate_actions: int = 0
    pending_overlaps: int = 0
    pending_overlap_rejections: int = 0
    postcondition_successes: int = 0
    postcondition_failures: int = 0
    host_focus_changes: int = 0
    privileged_action_dependencies: int = 0
    model_timeouts: int = 0
    motor_calls_after_model_timeout: int = 0
    metric_kind: str = MetricKind.MEASURED.value

    def add(self, other: "StepMetrics") -> None:
        for name in self.__dataclass_fields__:
            if name == "metric_kind":
                if self.metric_kind != other.metric_kind:
                    self.metric_kind = MetricKind.DERIVED.value
                continue
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskReceipt:
    task_id: str
    phase: str
    success: bool
    steps: Tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    final_screen_key: Optional[str]
    failure_reason: Optional[str] = None
    evidence_tier: str = "generic_utility_experiment"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "phase": self.phase,
            "success": self.success,
            "steps": [dict(v) for v in self.steps],
            "metrics": dict(self.metrics),
            "final_screen_key": self.final_screen_key,
            "failure_reason": self.failure_reason,
            "evidence_tier": self.evidence_tier,
        }
