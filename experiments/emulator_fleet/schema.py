"""Contracts for semantic multiboxing over Android emulator fleets.

The fleet layer separates three authorities:

* vendor lifecycle authority starts, stops, clones, and configures emulator VMs;
* coordinate-macro authority provides a reproducible baseline demonstration;
* ScreenGhost semantic authority resolves the meaning of that demonstration and
  executes one verified transaction independently in each instance.

The records are content addressed so baseline and semantic runs can be compared
without silently changing their inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
Point = Tuple[float, float]
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


def normalize_point(x: Any, y: Any) -> Point:
    vals = (float(x), float(y))
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"point contains non-finite values: {vals!r}")
    if not all(0.0 <= v <= 1.0 for v in vals):
        raise ValueError(f"normalized point must lie in [0,1]: {vals!r}")
    return vals


def normalize_bounds(value: Sequence[Any]) -> Bounds:
    if len(value) != 4:
        raise ValueError("bounds require four values")
    vals = tuple(float(v) for v in value)
    x1, y1, x2, y2 = vals
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("bounds contain non-finite values")
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"invalid normalized bounds: {vals!r}")
    return vals


class EmulatorVendor(str, Enum):
    SIMULATED = "simulated"
    MEMU = "memu"
    LDPLAYER = "ldplayer"
    BLUESTACKS = "bluestacks"


class InstanceStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class MacroActionKind(str, Enum):
    TAP = "tap"
    SWIPE = "swipe"
    LONG_PRESS = "long_press"
    TEXT = "text"
    KEY = "key"
    WAIT = "wait"
    LAUNCH = "launch"
    CONTROL = "control"
    UNKNOWN = "unknown"


class ProcedureOperator(str, Enum):
    OPEN = "open"
    ACTIVATE = "activate"
    TOGGLE = "toggle"
    FILL = "fill"
    BACK = "back"
    HOME = "home"
    SCROLL = "scroll"
    LONG_PRESS = "long_press"
    WAIT = "wait"


@dataclass(frozen=True)
class InstanceRef:
    vendor: EmulatorVendor
    instance_id: str
    name: str
    index: Optional[int] = None
    status: InstanceStatus = InstanceStatus.UNKNOWN
    window_handle: Optional[int] = None
    pid: Optional[int] = None
    adb_serial: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not clean_text(self.instance_id):
            raise ValueError("instance_id is required")
        if not clean_text(self.name):
            raise ValueError("instance name is required")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def selector(self) -> dict[str, Any]:
        if self.index is not None:
            return {"index": self.index}
        return {"name": self.name}

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["vendor"] = self.vendor.value
        value["status"] = self.status.value
        value["metadata"] = dict(self.metadata)
        return value


@dataclass(frozen=True)
class InstanceProfile:
    profile_id: str
    width: int
    height: int
    dpi: int
    cpu_cores: int = 2
    memory_mb: int = 2048
    fps: int = 10
    theme: str = "light"
    font_scale: float = 1.0
    locale: str = "en-US"
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not clean_text(self.profile_id):
            raise ValueError("profile_id is required")
        if self.width <= 0 or self.height <= 0 or self.dpi <= 0:
            raise ValueError("profile geometry must be positive")
        if self.cpu_cores <= 0 or self.memory_mb <= 0 or self.fps <= 0:
            raise ValueError("profile resources must be positive")
        if not 0.75 <= float(self.font_scale) <= 2.0:
            raise ValueError("font_scale must be in [0.75,2.0]")
        object.__setattr__(self, "tags", tuple(sorted(str(v) for v in self.tags)))

    @property
    def resolution(self) -> str:
        return f"{self.width},{self.height},{self.dpi}"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tags"] = list(self.tags)
        return value


@dataclass(frozen=True)
class MacroAction:
    sequence: int
    kind: MacroActionKind
    at_ms: float
    point: Optional[Point] = None
    path: Tuple[Point, ...] = ()
    duration_ms: float = 0.0
    key: Optional[str] = None
    text_ref: Optional[str] = None
    text_length: Optional[int] = None
    text_sha256: Optional[str] = None
    package: Optional[str] = None
    raw: Optional[str] = None
    supported: bool = True

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative")
        if self.at_ms < 0 or self.duration_ms < 0:
            raise ValueError("macro timings cannot be negative")
        if self.point is not None:
            object.__setattr__(self, "point", normalize_point(*self.point))
        object.__setattr__(self, "path", tuple(normalize_point(*p) for p in self.path))
        if self.kind in {MacroActionKind.TAP, MacroActionKind.LONG_PRESS} and self.point is None:
            raise ValueError(f"{self.kind.value} action requires a point")
        if self.kind is MacroActionKind.SWIPE and len(self.path) < 2:
            raise ValueError("swipe action requires at least two path points")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["point"] = list(self.point) if self.point is not None else None
        value["path"] = [list(p) for p in self.path]
        return value


@dataclass(frozen=True)
class CoordinateMacro:
    macro_id: str
    name: str
    vendor: EmulatorVendor
    source_sha256: str
    source_resolution: Tuple[int, int]
    actions: Tuple[MacroAction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not clean_text(self.name):
            raise ValueError("macro name is required")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a full SHA-256 digest")
        width, height = self.source_resolution
        if width <= 0 or height <= 0:
            raise ValueError("source resolution must be positive")
        ordered = tuple(sorted(self.actions, key=lambda row: row.sequence))
        if len({row.sequence for row in ordered}) != len(ordered):
            raise ValueError("macro action sequences must be unique")
        object.__setattr__(self, "actions", ordered)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def create(
        cls,
        *,
        name: str,
        vendor: EmulatorVendor,
        source_bytes: bytes,
        source_resolution: Tuple[int, int],
        actions: Sequence[MacroAction],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "CoordinateMacro":
        source_sha = sha256_bytes(source_bytes)
        payload = {
            "name": name,
            "vendor": vendor.value,
            "source_sha256": source_sha,
            "source_resolution": list(source_resolution),
            "actions": [row.to_dict() for row in actions],
        }
        return cls(
            macro_id="macro_" + sha256_json(payload),
            name=name,
            vendor=vendor,
            source_sha256=source_sha,
            source_resolution=source_resolution,
            actions=tuple(actions),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CoordinateMacro":
        actions = []
        for row in value.get("actions", []):
            actions.append(
                MacroAction(
                    sequence=int(row["sequence"]),
                    kind=MacroActionKind(str(row["kind"])),
                    at_ms=float(row.get("at_ms", 0.0)),
                    point=(tuple(row["point"]) if row.get("point") is not None else None),
                    path=tuple(tuple(point) for point in row.get("path", [])),
                    duration_ms=float(row.get("duration_ms", 0.0)),
                    key=row.get("key"),
                    text_ref=row.get("text_ref"),
                    text_length=row.get("text_length"),
                    text_sha256=row.get("text_sha256"),
                    package=row.get("package"),
                    raw=row.get("raw"),
                    supported=bool(row.get("supported", True)),
                )
            )
        return cls(
            macro_id=str(value["macro_id"]),
            name=str(value["name"]),
            vendor=EmulatorVendor(str(value["vendor"])),
            source_sha256=str(value["source_sha256"]),
            source_resolution=tuple(int(v) for v in value["source_resolution"]),
            actions=tuple(actions),
            metadata=dict(value.get("metadata") or {}),
        )

    @property
    def unsupported_actions(self) -> Tuple[MacroAction, ...]:
        return tuple(row for row in self.actions if not row.supported)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "coordinate_macro_v1",
            "macro_id": self.macro_id,
            "name": self.name,
            "vendor": self.vendor.value,
            "source_sha256": self.source_sha256,
            "source_resolution": list(self.source_resolution),
            "actions": [row.to_dict() for row in self.actions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SemanticTargetSpec:
    role: Optional[str]
    label: Optional[str]
    parent_role: Optional[str] = None
    parent_label: Optional[str] = None
    relative_position: Optional[str] = None
    allow_unique_role_fallback: bool = True

    def __post_init__(self) -> None:
        if not clean_text(self.role) and not clean_text(self.label):
            raise ValueError("semantic target requires role or label")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticProcedureStep:
    sequence: int
    operator: ProcedureOperator
    target: Optional[SemanticTargetSpec]
    value_ref: Optional[str]
    swipe_path: Tuple[Point, ...]
    expected_screen: Optional[str]
    expected_state_key: Optional[str]
    expected_state_value: Optional[str]
    source_action_sha256: str
    confidence: float
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.operator not in {ProcedureOperator.BACK, ProcedureOperator.HOME, ProcedureOperator.WAIT} and self.target is None:
            raise ValueError(f"{self.operator.value} step requires a target")
        object.__setattr__(self, "swipe_path", tuple(normalize_point(*p) for p in self.swipe_path))
        object.__setattr__(self, "evidence", dict(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "operator": self.operator.value,
            "target": self.target.to_dict() if self.target is not None else None,
            "value_ref": self.value_ref,
            "swipe_path": [list(p) for p in self.swipe_path],
            "expected_screen": self.expected_screen,
            "expected_state_key": self.expected_state_key,
            "expected_state_value": self.expected_state_value,
            "source_action_sha256": self.source_action_sha256,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class SemanticProcedure:
    procedure_id: str
    name: str
    app_family: str
    source_macro_id: str
    source_vendor: EmulatorVendor
    steps: Tuple[SemanticProcedureStep, ...]
    training_instance_id: str
    initial_screen: str
    terminal_screen: Optional[str]
    evidence_tier: str = "macro_distilled_semantic_procedure"
    limits: Tuple[str, ...] = (
        "compiled from one coordinate demonstration plus visible teacher structure",
        "does not assert vendor backend truth",
        "coordinates are demonstration evidence, not durable runtime targets",
        "runtime must resolve each target against the current instance",
    )

    def __post_init__(self) -> None:
        if not clean_text(self.name) or not clean_text(self.app_family):
            raise ValueError("procedure name and app_family are required")
        ordered = tuple(sorted(self.steps, key=lambda row: row.sequence))
        object.__setattr__(self, "steps", ordered)
        object.__setattr__(self, "limits", tuple(self.limits))

    @classmethod
    def create(
        cls,
        *,
        name: str,
        app_family: str,
        source_macro: CoordinateMacro,
        steps: Sequence[SemanticProcedureStep],
        training_instance_id: str,
        initial_screen: str,
        terminal_screen: Optional[str],
    ) -> "SemanticProcedure":
        payload = {
            "name": name,
            "app_family": app_family,
            "source_macro_id": source_macro.macro_id,
            "steps": [row.to_dict() for row in steps],
            "training_instance_id": training_instance_id,
            "initial_screen": initial_screen,
            "terminal_screen": terminal_screen,
        }
        return cls(
            procedure_id="procedure_" + sha256_json(payload),
            name=name,
            app_family=app_family,
            source_macro_id=source_macro.macro_id,
            source_vendor=source_macro.vendor,
            steps=tuple(steps),
            training_instance_id=training_instance_id,
            initial_screen=initial_screen,
            terminal_screen=terminal_screen,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticProcedure":
        steps = []
        for row in value.get("steps", []):
            target_row = row.get("target")
            target = SemanticTargetSpec(**target_row) if target_row is not None else None
            steps.append(
                SemanticProcedureStep(
                    sequence=int(row["sequence"]),
                    operator=ProcedureOperator(str(row["operator"])),
                    target=target,
                    value_ref=row.get("value_ref"),
                    swipe_path=tuple(tuple(point) for point in row.get("swipe_path", [])),
                    expected_screen=row.get("expected_screen"),
                    expected_state_key=row.get("expected_state_key"),
                    expected_state_value=row.get("expected_state_value"),
                    source_action_sha256=str(row["source_action_sha256"]),
                    confidence=float(row.get("confidence", 0.0)),
                    evidence=dict(row.get("evidence") or {}),
                )
            )
        return cls(
            procedure_id=str(value["procedure_id"]),
            name=str(value["name"]),
            app_family=str(value["app_family"]),
            source_macro_id=str(value["source_macro_id"]),
            source_vendor=EmulatorVendor(str(value["source_vendor"])),
            steps=tuple(steps),
            training_instance_id=str(value["training_instance_id"]),
            initial_screen=str(value["initial_screen"]),
            terminal_screen=value.get("terminal_screen"),
            evidence_tier=str(value.get("evidence_tier") or "macro_distilled_semantic_procedure"),
            limits=tuple(value.get("limits") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "semantic_procedure_v1",
            "procedure_id": self.procedure_id,
            "name": self.name,
            "app_family": self.app_family,
            "source_macro_id": self.source_macro_id,
            "source_vendor": self.source_vendor.value,
            "training_instance_id": self.training_instance_id,
            "initial_screen": self.initial_screen,
            "terminal_screen": self.terminal_screen,
            "steps": [row.to_dict() for row in self.steps],
            "evidence_tier": self.evidence_tier,
            "limits": list(self.limits),
        }


@dataclass(frozen=True)
class InstanceRunReceipt:
    instance_id: str
    mode: str
    success: bool
    expected_success: bool
    actions_attempted: int
    actions_injected: int
    committed_actions: int
    duplicate_actions: int
    pending_overlap_rejections: int
    teacher_reads_runtime: int
    large_model_calls: int
    final_screen: Optional[str]
    failure_reason: Optional[str]
    step_receipts: Tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_receipts", tuple(dict(v) for v in self.step_receipts))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "mode": self.mode,
            "success": self.success,
            "expected_success": self.expected_success,
            "actions_attempted": self.actions_attempted,
            "actions_injected": self.actions_injected,
            "committed_actions": self.committed_actions,
            "duplicate_actions": self.duplicate_actions,
            "pending_overlap_rejections": self.pending_overlap_rejections,
            "teacher_reads_runtime": self.teacher_reads_runtime,
            "large_model_calls": self.large_model_calls,
            "final_screen": self.final_screen,
            "failure_reason": self.failure_reason,
            "step_receipts": [dict(v) for v in self.step_receipts],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FleetComparison:
    comparison_id: str
    procedure_id: str
    baseline: Tuple[InstanceRunReceipt, ...]
    semantic: Tuple[InstanceRunReceipt, ...]
    gates: Mapping[str, bool]
    metrics: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        procedure_id: str,
        baseline: Sequence[InstanceRunReceipt],
        semantic: Sequence[InstanceRunReceipt],
        gates: Mapping[str, bool],
        metrics: Mapping[str, Any],
    ) -> "FleetComparison":
        payload = {
            "procedure_id": procedure_id,
            "baseline": [row.to_dict() for row in baseline],
            "semantic": [row.to_dict() for row in semantic],
            "gates": dict(gates),
            "metrics": dict(metrics),
        }
        return cls(
            comparison_id="fleet_comparison_" + sha256_json(payload),
            procedure_id=procedure_id,
            baseline=tuple(baseline),
            semantic=tuple(semantic),
            gates=dict(gates),
            metrics=dict(metrics),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "semantic_multibox_comparison_v1",
            "comparison_id": self.comparison_id,
            "procedure_id": self.procedure_id,
            "baseline": [row.to_dict() for row in self.baseline],
            "semantic": [row.to_dict() for row in self.semantic],
            "gates": dict(self.gates),
            "metrics": dict(self.metrics),
        }


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(json_bytes(value))
    return target
