"""Optional measured semantic multibox run over installed vendor emulators.

The deterministic campaign is the premise proof.  This module is the final local
compatibility layer: the operator prepares independent clones at the same initial
state, supplies their names or indices, and ScreenGhost compares raw coordinate
replay with independently settled semantic replay.

Baseline and semantic instances are deliberately distinct.  The harness never
pretends it can restore a third-party app to a known state after one run unless the
operator supplied fresh clones or performed that reset externally.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from experiments.emulator_fleet.command import SubprocessCommandRunner
from experiments.emulator_fleet.distill import DistillationPolicy, distill_macro
from experiments.emulator_fleet.live import LiveFleetInstanceAdapter
from experiments.emulator_fleet.macro import load_macro
from experiments.emulator_fleet.providers.base import FleetProviderError
from experiments.emulator_fleet.providers.ldplayer import LDPlayerFleetProvider
from experiments.emulator_fleet.providers.memu import MEmuFleetProvider
from experiments.emulator_fleet.replay import ReplayPolicy, run_coordinate_baseline, run_semantic_procedure
from experiments.emulator_fleet.schema import (
    EmulatorVendor,
    FleetComparison,
    InstanceRef,
    SemanticProcedure,
    write_json,
)
from experiments.generic_utility.visual_index import VisualIndexPolicy, VisualStateIndex


MACHINE_PLAN_SCHEMA = "semantic_multibox_machine_plan_v1"


@dataclass(frozen=True)
class InstanceSelector:
    name: Optional[str] = None
    index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.name is None and self.index is None:
            raise ValueError("instance selector requires name or index")

    @classmethod
    def from_value(cls, value: Any) -> "InstanceSelector":
        if isinstance(value, int):
            return cls(index=value)
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, Mapping):
            return cls(
                name=(str(value["name"]) if value.get("name") is not None else None),
                index=(int(value["index"]) if value.get("index") is not None else None),
            )
        raise ValueError(f"invalid instance selector: {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "index": self.index}


@dataclass(frozen=True)
class MachinePlan:
    vendor: EmulatorVendor
    executable: str
    app_family: str
    macro_path: str
    macro_format: Optional[str]
    leader: InstanceSelector
    baseline_instances: tuple[InstanceSelector, ...]
    semantic_instances: tuple[InstanceSelector, ...]
    visual_teacher_instances: tuple[InstanceSelector, ...] = ()
    text_values: Mapping[str, str] = field(default_factory=dict)
    start_instances: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.vendor not in {EmulatorVendor.MEMU, EmulatorVendor.LDPLAYER}:
            raise ValueError("measured machine plan supports MEmu or LDPlayer")
        if not self.executable or not self.app_family or not self.macro_path:
            raise ValueError("executable, app_family, and macro_path are required")
        if not self.baseline_instances or not self.semantic_instances:
            raise ValueError("separate baseline and semantic instance sets are required")
        object.__setattr__(self, "text_values", dict(self.text_values))
        object.__setattr__(self, "notes", tuple(str(v) for v in self.notes))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MachinePlan":
        if value.get("schema") not in {None, MACHINE_PLAN_SCHEMA}:
            raise ValueError(f"unsupported machine plan schema: {value.get('schema')!r}")
        return cls(
            vendor=EmulatorVendor(str(value["vendor"])),
            executable=str(value["executable"]),
            app_family=str(value["app_family"]),
            macro_path=str(value["macro_path"]),
            macro_format=value.get("macro_format"),
            leader=InstanceSelector.from_value(value["leader"]),
            visual_teacher_instances=tuple(
                InstanceSelector.from_value(v)
                for v in value.get("visual_teacher_instances", ())
            ),
            baseline_instances=tuple(
                InstanceSelector.from_value(v) for v in value["baseline_instances"]
            ),
            semantic_instances=tuple(
                InstanceSelector.from_value(v) for v in value["semantic_instances"]
            ),
            text_values={str(k): str(v) for k, v in (value.get("text_values") or {}).items()},
            start_instances=bool(value.get("start_instances", False)),
            notes=tuple(value.get("notes") or ()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MachinePlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MACHINE_PLAN_SCHEMA,
            "vendor": self.vendor.value,
            "executable": self.executable,
            "app_family": self.app_family,
            "macro_path": self.macro_path,
            "macro_format": self.macro_format,
            "leader": self.leader.to_dict(),
            "visual_teacher_instances": [
                v.to_dict() for v in self.visual_teacher_instances
            ],
            "baseline_instances": [v.to_dict() for v in self.baseline_instances],
            "semantic_instances": [v.to_dict() for v in self.semantic_instances],
            "text_values": dict(self.text_values),
            "start_instances": self.start_instances,
            "notes": list(self.notes),
        }


def provider_for_plan(plan: MachinePlan, *, apply: bool):
    runner = SubprocessCommandRunner()
    if plan.vendor is EmulatorVendor.MEMU:
        return MEmuFleetProvider(plan.executable, runner=runner, apply=apply)
    if plan.vendor is EmulatorVendor.LDPLAYER:
        return LDPlayerFleetProvider(plan.executable, runner=runner, apply=apply)
    raise ValueError(f"unsupported provider {plan.vendor.value}")


def resolve_instance(
    instances: Sequence[InstanceRef], selector: InstanceSelector
) -> InstanceRef:
    matches = [
        row
        for row in instances
        if (selector.index is None or row.index == selector.index)
        and (selector.name is None or row.name == selector.name)
    ]
    if len(matches) != 1:
        raise FleetProviderError(
            f"selector {selector.to_dict()} matched {len(matches)} instances"
        )
    return matches[0]


class _IndexingTeacherBackend:
    """Record every distilled teacher state into the visual index."""

    def __init__(self, backend: LiveFleetInstanceAdapter, index: VisualStateIndex) -> None:
        self.backend = backend
        self.index = index
        self.instance_id = backend.instance_id

    @property
    def pending(self):
        return self.backend.pending

    def advance(self, milliseconds: float) -> None:
        self.backend.advance(milliseconds)

    def execute_macro_action(self, action, *, text_values):
        # Distillation uses normalized macro coordinates.  Keep the motor local to
        # the chosen leader and do not broadcast the demonstration.
        from experiments.emulator_fleet.schema import MacroActionKind

        if action.kind is MacroActionKind.TAP:
            return self.backend.tap_normalized(*action.point)
        if action.kind is MacroActionKind.LONG_PRESS:
            return self.backend.long_press_normalized(*action.point, action.duration_ms)
        if action.kind is MacroActionKind.SWIPE:
            return self.backend.swipe_normalized(action.path, action.duration_ms)
        if action.kind is MacroActionKind.TEXT:
            return self.backend.type_text(text_values[action.text_ref or ""])
        if action.kind is MacroActionKind.KEY:
            key = str(action.key or "").casefold()
            if key in {"back", "escape", "4", "keycode_back"}:
                return self.backend.back()
        raise FleetProviderError(f"unsupported live macro action: {action.kind.value}")

    def capture_teacher(self):
        projection = self.backend.capture_teacher()
        png = self.backend.last_teacher_png
        if png is None:
            raise FleetProviderError("teacher capture did not retain its aligned PNG")
        family = f"{projection.get('app_family')}:{projection.get('screen_name')}"
        self.index.add(
            png,
            projection,
            family_id=family,
            metadata={"source": "live_macro_distillation", "instance": self.instance_id},
        )
        return projection


def _teach_live_visual_variant(
    backend: LiveFleetInstanceAdapter,
    procedure: SemanticProcedure,
    index: VisualStateIndex,
) -> list[dict[str, Any]]:
    """Enroll one prepared geometry variant before runtime replay.

    The declared teacher instance is separate from every measured baseline and
    semantic instance. Privileged structure selects each taught target, while the
    resulting runtime index retains only pixels and the compiled projection.
    """

    receipts: list[dict[str, Any]] = []

    def capture(position: int, family_screen: str) -> Mapping[str, Any]:
        projection = dict(backend.capture_teacher())
        projection["screen_name"] = family_screen
        png = backend.last_teacher_png
        if png is None:
            raise FleetProviderError("visual variant teacher did not retain its aligned PNG")
        variant = index.add(
            png,
            projection,
            family_id=f"{procedure.app_family}:{family_screen}",
            metadata={
                "source": "live_visual_variant_teacher",
                "instance": backend.instance_id,
                "position": position,
            },
        )
        receipts.append(
            {
                "position": position,
                "family_screen": family_screen,
                "variant_id": variant.variant_id,
                "teacher_lesson_id": projection.get("lesson_id"),
            }
        )
        return projection

    projection = capture(0, procedure.initial_screen)
    for position, step in enumerate(procedure.steps, start=1):
        if step.target is None or not step.target.label:
            raise FleetProviderError(
                f"visual variant teaching requires a labeled target at step {step.sequence}"
            )
        candidates = [
            row
            for row in projection.get("elements", ())
            if row.get("interactive")
            and row.get("enabled", True)
            and row.get("label") == step.target.label
            and (not step.target.role or row.get("role") == step.target.role)
        ]
        if len(candidates) != 1:
            raise FleetProviderError(
                f"visual variant teacher target {step.target.label!r} "
                f"resolved to {len(candidates)} elements at step {step.sequence}"
            )
        x1, y1, x2, y2 = map(float, candidates[0]["normalized_bounds"])
        result = backend.tap_normalized((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if not result.injected:
            raise FleetProviderError(
                f"visual variant teacher failed at step {step.sequence}: {result.reason}"
            )
        backend.advance(1200)
        family_screen = step.expected_screen or (
            procedure.terminal_screen if position == len(procedure.steps) else ""
        )
        if not family_screen:
            raise FleetProviderError(
                f"visual variant teaching has no expected screen at step {step.sequence}"
            )
        projection = capture(position, family_screen)
    return receipts


def _terminal_teacher_score(
    backend: LiveFleetInstanceAdapter,
    procedure: SemanticProcedure,
    index: Optional[VisualStateIndex] = None,
) -> tuple[bool, Mapping[str, Any]]:
    projection = dict(backend.capture_teacher())
    terminal = str(procedure.terminal_screen or "")
    actual = str(projection.get("screen_name") or projection.get("screen_key") or "")
    ok = bool(terminal and actual == terminal)
    if procedure.steps:
        step = procedure.steps[-1]
        if step.expected_state_key and step.target is not None:
            ok = False
            for element in projection.get("elements", []):
                if step.target.label and element.get("label") != step.target.label:
                    continue
                if step.target.role and element.get("role") != step.target.role:
                    continue
                actual_state = str((element.get("states") or {}).get(step.expected_state_key) or "")
                if actual_state.casefold() == str(step.expected_state_value or "").casefold():
                    ok = True
                    break
    visual_match = None
    if not ok and index is not None and backend.last_teacher_png is not None and terminal:
        visual_match = index.match(
            backend.last_teacher_png,
            app_family_hint=procedure.app_family,
            screen_name_hint=terminal,
        )
        ok = visual_match.known
    return ok, {
        "teacher_score": ok,
        "expected_terminal_screen": terminal,
        "actual_terminal_screen": actual,
        "teacher_lesson_id": projection.get("lesson_id"),
        "terminal_visual_match": (
            {
                "known": visual_match.known,
                "confidence": visual_match.confidence,
                "margin": visual_match.margin,
                "reason": visual_match.reason,
            }
            if visual_match is not None
            else None
        ),
    }


def run_machine_plan(
    plan: MachinePlan,
    *,
    out_dir: str | Path,
    apply: bool,
) -> Path:
    if not apply:
        raise FleetProviderError(
            "measured fleet execution requires --apply; discovery and command planning remain dry-run by default"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "machine_plan.json", plan.to_dict())
    provider = provider_for_plan(plan, apply=True)
    capability = provider.capability()
    write_json(out / "provider_capability.json", capability.to_dict())
    if not capability.installed:
        raise FleetProviderError(f"provider executable not installed: {capability.executable}")
    inventory = provider.list_instances()
    write_json(out / "inventory.json", [row.to_dict() for row in inventory])
    leader_ref = resolve_instance(inventory, plan.leader)
    visual_teacher_refs = [
        resolve_instance(inventory, row) for row in plan.visual_teacher_instances
    ]
    baseline_refs = [resolve_instance(inventory, row) for row in plan.baseline_instances]
    semantic_refs = [resolve_instance(inventory, row) for row in plan.semantic_instances]
    selected = [leader_ref, *visual_teacher_refs, *baseline_refs, *semantic_refs]
    if len({row.instance_id for row in selected}) != len(selected):
        raise FleetProviderError("leader, baseline, and semantic instance sets must be disjoint")
    if plan.start_instances:
        for ref in selected:
            result = provider.start(ref)
            if not result.ok:
                raise FleetProviderError(f"failed to start {ref.instance_id}: {result.stderr_text()}")

    macro = load_macro(plan.macro_path, format_hint=plan.macro_format)
    write_json(out / "macro.json", macro.to_dict())
    index = VisualStateIndex(
        out / "visual_index.json",
        policy=VisualIndexPolicy(
            minimum_confidence=0.92,
            minimum_margin=0.03,
            minimum_crop_confidence=0.965,
            maximum_variants_per_family=96,
        ),
    )
    leader = LiveFleetInstanceAdapter(
        provider,
        leader_ref,
        app_family=plan.app_family,
    )
    procedure, distillation = distill_macro(
        macro,
        _IndexingTeacherBackend(leader, index),
        app_family=plan.app_family,
        text_values=plan.text_values,
        policy=DistillationPolicy(),
    )
    write_json(out / "semantic_procedure.json", procedure.to_dict())
    write_json(out / "distillation_receipt.json", distillation.to_dict())
    visual_teaching_receipts = []
    for ref in visual_teacher_refs:
        backend = LiveFleetInstanceAdapter(provider, ref, app_family=plan.app_family)
        visual_teaching_receipts.append(
            {
                "instance_id": ref.instance_id,
                "variants": _teach_live_visual_variant(backend, procedure, index),
            }
        )
    write_json(out / "visual_variant_teaching_receipts.json", visual_teaching_receipts)

    baseline_receipts = []
    for ref in baseline_refs:
        backend = LiveFleetInstanceAdapter(provider, ref, app_family=plan.app_family)
        receipt = run_coordinate_baseline(
            macro,
            backend,
            text_values=plan.text_values,
            expected_success=True,
        )
        score, detail = _terminal_teacher_score(backend, procedure, index)
        baseline_receipts.append(
            replace(
                receipt,
                success=score,
                failure_reason=None if score else "hidden teacher terminal score failed",
                metadata={**receipt.metadata, **dict(detail), "teacher_reads_for_scoring": 1},
            )
        )

    semantic_receipts = []
    for ref in semantic_refs:
        backend = LiveFleetInstanceAdapter(provider, ref, app_family=plan.app_family)
        receipt = run_semantic_procedure(
            procedure,
            backend,
            index,
            text_values=plan.text_values,
            expected_success=True,
            policy=ReplayPolicy(app_family_hint=plan.app_family),
        )
        score, detail = _terminal_teacher_score(backend, procedure, index)
        semantic_receipts.append(
            replace(
                receipt,
                success=receipt.success and score,
                failure_reason=(
                    receipt.failure_reason
                    if not receipt.success
                    else None if score else "hidden teacher terminal score failed"
                ),
                metadata={**receipt.metadata, **dict(detail), "teacher_reads_for_scoring": 1},
            )
        )

    runtime_teacher_reads = sum(row.teacher_reads_runtime for row in semantic_receipts)
    gates = {
        "distillation_has_steps": bool(procedure.steps),
        "baseline_instances_scored": all("teacher_score" in row.metadata for row in baseline_receipts),
        "semantic_instances_pass": all(row.success for row in semantic_receipts),
        "semantic_runtime_teacher_reads_zero": runtime_teacher_reads == 0,
        "semantic_duplicate_actions_zero": sum(row.duplicate_actions for row in semantic_receipts) == 0,
        "semantic_pending_overlap_zero": sum(row.pending_overlap_rejections for row in semantic_receipts) == 0,
    }
    comparison = FleetComparison.create(
        procedure_id=procedure.procedure_id,
        baseline=baseline_receipts,
        semantic=semantic_receipts,
        gates=gates,
        metrics={
            "baseline_count": len(baseline_receipts),
            "semantic_count": len(semantic_receipts),
            "baseline_successes": sum(row.success for row in baseline_receipts),
            "semantic_successes": sum(row.success for row in semantic_receipts),
            "runtime_teacher_reads": runtime_teacher_reads,
            "visual_families": index.family_count,
            "visual_variants": len(index.variants),
        },
    )
    write_json(out / "comparison.json", comparison.to_dict())
    write_json(out / "baseline_receipts.json", [row.to_dict() for row in baseline_receipts])
    write_json(out / "semantic_receipts.json", [row.to_dict() for row in semantic_receipts])
    conclusion = {
        "schema": "semantic_multibox_machine_conclusion_v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "allowed_conclusion": (
            f"The measured {plan.vendor.value} fleet executed the distilled semantic procedure independently across the declared clones."
            if all(gates.values())
            else "No measured fleet conclusion; one or more required gates failed."
        ),
        "forbidden_conclusions": [
            "other emulator vendors passed",
            "untested third-party apps passed",
            "coordinate synchronization is safe under divergent state",
        ],
    }
    write_json(out / "CONCLUSION.json", conclusion)
    if not all(gates.values()):
        raise FleetProviderError(
            "measured fleet gates failed: " + ", ".join(name for name, ok in gates.items() if not ok)
        )
    return out
