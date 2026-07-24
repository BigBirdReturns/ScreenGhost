"""Distill coordinate demonstrations into semantic ScreenGhost procedures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from experiments.emulator_fleet.schema import (
    CoordinateMacro,
    MacroAction,
    MacroActionKind,
    ProcedureOperator,
    SemanticProcedure,
    SemanticProcedureStep,
    SemanticTargetSpec,
    sha256_json,
)


class DistillationError(RuntimeError):
    pass


@runtime_checkable
class MacroDistillationBackend(Protocol):
    instance_id: str

    @property
    def pending(self) -> bool: ...
    def capture_teacher(self) -> Mapping[str, Any]: ...
    def execute_macro_action(self, action: MacroAction, *, text_values: Mapping[str, str]) -> Any: ...
    def advance(self, milliseconds: float) -> None: ...


@dataclass(frozen=True)
class DistillationPolicy:
    settle_poll_ms: float = 80.0
    settle_timeout_ms: float = 6000.0
    minimum_after_ms: float = 180.0
    minimum_target_confidence: float = 0.84
    allow_unlabeled_unique_role: bool = True

    def require_valid(self) -> None:
        if self.settle_poll_ms <= 0 or self.settle_timeout_ms <= 0:
            raise ValueError("settlement timings must be positive")
        if self.minimum_after_ms < 0:
            raise ValueError("minimum_after_ms cannot be negative")
        if not 0 <= self.minimum_target_confidence <= 1:
            raise ValueError("minimum_target_confidence must be in [0,1]")


@dataclass(frozen=True)
class DistillationReceipt:
    receipt_id: str
    macro_id: str
    procedure_id: str
    training_instance_id: str
    steps_compiled: int
    wait_actions_consumed: int
    teacher_reads: int
    action_receipts: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "macro_distillation_receipt_v1",
            "receipt_id": self.receipt_id,
            "macro_id": self.macro_id,
            "procedure_id": self.procedure_id,
            "training_instance_id": self.training_instance_id,
            "steps_compiled": self.steps_compiled,
            "wait_actions_consumed": self.wait_actions_consumed,
            "teacher_reads": self.teacher_reads,
            "action_receipts": [dict(v) for v in self.action_receipts],
        }


def _elements(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = projection.get("elements") or []
    if not isinstance(rows, Sequence):
        raise DistillationError("teacher projection elements must be a sequence")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _contains(bounds: Sequence[Any], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = map(float, bounds)
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def _area(bounds: Sequence[Any]) -> float:
    x1, y1, x2, y2 = map(float, bounds)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _target_at(projection: Mapping[str, Any], point: tuple[float, float]) -> Optional[dict[str, Any]]:
    candidates = [
        row
        for row in _elements(projection)
        if row.get("interactive")
        and row.get("enabled", True)
        and row.get("normalized_bounds")
        and _contains(row["normalized_bounds"], point)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            _area(row["normalized_bounds"]),
            not bool(row.get("label")),
            str(row.get("role") or ""),
            str(row.get("element_id") or ""),
        )
    )
    return candidates[0]


def _scroll_target(projection: Mapping[str, Any], point: tuple[float, float]) -> Optional[dict[str, Any]]:
    candidates = [
        row
        for row in _elements(projection)
        if row.get("normalized_bounds")
        and _contains(row["normalized_bounds"], point)
        and str(row.get("role") or "") in {"list", "scroll_view", "web_view", "group"}
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-_area(row["normalized_bounds"]), str(row.get("element_id") or "")))
    return candidates[0]


def _focused_field(projection: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    fields = [
        row
        for row in _elements(projection)
        if str(row.get("role") or "") in {"text_field", "input", "textbox", "searchbox"}
        and row.get("interactive")
        and row.get("enabled", True)
    ]
    focused = [
        row
        for row in fields
        if str((row.get("states") or {}).get("focused", "false")).lower() == "true"
    ]
    pool = focused or fields
    if len(pool) == 1:
        return pool[0]
    return None


def _same_semantic_element(target: Mapping[str, Any], projection: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    label = str(target.get("label") or "").casefold()
    role = str(target.get("role") or "")
    candidates = [row for row in _elements(projection) if str(row.get("role") or "") == role]
    if label:
        matches = [row for row in candidates if str(row.get("label") or "").casefold() == label]
        if matches:
            return sorted(matches, key=lambda row: str(row.get("element_id") or ""))[0]
    return candidates[0] if len(candidates) == 1 else None


def _derive_postcondition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    target: Optional[Mapping[str, Any]],
) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    before_screen = str(before.get("screen_name") or before.get("screen_key") or "")
    after_screen = str(after.get("screen_name") or after.get("screen_key") or "")
    if after_screen and after_screen != before_screen:
        return after_screen, None, None, "screen_transition"
    if target is not None:
        current = _same_semantic_element(target, before)
        following = _same_semantic_element(target, after)
        before_states = dict((current or {}).get("states") or {})
        after_states = dict((following or {}).get("states") or {})
        for key in sorted(set(before_states) | set(after_states)):
            old = str(before_states.get(key, ""))
            new = str(after_states.get(key, ""))
            if new != old and new:
                return None, key, new, "element_state_transition"
    if before.get("content_hash") != after.get("content_hash"):
        return None, None, None, "visible_content_transition"
    return None, None, None, "action_acceptance_only"


def _operator(action: MacroAction, target: Optional[Mapping[str, Any]], transition_kind: str) -> ProcedureOperator:
    if action.kind is MacroActionKind.TEXT:
        return ProcedureOperator.FILL
    if action.kind is MacroActionKind.KEY:
        key = str(action.key or "").casefold()
        if key in {"back", "escape", "keycode_back", "4"}:
            return ProcedureOperator.BACK
        if key in {"home", "keycode_home", "3"}:
            return ProcedureOperator.HOME
        return ProcedureOperator.ACTIVATE
    if action.kind is MacroActionKind.SWIPE:
        return ProcedureOperator.SCROLL
    if action.kind is MacroActionKind.LONG_PRESS:
        return ProcedureOperator.LONG_PRESS
    role = str((target or {}).get("role") or "")
    if role in {"switch", "checkbox", "radio", "toggle"}:
        return ProcedureOperator.TOGGLE
    if transition_kind == "screen_transition":
        return ProcedureOperator.OPEN
    return ProcedureOperator.ACTIVATE


def _target_spec(target: Optional[Mapping[str, Any]], *, policy: DistillationPolicy) -> Optional[SemanticTargetSpec]:
    if target is None:
        return None
    role = str(target.get("role") or "") or None
    label = str(target.get("label") or "").strip() or None
    if label is None and not policy.allow_unlabeled_unique_role:
        raise DistillationError("demonstration target has no stable visible label")
    return SemanticTargetSpec(
        role=role,
        label=label,
        allow_unique_role_fallback=policy.allow_unlabeled_unique_role,
    )


def distill_macro(
    macro: CoordinateMacro,
    backend: MacroDistillationBackend,
    *,
    app_family: str,
    procedure_name: Optional[str] = None,
    text_values: Optional[Mapping[str, str]] = None,
    policy: DistillationPolicy = DistillationPolicy(),
) -> tuple[SemanticProcedure, DistillationReceipt]:
    """Execute one coordinate macro with teacher visibility and compile its meaning."""

    policy.require_valid()
    unsupported = macro.unsupported_actions
    if unsupported:
        raise DistillationError(
            f"macro contains {len(unsupported)} unsupported action(s): "
            + ", ".join(row.kind.value for row in unsupported[:5])
        )
    values = dict(text_values or {})
    initial = backend.capture_teacher()
    teacher_reads = 1
    initial_screen = str(initial.get("screen_name") or initial.get("screen_key") or "unknown")
    compiled: list[SemanticProcedureStep] = []
    logs: list[Mapping[str, Any]] = []
    wait_actions = 0
    carried_wait_ms = 0.0

    for action in macro.actions:
        if action.kind is MacroActionKind.WAIT:
            backend.advance(action.duration_ms)
            carried_wait_ms += action.duration_ms
            wait_actions += 1
            continue
        before = backend.capture_teacher()
        teacher_reads += 1
        target: Optional[dict[str, Any]] = None
        if action.kind in {MacroActionKind.TAP, MacroActionKind.LONG_PRESS} and action.point is not None:
            target = _target_at(before, action.point)
            if target is None:
                raise DistillationError(
                    f"macro action {action.sequence} points at no enabled teacher element"
                )
        elif action.kind is MacroActionKind.SWIPE:
            target = _scroll_target(before, action.path[0])
            if target is None:
                raise DistillationError(
                    f"macro swipe {action.sequence} starts outside a known scroll container"
                )
        elif action.kind is MacroActionKind.TEXT:
            target = _focused_field(before)
            if target is None:
                raise DistillationError(
                    f"macro text action {action.sequence} has no unique focused/editable field"
                )
            if action.text_ref not in values:
                raise DistillationError(
                    f"macro text action {action.sequence} requires runtime value {action.text_ref!r}"
                )
        result = backend.execute_macro_action(action, text_values=values)
        injected = bool(getattr(result, "injected", False))
        accepted = bool(getattr(result, "accepted", injected))
        if action.kind not in {MacroActionKind.KEY} and not injected:
            raise DistillationError(
                f"macro action {action.sequence} was not injected: {getattr(result, 'reason', '')}"
            )
        elapsed = 0.0
        while backend.pending and elapsed < policy.settle_timeout_ms:
            backend.advance(policy.settle_poll_ms)
            elapsed += policy.settle_poll_ms
        if backend.pending:
            raise DistillationError(f"macro action {action.sequence} did not settle")
        backend.advance(policy.minimum_after_ms)
        after = backend.capture_teacher()
        teacher_reads += 1
        expected_screen, state_key, state_value, transition_kind = _derive_postcondition(before, after, target)
        operator = _operator(action, target, transition_kind)
        target_spec = _target_spec(target, policy=policy)
        if operator in {ProcedureOperator.BACK, ProcedureOperator.HOME}:
            target_spec = None
        confidence = 1.0 if target and target.get("label") else 0.9 if target else 1.0
        if confidence < policy.minimum_target_confidence:
            raise DistillationError(
                f"macro action {action.sequence} target confidence {confidence:.3f} below gate"
            )
        step = SemanticProcedureStep(
            sequence=len(compiled),
            operator=operator,
            target=target_spec,
            value_ref=action.text_ref,
            swipe_path=action.path,
            expected_screen=expected_screen,
            expected_state_key=state_key,
            expected_state_value=state_value,
            source_action_sha256=sha256_json(action.to_dict()),
            confidence=confidence,
            evidence={
                "macro_sequence": action.sequence,
                "demonstration_point": list(action.point) if action.point else None,
                "demonstration_wait_ms": carried_wait_ms,
                "settlement_ms": elapsed + policy.minimum_after_ms,
                "transition_kind": transition_kind,
                "accepted": accepted,
                "before_screen": before.get("screen_name") or before.get("screen_key"),
                "after_screen": after.get("screen_name") or after.get("screen_key"),
                "target_element_id": (target or {}).get("element_id"),
            },
        )
        compiled.append(step)
        logs.append(
            {
                "macro_action": action.to_dict(),
                "semantic_step": step.to_dict(),
                "result": {
                    "accepted": accepted,
                    "injected": injected,
                    "reason": getattr(result, "reason", None),
                },
            }
        )
        carried_wait_ms = 0.0

    final = backend.capture_teacher()
    teacher_reads += 1
    terminal_screen = str(final.get("screen_name") or final.get("screen_key") or "") or None
    procedure = SemanticProcedure.create(
        name=procedure_name or f"Distilled {macro.name}",
        app_family=app_family,
        source_macro=macro,
        steps=compiled,
        training_instance_id=backend.instance_id,
        initial_screen=initial_screen,
        terminal_screen=terminal_screen,
    )
    receipt = DistillationReceipt(
        receipt_id="distillation_" + sha256_json(
            {
                "macro": macro.macro_id,
                "procedure": procedure.procedure_id,
                "instance": backend.instance_id,
                "logs": logs,
            }
        ),
        macro_id=macro.macro_id,
        procedure_id=procedure.procedure_id,
        training_instance_id=backend.instance_id,
        steps_compiled=len(compiled),
        wait_actions_consumed=wait_actions,
        teacher_reads=teacher_reads,
        action_receipts=tuple(logs),
    )
    return procedure, receipt
