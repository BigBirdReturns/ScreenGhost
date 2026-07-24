"""Coordinate baseline and independently verified semantic fleet replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from experiments.emulator_fleet.schema import (
    CoordinateMacro,
    InstanceRunReceipt,
    MacroActionKind,
    ProcedureOperator,
    SemanticProcedure,
    SemanticProcedureStep,
)
from experiments.generic_utility.phone_grammar import PhoneGrammar
from experiments.generic_utility.schema import Operator, SemanticGoal, StudentObservation
from experiments.generic_utility.transaction import (
    ObservationContext,
    PendingActionError,
    SettlementPolicy,
    TransactionalController,
)
from experiments.generic_utility.visual_index import VisualStateIndex


@runtime_checkable
class FleetRuntimeBackend(Protocol):
    instance_id: str

    @property
    def pending(self) -> bool: ...
    @property
    def teacher_reads(self) -> int: ...
    @property
    def actions_injected(self) -> int: ...
    def capture_png(self) -> bytes: ...
    def tap_normalized(self, x: float, y: float) -> Any: ...
    def tap_source_pixels(self, x: int, y: int) -> Any: ...
    def type_text(self, text: str) -> Any: ...
    def back(self) -> Any: ...
    def advance(self, milliseconds: float) -> None: ...
    def swipe_normalized(self, path, duration_ms: float = 300.0) -> Any: ...
    def long_press_normalized(self, x: float, y: float, duration_ms: float) -> Any: ...
    def current_screen(self) -> str: ...


@dataclass(frozen=True)
class ReplayPolicy:
    visual_surface_hint: Optional[str] = None
    app_family_hint: Optional[str] = None
    minimum_semantic_confidence: float = 0.80
    settlement: SettlementPolicy = SettlementPolicy(
        poll_interval_ms=80.0,
        timeout_ms=5000.0,
        stable_samples=3,
        minimum_stable_ms=160.0,
        require_visible_change=True,
    )


class _MotorAdapter:
    def __init__(self, backend: FleetRuntimeBackend) -> None:
        self.backend = backend

    def tap_normalized(self, x: float, y: float):
        return self.backend.tap_normalized(x, y)

    def type_text(self, text: str):
        return self.backend.type_text(text)

    def back(self):
        return self.backend.back()

    def advance(self, milliseconds: float) -> None:
        self.backend.advance(milliseconds)


def _operator(step: SemanticProcedureStep) -> Operator:
    return {
        ProcedureOperator.OPEN: Operator.OPEN,
        ProcedureOperator.ACTIVATE: Operator.ACTIVATE,
        ProcedureOperator.TOGGLE: Operator.TOGGLE,
        ProcedureOperator.FILL: Operator.FILL,
        ProcedureOperator.BACK: Operator.BACK,
        ProcedureOperator.HOME: Operator.BACK,
        ProcedureOperator.LONG_PRESS: Operator.ACTIVATE,
        ProcedureOperator.SCROLL: Operator.ACTIVATE,
        ProcedureOperator.WAIT: Operator.CHECK,
    }[step.operator]


def _observation(
    backend: FleetRuntimeBackend,
    index: VisualStateIndex,
    context: Optional[ObservationContext],
    policy: ReplayPolicy,
) -> StudentObservation:
    expected = context.expected_screen if context is not None else None
    match = index.match(
        backend.capture_png(),
        surface_hint=policy.visual_surface_hint,
        app_family_hint=policy.app_family_hint,
        screen_name_hint=expected,
        target_label_hint=(context.target_label if context is not None else None),
        target_role_hint=(context.target_role if context is not None else None),
        state_key_hint=(context.expected_state_key if context is not None else None),
        state_value_hint=(context.expected_state_value if context is not None else None),
    )
    return match.to_student_observation()


def _goal(step: SemanticProcedureStep, *, value: Optional[str]) -> SemanticGoal:
    target = step.target
    return SemanticGoal(
        goal_id=f"procedure-step-{step.sequence}",
        operator=_operator(step),
        target_label=target.label if target is not None else None,
        target_role=target.role if target is not None else None,
        value=value,
        expected_screen=step.expected_screen,
        expected_state_key=step.expected_state_key,
        expected_state_value=step.expected_state_value,
    )


def _resolve_with_role_fallback(
    grammar: PhoneGrammar,
    step: SemanticProcedureStep,
    observation: StudentObservation,
    *,
    value: Optional[str],
    minimum_confidence: float,
):
    goal = _goal(step, value=value)
    result = grammar.resolve(
        goal,
        observation,
        app_specific_memory=True,
        minimum_confidence=minimum_confidence,
    )
    if result.resolved or step.target is None or not step.target.allow_unique_role_fallback:
        return result
    fallback = SemanticGoal(
        goal_id=goal.goal_id + "-role-fallback",
        operator=goal.operator,
        target_label=None,
        target_role=step.target.role,
        value=value,
        expected_screen=goal.expected_screen,
        expected_state_key=goal.expected_state_key,
        expected_state_value=goal.expected_state_value,
    )
    return grammar.resolve(
        fallback,
        observation,
        app_specific_memory=True,
        minimum_confidence=minimum_confidence,
    )


def run_coordinate_baseline(
    macro: CoordinateMacro,
    backend: FleetRuntimeBackend,
    *,
    text_values: Optional[Mapping[str, str]] = None,
    expected_success: bool = True,
    task_success=None,
) -> InstanceRunReceipt:
    """Replay source pixels exactly, matching vendor synchronizer semantics."""

    values = dict(text_values or {})
    width, height = macro.source_resolution
    before_teacher = backend.teacher_reads
    before_injected = backend.actions_injected
    logs = []
    attempted = 0
    for action in macro.actions:
        if action.kind is MacroActionKind.WAIT:
            backend.advance(action.duration_ms)
            logs.append({"sequence": action.sequence, "kind": "wait", "duration_ms": action.duration_ms})
            continue
        attempted += 1
        if action.kind is MacroActionKind.TAP and action.point is not None:
            result = backend.tap_source_pixels(
                int(round(action.point[0] * width)),
                int(round(action.point[1] * height)),
            )
        elif action.kind is MacroActionKind.TEXT:
            result = backend.type_text(values[action.text_ref or ""])
        elif action.kind is MacroActionKind.KEY and str(action.key or "").casefold() in {"back", "escape", "4", "keycode_back"}:
            result = backend.back()
        else:
            result = type("BaselineRefusal", (), {"accepted": False, "injected": False, "reason": "unsupported baseline action"})()
        elapsed = 0.0
        while backend.pending and elapsed < 5000:
            backend.advance(80)
            elapsed += 80
        backend.advance(120)
        logs.append(
            {
                "sequence": action.sequence,
                "kind": action.kind.value,
                "accepted": bool(getattr(result, "accepted", False)),
                "injected": bool(getattr(result, "injected", False)),
                "reason": getattr(result, "reason", None),
                "settlement_ms": elapsed + 120,
            }
        )
    success = bool(task_success()) if task_success is not None else all(row.get("accepted", True) for row in logs)
    return InstanceRunReceipt(
        instance_id=backend.instance_id,
        mode="coordinate_baseline",
        success=success,
        expected_success=expected_success,
        actions_attempted=attempted,
        actions_injected=backend.actions_injected - before_injected,
        committed_actions=sum(1 for row in logs if row.get("injected")),
        duplicate_actions=0,
        pending_overlap_rejections=0,
        teacher_reads_runtime=backend.teacher_reads - before_teacher,
        large_model_calls=0,
        final_screen=backend.current_screen(),
        failure_reason=None if success else "coordinate macro did not reach the task postcondition",
        step_receipts=tuple(logs),
        metadata={"source_resolution": list(macro.source_resolution)},
    )


def run_semantic_procedure(
    procedure: SemanticProcedure,
    backend: FleetRuntimeBackend,
    index: VisualStateIndex,
    *,
    text_values: Optional[Mapping[str, str]] = None,
    expected_success: bool = True,
    task_success=None,
    policy: ReplayPolicy = ReplayPolicy(),
) -> InstanceRunReceipt:
    """Resolve and verify every procedure step independently for one instance."""

    values = dict(text_values or {})
    grammar = PhoneGrammar()
    before_teacher = backend.teacher_reads
    before_injected = backend.actions_injected
    logs = []
    controller = TransactionalController(
        _MotorAdapter(backend),
        lambda context=None: _observation(backend, index, context, policy),
        policy=policy.settlement,
        now_ms=(
            backend.now_ms
            if callable(getattr(backend, "now_ms", None))
            else None
        ),
    )
    failure = None
    current_expected_screen: Optional[str] = procedure.initial_screen
    for step in procedure.steps:
        if step.operator is ProcedureOperator.WAIT:
            backend.advance(float(step.evidence.get("demonstration_wait_ms") or 0.0))
            continue
        target = step.target
        observation = _observation(
            backend,
            index,
            ObservationContext(
                stage="pre_action",
                expected_screen=current_expected_screen,
                target_label=(target.label if target is not None else None),
                target_role=(target.role if target is not None else None),
            ),
            policy,
        )
        if observation.unknown:
            failure = f"unknown surface before semantic step {step.sequence}"
            logs.append({"sequence": step.sequence, "status": "unknown", "observation": observation.to_dict()})
            break
        value = values.get(step.value_ref or "") if step.value_ref else None
        resolution = _resolve_with_role_fallback(
            grammar,
            step,
            observation,
            value=value,
            minimum_confidence=policy.minimum_semantic_confidence,
        )
        if not resolution.resolved or resolution.action is None:
            failure = resolution.reason
            logs.append({"sequence": step.sequence, "status": "unresolved", "reason": resolution.reason})
            break
        action = resolution.action
        if step.operator is ProcedureOperator.SCROLL:
            result = backend.swipe_normalized(step.swipe_path, float(step.evidence.get("duration_ms") or 300.0))
            logs.append(
                {
                    "sequence": step.sequence,
                    "status": "gesture",
                    "accepted": bool(getattr(result, "accepted", False)),
                    "injected": bool(getattr(result, "injected", False)),
                    "reason": getattr(result, "reason", None),
                }
            )
            if not getattr(result, "injected", False):
                failure = getattr(result, "reason", "scroll refused")
                break
            continue
        if step.operator is ProcedureOperator.LONG_PRESS:
            point = action.normalized_point
            if point is None:
                failure = "long press target has no point"
                break
            result = backend.long_press_normalized(*point, float(step.evidence.get("duration_ms") or 700.0))
            logs.append(
                {
                    "sequence": step.sequence,
                    "status": "long_press",
                    "accepted": bool(getattr(result, "accepted", False)),
                    "injected": bool(getattr(result, "injected", False)),
                }
            )
            if not getattr(result, "injected", False):
                failure = getattr(result, "reason", "long press refused")
                break
            continue
        try:
            receipt = controller.execute(
                action,
                idempotency_key=f"{procedure.procedure_id}:{backend.instance_id}:{step.sequence}",
            )
        except PendingActionError as exc:
            failure = str(exc)
            logs.append({"sequence": step.sequence, "status": "pending_overlap", "reason": failure})
            break
        logs.append(
            {
                "sequence": step.sequence,
                "resolution": {
                    "reason": resolution.reason,
                    "generic_transfer": resolution.generic_transfer,
                    "candidates_considered": resolution.candidates_considered,
                },
                "controller_receipt": receipt.to_dict(),
            }
        )
        if not receipt.committed:
            failure = receipt.reason
            break
        current_expected_screen = step.expected_screen or current_expected_screen
    success = bool(task_success()) if task_success is not None and failure is None else failure is None
    return InstanceRunReceipt(
        instance_id=backend.instance_id,
        mode="semantic_replay",
        success=success,
        expected_success=expected_success,
        actions_attempted=len([row for row in procedure.steps if row.operator is not ProcedureOperator.WAIT]),
        actions_injected=backend.actions_injected - before_injected,
        committed_actions=controller.counters.postcondition_successes,
        duplicate_actions=controller.counters.duplicate_requests,
        pending_overlap_rejections=controller.counters.pending_overlap_rejections,
        teacher_reads_runtime=backend.teacher_reads - before_teacher,
        large_model_calls=0,
        final_screen=backend.current_screen(),
        failure_reason=None if success else (failure or "semantic replay did not reach task postcondition"),
        step_receipts=tuple(logs),
        metadata={
            "visual_index_families": index.family_count,
            "visual_index_variants": len(index.variants),
        },
    )
