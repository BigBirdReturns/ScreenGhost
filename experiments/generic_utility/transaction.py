"""Single-flight pending -> settle -> verify transactional controller.

The controller owns no planning authority. It receives one already-resolved,
policy-authorized action, injects it at most once through the supplied motor,
waits for a stable visible postcondition, and commits or aborts with an
inspectable receipt. A second action cannot begin while one is pending.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from experiments.generic_utility.schema import (
    ControllerReceipt,
    EvidenceSource,
    Operator,
    ResolvedAction,
    StudentObservation,
    clean_text,
    sha256_json,
)


class TransactionError(RuntimeError):
    pass


class PendingActionError(TransactionError):
    pass


@runtime_checkable
class TransactionMotor(Protocol):
    def tap_normalized(self, x: float, y: float) -> Any: ...
    def type_text(self, text: str) -> Any: ...
    def back(self) -> Any: ...
    def advance(self, milliseconds: float) -> None: ...


@dataclass(frozen=True)
class ObservationContext:
    """Non-privileged context already known at the action boundary."""

    stage: str
    action_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    expected_screen: Optional[str] = None
    expected_state_key: Optional[str] = None
    expected_state_value: Optional[str] = None
    target_label: Optional[str] = None
    target_role: Optional[str] = None
    previous_screen_key: Optional[str] = None
    previous_app_family: Optional[str] = None


ObservationFn = Callable[..., StudentObservation]
ClockFn = Callable[[], float]


@dataclass(frozen=True)
class SettlementPolicy:
    timeout_ms: float = 5000.0
    poll_interval_ms: float = 80.0
    stable_samples: int = 3
    minimum_stable_ms: float = 160.0
    require_visible_change: bool = True

    def require_valid(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be positive")
        if self.stable_samples < 1:
            raise ValueError("stable_samples must be positive")
        if self.minimum_stable_ms < 0:
            raise ValueError("minimum_stable_ms cannot be negative")


@dataclass
class PendingTransaction:
    action: ResolvedAction
    idempotency_key: str
    before: StudentObservation
    started_ms: float
    injected: bool
    injection_detail: Mapping[str, Any]
    observed_change: bool = False
    stable_signature: Optional[str] = None
    stable_count: int = 0
    stable_since_ms: Optional[float] = None
    observations: list[StudentObservation] = field(default_factory=list)


@dataclass
class ControllerCounters:
    duplicate_requests: int = 0
    pending_overlap_rejections: int = 0
    actions_injected: int = 0
    postcondition_successes: int = 0
    postcondition_failures: int = 0


class TransactionalController:
    def __init__(
        self,
        motor: TransactionMotor,
        observe: ObservationFn,
        *,
        policy: SettlementPolicy = SettlementPolicy(),
        now_ms: Optional[ClockFn] = None,
    ) -> None:
        policy.require_valid()
        self.motor = motor
        self.observe = observe
        self._observe_accepts_context = self._detect_context_support(observe)
        self.policy = policy
        self.now_ms = now_ms or (lambda: time.monotonic() * 1000.0)
        self.pending: Optional[PendingTransaction] = None
        self.receipts: dict[str, ControllerReceipt] = {}
        self.counters = ControllerCounters()

    def begin(
        self, action: ResolvedAction, *, idempotency_key: str
    ) -> PendingTransaction | ControllerReceipt:
        key = clean_text(idempotency_key)
        if not key:
            raise TransactionError("idempotency_key is required")
        if key in self.receipts:
            self.counters.duplicate_requests += 1
            return self.receipts[key]
        if self.pending is not None:
            self.counters.pending_overlap_rejections += 1
            raise PendingActionError(
                f"action {self.pending.action.action_id!r} is still pending; refusing {action.action_id!r}"
            )
        forbidden = {EvidenceSource.TEACHER.value, EvidenceSource.HUMAN.value}
        leaked = forbidden.intersection(action.evidence_sources)
        if leaked:
            raise TransactionError(
                f"action depends on evidence forbidden at this boundary: {sorted(leaked)}"
            )
        before = self._observe(
            ObservationContext(
                stage="pre_action",
                action_id=action.action_id,
                target_label=action.target_label,
                target_role=action.target_role,
            )
        )
        started = self.now_ms()
        detail = self._inject(action)
        injected = bool(getattr(detail, "injected", False))
        if not injected:
            receipt = self._receipt(
                action,
                key,
                status="execution_failed",
                committed=False,
                injected=False,
                started_ms=started,
                completed_ms=self.now_ms(),
                before=before,
                after=None,
                reason=str(getattr(detail, "reason", "motor refused action")),
            )
            self.receipts[key] = receipt
            return receipt
        self.counters.actions_injected += 1
        self.pending = PendingTransaction(
            action=action,
            idempotency_key=key,
            before=before,
            started_ms=started,
            injected=True,
            injection_detail={
                "accepted": bool(getattr(detail, "accepted", True)),
                "target_key": getattr(detail, "target_key", None),
                "target_label": getattr(detail, "target_label", None),
                "reason": getattr(detail, "reason", None),
            },
        )
        return self.pending

    def execute(self, action: ResolvedAction, *, idempotency_key: str) -> ControllerReceipt:
        started = self.begin(action, idempotency_key=idempotency_key)
        if isinstance(started, ControllerReceipt):
            return started
        return self.settle()

    def settle(self) -> ControllerReceipt:
        pending = self.pending
        if pending is None:
            raise TransactionError("no transaction is pending")
        deadline = pending.started_ms + self.policy.timeout_ms
        while self.now_ms() < deadline:
            self.motor.advance(self.policy.poll_interval_ms)
            after = self._observe(self._context(pending, stage="settlement"))
            pending.observations.append(after)
            before_signature = self._observation_signature(pending.before)
            signature = self._observation_signature(after)
            if signature != before_signature:
                pending.observed_change = True
            now = self.now_ms()
            if signature == pending.stable_signature:
                pending.stable_count += 1
            else:
                pending.stable_signature = signature
                pending.stable_count = 1
                pending.stable_since_ms = now
            stable_for = 0.0 if pending.stable_since_ms is None else now - pending.stable_since_ms
            postcondition_ok, reason = self._postcondition(pending.action, after)
            change_gate = pending.observed_change or not self.policy.require_visible_change
            if (
                postcondition_ok
                and change_gate
                and pending.stable_count >= self.policy.stable_samples
                and stable_for >= self.policy.minimum_stable_ms
            ):
                self.counters.postcondition_successes += 1
                receipt = self._receipt(
                    pending.action,
                    pending.idempotency_key,
                    status="verified",
                    committed=True,
                    injected=True,
                    started_ms=pending.started_ms,
                    completed_ms=now,
                    before=pending.before,
                    after=after,
                    reason=reason,
                )
                self.receipts[pending.idempotency_key] = receipt
                self.pending = None
                return receipt
        final = (
            pending.observations[-1]
            if pending.observations
            else self._observe(self._context(pending, stage="final"))
        )
        postcondition_ok, reason = self._postcondition(pending.action, final)
        status = "settlement_timeout" if not postcondition_ok else "stability_timeout"
        self.counters.postcondition_failures += 1
        receipt = self._receipt(
            pending.action,
            pending.idempotency_key,
            status=status,
            committed=False,
            injected=True,
            started_ms=pending.started_ms,
            completed_ms=self.now_ms(),
            before=pending.before,
            after=final,
            reason=reason,
        )
        self.receipts[pending.idempotency_key] = receipt
        self.pending = None
        return receipt

    @staticmethod
    def _detect_context_support(observe: ObservationFn) -> bool:
        try:
            signature = inspect.signature(observe)
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
            for parameter in signature.parameters.values()
        )

    def _observe(self, context: ObservationContext) -> StudentObservation:
        if self._observe_accepts_context:
            return self.observe(context)
        return self.observe()

    @staticmethod
    def _context(pending: PendingTransaction, *, stage: str) -> ObservationContext:
        return ObservationContext(
            stage=stage,
            action_id=pending.action.action_id,
            idempotency_key=pending.idempotency_key,
            expected_screen=pending.action.expected_screen,
            expected_state_key=pending.action.expected_state_key,
            expected_state_value=pending.action.expected_state_value,
            target_label=pending.action.target_label,
            target_role=pending.action.target_role,
            previous_screen_key=pending.before.screen_key,
            previous_app_family=pending.before.app_family,
        )

    def _inject(self, action: ResolvedAction) -> Any:
        if action.operator is Operator.FILL:
            return self.motor.type_text(action.text_value or "")
        if action.operator is Operator.BACK:
            return self.motor.back()
        if action.normalized_point is None:
            raise TransactionError(f"action {action.action_id!r} has no normalized point")
        return self.motor.tap_normalized(*action.normalized_point)

    @staticmethod
    def _observation_signature(observation: StudentObservation) -> str:
        elements = [
            {
                "id": element.element_id,
                "role": element.role,
                "label": element.label,
                "states": dict(element.states),
                "enabled": element.enabled,
            }
            for element in observation.elements
        ]
        return sha256_json(
            {
                "screen_key": observation.screen_key,
                "unknown": observation.unknown,
                "visual_variant": observation.match_detail.get("best_variant_id"),
                "elements": elements,
            }
        )

    @staticmethod
    def _postcondition(action: ResolvedAction, observation: StudentObservation) -> tuple[bool, str]:
        if observation.unknown:
            return False, "postcondition cannot be verified on an unknown screen"
        if action.expected_screen:
            actual = clean_text(observation.match_detail.get("screen_name"))
            if actual is None:
                actual = clean_text(observation.explanation)
            expected_cf = action.expected_screen.casefold()
            haystack = " ".join(
                value for value in (observation.screen_key or "", actual or "") if value
            ).casefold()
            if expected_cf not in haystack:
                return False, f"expected screen {action.expected_screen!r} not visible"
        if action.expected_state_key:
            candidates = []
            if action.target_label:
                element = observation.get_element(action.target_label)
                if element is not None:
                    candidates.append(element)
            if action.target_role:
                candidates.extend(
                    element
                    for element in observation.elements
                    if element.role == action.target_role and element not in candidates
                )
            expected = (action.expected_state_value or "").casefold()
            for element in candidates:
                actual = (element.states.get(action.expected_state_key) or "").casefold()
                if actual == expected:
                    return True, "visible element state matched"
            return False, (
                f"expected {action.expected_state_key}={action.expected_state_value!r} "
                f"for {action.target_label or action.target_role!r}"
            )
        return True, "visible screen postcondition matched"

    def _receipt(
        self,
        action: ResolvedAction,
        idempotency_key: str,
        *,
        status: str,
        committed: bool,
        injected: bool,
        started_ms: float,
        completed_ms: float,
        before: StudentObservation,
        after: Optional[StudentObservation],
        reason: str,
    ) -> ControllerReceipt:
        postcondition = {
            "expected_screen": action.expected_screen,
            "expected_state_key": action.expected_state_key,
            "expected_state_value": action.expected_state_value,
            "target_label": action.target_label,
            "target_role": action.target_role,
        }
        receipt_id = "controller_" + sha256_json(
            {
                "idempotency_key": idempotency_key,
                "action_id": action.action_id,
                "status": status,
                "before": before.observation_id,
                "after": after.observation_id if after else None,
                "postcondition": postcondition,
            }
        )
        return ControllerReceipt(
            receipt_id=receipt_id,
            idempotency_key=idempotency_key,
            action_id=action.action_id,
            status=status,
            committed=committed,
            injected=injected,
            started_ms=started_ms,
            completed_ms=completed_ms,
            settlement_ms=max(0.0, completed_ms - started_ms),
            before_observation_id=before.observation_id,
            after_observation_id=after.observation_id if after else None,
            postcondition=postcondition,
            reason=reason,
        )
