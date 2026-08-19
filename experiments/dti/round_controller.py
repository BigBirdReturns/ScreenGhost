"""Single-flight visible game action settlement."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from experiments.dti.policy import DTISafetyPolicy, PolicyDecision
from experiments.dti.schema import (
    ActionReceipt, DTIObservation, GameAction, GameActionKind, RunContext, clean_text,
)

class ControllerError(RuntimeError):
    pass


class PendingActionError(ControllerError):
    pass


@runtime_checkable
class GameMotor(Protocol):
    def press(self, keys: Sequence[str]) -> None: ...
    def hold(self, keys: Sequence[str], duration_ms: int) -> None: ...
    def click_normalized(self, x: float, y: float) -> None: ...
    def move_mouse(self, dx: int, dy: int) -> None: ...
    def scroll(self, delta: int) -> None: ...
    def wait(self, duration_ms: int) -> None: ...


@dataclass(frozen=True)
class SettlementPolicy:
    timeout_ms: int = 5000
    poll_interval_ms: int = 100
    stable_samples: int = 2
    minimum_stable_ms: int = 150

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0 or self.poll_interval_ms <= 0:
            raise ValueError("settlement times must be positive")
        if self.stable_samples < 1 or self.minimum_stable_ms < 0:
            raise ValueError("invalid settlement stability policy")


@dataclass
class PendingAction:
    context: RunContext
    action: GameAction
    idempotency_key: str
    before: DTIObservation
    started_ms: float
    policy_decision: PolicyDecision
    injected: bool
    observed_change: bool = False
    stable_signature: Optional[str] = None
    stable_count: int = 0
    stable_since_ms: Optional[float] = None
    observations: list[DTIObservation] = field(default_factory=list)


ObserveFn = Callable[[], DTIObservation]
ClockFn = Callable[[], float]


class BoundedGameController:
    """One visible game action at a time, with idempotency and postconditions."""

    def __init__(
        self,
        motor: GameMotor,
        observe: ObserveFn,
        *,
        safety: Optional[DTISafetyPolicy] = None,
        settlement: SettlementPolicy = SettlementPolicy(),
        now_ms: Optional[ClockFn] = None,
    ) -> None:
        self.motor = motor
        self.observe = observe
        self.safety = safety or DTISafetyPolicy()
        self.settlement = settlement
        self.now_ms = now_ms or (lambda: time.monotonic() * 1000.0)
        self.pending: Optional[PendingAction] = None
        self.receipts: dict[str, ActionReceipt] = {}
        self.actions_injected = 0
        self.duplicate_requests = 0
        self.pending_overlap_rejections = 0

    def begin(
        self,
        context: RunContext,
        action: GameAction,
        *,
        idempotency_key: str,
    ) -> PendingAction | ActionReceipt:
        key = clean_text(idempotency_key)
        if not key:
            raise ControllerError("idempotency_key is required")
        if key in self.receipts:
            self.duplicate_requests += 1
            return self.receipts[key]
        if self.pending is not None:
            self.pending_overlap_rejections += 1
            raise PendingActionError(
                f"action {self.pending.action.action_id!r} is still pending"
            )

        decision = self.safety.evaluate(context, action)
        before = self.observe()
        started = self.now_ms()
        if not decision.allowed:
            receipt = ActionReceipt.build(
                idempotency_key=key,
                action=action,
                status="policy_blocked",
                committed=False,
                injected=False,
                started_ms=started,
                completed_ms=self.now_ms(),
                before_observation_id=before.observation_id,
                after_observation_id=None,
                reason="; ".join(decision.reasons),
                policy_reasons=decision.reasons,
            )
            self.receipts[key] = receipt
            return receipt

        try:
            self._inject(action)
        except Exception as exc:
            receipt = ActionReceipt.build(
                idempotency_key=key,
                action=action,
                status="execution_failed",
                committed=False,
                injected=False,
                started_ms=started,
                completed_ms=self.now_ms(),
                before_observation_id=before.observation_id,
                after_observation_id=None,
                reason=str(exc),
                policy_reasons=decision.reasons,
            )
            self.receipts[key] = receipt
            return receipt

        self.actions_injected += 1
        self.pending = PendingAction(
            context=context,
            action=action,
            idempotency_key=key,
            before=before,
            started_ms=started,
            policy_decision=decision,
            injected=True,
        )
        return self.pending

    def execute(
        self,
        context: RunContext,
        action: GameAction,
        *,
        idempotency_key: str,
    ) -> ActionReceipt:
        result = self.begin(context, action, idempotency_key=idempotency_key)
        if isinstance(result, ActionReceipt):
            return result
        return self.settle()

    def settle(self) -> ActionReceipt:
        pending = self.pending
        if pending is None:
            raise ControllerError("no action is pending")
        deadline = pending.started_ms + self.settlement.timeout_ms

        while self.now_ms() < deadline:
            self.motor.wait(self.settlement.poll_interval_ms)
            after = self.observe()
            pending.observations.append(after)
            if after.semantic_signature != pending.before.semantic_signature:
                pending.observed_change = True

            now = self.now_ms()
            if after.semantic_signature == pending.stable_signature:
                pending.stable_count += 1
            else:
                pending.stable_signature = after.semantic_signature
                pending.stable_count = 1
                pending.stable_since_ms = now
            stable_for = (
                0.0
                if pending.stable_since_ms is None
                else now - pending.stable_since_ms
            )
            postcondition_ok, reason = self._postcondition(pending.action, after)
            change_ok = pending.observed_change or not pending.action.require_visible_change
            if (
                postcondition_ok
                and change_ok
                and pending.stable_count >= self.settlement.stable_samples
                and stable_for >= self.settlement.minimum_stable_ms
            ):
                receipt = ActionReceipt.build(
                    idempotency_key=pending.idempotency_key,
                    action=pending.action,
                    status="verified",
                    committed=True,
                    injected=True,
                    started_ms=pending.started_ms,
                    completed_ms=now,
                    before_observation_id=pending.before.observation_id,
                    after_observation_id=after.observation_id,
                    reason=reason,
                    policy_reasons=pending.policy_decision.reasons,
                )
                self.receipts[pending.idempotency_key] = receipt
                self.pending = None
                return receipt

        final = pending.observations[-1] if pending.observations else self.observe()
        postcondition_ok, reason = self._postcondition(pending.action, final)
        status = "stability_timeout" if postcondition_ok else "postcondition_timeout"
        receipt = ActionReceipt.build(
            idempotency_key=pending.idempotency_key,
            action=pending.action,
            status=status,
            committed=False,
            injected=True,
            started_ms=pending.started_ms,
            completed_ms=self.now_ms(),
            before_observation_id=pending.before.observation_id,
            after_observation_id=final.observation_id,
            reason=reason,
            policy_reasons=pending.policy_decision.reasons,
        )
        self.receipts[pending.idempotency_key] = receipt
        self.pending = None
        return receipt

    def _inject(self, action: GameAction) -> None:
        if action.kind is GameActionKind.PRESS:
            self.motor.press(action.keys)
        elif action.kind is GameActionKind.HOLD:
            self.motor.hold(action.keys, action.duration_ms)
        elif action.kind is GameActionKind.CLICK:
            assert action.point is not None
            self.motor.click_normalized(*action.point)
        elif action.kind is GameActionKind.MOVE_MOUSE:
            assert action.delta is not None
            self.motor.move_mouse(*action.delta)
        elif action.kind is GameActionKind.SCROLL:
            self.motor.scroll(action.wheel_delta)
        elif action.kind is GameActionKind.WAIT:
            self.motor.wait(action.duration_ms)
        else:  # pragma: no cover - Enum exhaustiveness guard
            raise ControllerError(f"unsupported game action kind: {action.kind}")

    @staticmethod
    def _postcondition(action: GameAction, observation: DTIObservation) -> tuple[bool, str]:
        if observation.unknown:
            return False, "postcondition cannot be verified on an unknown screen"
        if action.expected_phase is not None and observation.phase is not action.expected_phase:
            return False, (
                f"expected phase {action.expected_phase.value!r}, "
                f"observed {observation.phase.value!r}"
            )
        if action.expected_label is not None and not observation.has_label(action.expected_label):
            return False, f"expected visible label {action.expected_label!r} was absent"
        return True, "visible postcondition verified"
