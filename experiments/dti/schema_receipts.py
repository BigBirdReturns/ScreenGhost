"""Content-addressed action and round receipts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Tuple

from experiments.dti.schema_base import RoundPhase, clean_text, content_id
from experiments.dti.schema_actions import GameAction, RunContext

@dataclass(frozen=True)
class ActionReceipt:
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
    reason: str
    policy_reasons: Tuple[str, ...] = ()
    evidence_tier: str = "visible_surface_action"

    @classmethod
    def build(
        cls,
        *,
        idempotency_key: str,
        action: GameAction,
        status: str,
        committed: bool,
        injected: bool,
        started_ms: float,
        completed_ms: float,
        before_observation_id: str,
        after_observation_id: Optional[str],
        reason: str,
        policy_reasons: Sequence[str] = (),
    ) -> "ActionReceipt":
        settlement_ms = max(0.0, float(completed_ms) - float(started_ms))
        payload = {
            "idempotency_key": idempotency_key,
            "action": action.to_dict(),
            "status": status,
            "committed": bool(committed),
            "injected": bool(injected),
            "started_ms": round(float(started_ms), 3),
            "completed_ms": round(float(completed_ms), 3),
            "before_observation_id": before_observation_id,
            "after_observation_id": after_observation_id,
            "reason": reason,
            "policy_reasons": list(policy_reasons),
        }
        return cls(
            receipt_id=content_id("dtiactionreceipt1", payload),
            idempotency_key=idempotency_key,
            action_id=action.action_id,
            status=status,
            committed=bool(committed),
            injected=bool(injected),
            started_ms=float(started_ms),
            completed_ms=float(completed_ms),
            settlement_ms=settlement_ms,
            before_observation_id=before_observation_id,
            after_observation_id=after_observation_id,
            reason=reason,
            policy_reasons=tuple(policy_reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoundReceipt:
    round_id: str
    venue: str
    control_mode: str
    theme_text: Optional[str]
    theme_card_id: Optional[str]
    plan_id: Optional[str]
    action_receipt_ids: Tuple[str, ...]
    terminal_phase: str
    completed: bool
    placement: Optional[int]
    stars: Optional[int]
    failure_reason: Optional[str]
    evidence_tier: str = "dti_visible_round"

    @classmethod
    def build(
        cls,
        *,
        context: RunContext,
        theme_text: Optional[str],
        theme_card_id: Optional[str],
        plan_id: Optional[str],
        action_receipts: Sequence[ActionReceipt],
        terminal_phase: RoundPhase,
        completed: bool,
        placement: Optional[int] = None,
        stars: Optional[int] = None,
        failure_reason: Optional[str] = None,
    ) -> "RoundReceipt":
        payload = {
            "context": context.to_dict(),
            "theme_text": clean_text(theme_text),
            "theme_card_id": theme_card_id,
            "plan_id": plan_id,
            "action_receipt_ids": [value.receipt_id for value in action_receipts],
            "terminal_phase": terminal_phase.value,
            "completed": bool(completed),
            "placement": placement,
            "stars": stars,
            "failure_reason": clean_text(failure_reason),
        }
        return cls(
            round_id=content_id("dtiround1", payload),
            venue=context.venue.value,
            control_mode=context.mode.value,
            theme_text=clean_text(theme_text),
            theme_card_id=theme_card_id,
            plan_id=plan_id,
            action_receipt_ids=tuple(payload["action_receipt_ids"]),
            terminal_phase=terminal_phase.value,
            completed=bool(completed),
            placement=placement,
            stars=stars,
            failure_reason=clean_text(failure_reason),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
