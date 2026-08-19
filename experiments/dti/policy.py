"""Structural policy for the DTI cartridge.

The policy makes the intended family co-play boundary executable.  It rejects
client modification, emulation, process access, multi-account operation, farming,
autonomous public play, automated voting, and unattended operation before the
motor receives an action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from experiments.dti.schema import (
    ControlMode,
    GameAction,
    GameActionKind,
    PlayVenue,
    RunContext,
)


class DTIPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reasons: Tuple[str, ...]
    warnings: Tuple[str, ...] = ()

    def require_allowed(self) -> None:
        if not self.allowed:
            raise DTIPolicyError("; ".join(self.reasons) or "DTI policy refused the action")


class DTISafetyPolicy:
    """Fail-closed policy for ordinary visible-surface co-play."""

    SAFE_KEYS = frozenset(
        {
            "w",
            "a",
            "s",
            "d",
            "q",
            "e",
            "space",
            "shift",
            "ctrl",
            "tab",
            "escape",
            "enter",
            "up",
            "down",
            "left",
            "right",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
        }
    )
    FORBIDDEN_PURPOSES = frozenset(
        {
            "vote",
            "voting",
            "chat",
            "afk",
            "farm",
            "currency",
            "reward",
            "rejoin",
            "captcha",
        }
    )
    FORBIDDEN_PURPOSE_PHRASES = frozenset(
        {
            "anti_idle",
            "auto_rejoin",
            "claim_reward",
            "collect_currency",
            "give_star",
            "give_stars",
            "multi_account",
            "open_chat",
            "rate_player",
            "rating_player",
            "redeem_code",
            "send_message",
            "submit_vote",
        }
    )

    def evaluate(self, context: RunContext, action: GameAction | None = None) -> PolicyDecision:
        reasons: list[str] = []
        warnings: list[str] = []

        if not context.official_client:
            reasons.append("official unmodified Roblox client is required")
        if context.emulator:
            reasons.append("third-party emulator execution is refused")
        if context.modified_client:
            reasons.append("modified Roblox clients are refused")
        if context.process_injection:
            reasons.append("process injection is refused")
        if context.process_memory:
            reasons.append("process-memory reads or writes are refused")
        if context.multi_account:
            reasons.append("multi-account operation is refused")
        if context.reward_farming:
            reasons.append("reward, currency, or AFK farming is refused")
        if context.autonomous_voting:
            reasons.append("automated voting is refused")
        if not context.human_present:
            reasons.append("a human must remain present for the live cartridge")

        if context.venue is PlayVenue.PUBLIC_SERVER and context.mode is ControlMode.AUTONOMOUS:
            reasons.append("autonomous public-server play is refused")
        if context.venue is PlayVenue.PUBLIC_SERVER:
            warnings.append("public-server mode is restricted to bounded co-pilot assistance")
        if context.venue is PlayVenue.FREEPLAY:
            warnings.append("Freeplay is the preferred teaching and autonomous practice venue")

        if action is not None:
            if action.kind in {GameActionKind.PRESS, GameActionKind.HOLD}:
                unsafe = sorted(set(action.keys) - self.SAFE_KEYS)
                if unsafe:
                    reasons.append(f"keys outside the DTI allowlist are refused: {unsafe}")
            purpose_values: Sequence[str] = (
                action.semantic_label,
                str(action.metadata.get("purpose", "")),
                str(action.metadata.get("category", "")),
            )
            normalized_purposes = tuple(
                value.casefold().replace("-", "_").replace("/", "_").replace(" ", "_")
                for value in purpose_values
            )
            purpose_tokens = {
                token
                for value in normalized_purposes
                for token in value.split("_")
                if token
            }
            forbidden_tokens = sorted(
                purpose_tokens.intersection(self.FORBIDDEN_PURPOSES)
            )
            forbidden_phrases = sorted(
                phrase
                for phrase in self.FORBIDDEN_PURPOSE_PHRASES
                if any(phrase in value for value in normalized_purposes)
            )
            forbidden = tuple(dict.fromkeys((*forbidden_tokens, *forbidden_phrases)))
            if forbidden:
                reasons.append(f"action purpose is outside the co-play boundary: {list(forbidden)}")

        return PolicyDecision(allowed=not reasons, reasons=tuple(reasons), warnings=tuple(warnings))

    def require(self, context: RunContext, action: GameAction | None = None) -> PolicyDecision:
        decision = self.evaluate(context, action)
        decision.require_allowed()
        return decision
