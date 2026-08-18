"""Debounced, fail-closed DTI round phase custody."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from experiments.dti.schema import DTIObservation, RoundPhase

@dataclass(frozen=True)
class PhaseTransition:
    previous: RoundPhase
    observed: RoundPhase
    current: RoundPhase
    accepted: bool
    reason: str
    unknown_streak: int


class DTIRoundTracker:
    """Debounced, fail-closed round state machine."""

    ALLOWED = {
        RoundPhase.UNKNOWN: frozenset(
            {
                RoundPhase.LOBBY,
                RoundPhase.FREEPLAY,
                RoundPhase.THEME_BRIEF,
                RoundPhase.DRESSING,
                RoundPhase.RECOVERY,
                RoundPhase.HALTED,
            }
        ),
        RoundPhase.LOBBY: frozenset(
            {RoundPhase.THEME_BRIEF, RoundPhase.DRESSING, RoundPhase.RECOVERY, RoundPhase.HALTED}
        ),
        RoundPhase.FREEPLAY: frozenset(
            {
                RoundPhase.RUNWAY_READY,
                RoundPhase.RUNWAY,
                RoundPhase.FREEPLAY,
                RoundPhase.RECOVERY,
                RoundPhase.HALTED,
            }
        ),
        RoundPhase.THEME_BRIEF: frozenset(
            {RoundPhase.DRESSING, RoundPhase.RECOVERY, RoundPhase.HALTED}
        ),
        RoundPhase.DRESSING: frozenset(
            {
                RoundPhase.RUNWAY,
                RoundPhase.VOTING,
                RoundPhase.RESULTS,
                RoundPhase.RECOVERY,
                RoundPhase.HALTED,
            }
        ),
        RoundPhase.RUNWAY_READY: frozenset(
            {RoundPhase.RUNWAY, RoundPhase.FREEPLAY, RoundPhase.RECOVERY, RoundPhase.HALTED}
        ),
        RoundPhase.RUNWAY: frozenset(
            {
                RoundPhase.VOTING,
                RoundPhase.RESULTS,
                RoundPhase.FREEPLAY,
                RoundPhase.RECOVERY,
                RoundPhase.HALTED,
            }
        ),
        RoundPhase.VOTING: frozenset(
            {RoundPhase.RESULTS, RoundPhase.RECOVERY, RoundPhase.HALTED}
        ),
        RoundPhase.RESULTS: frozenset(
            {RoundPhase.LOBBY, RoundPhase.FREEPLAY, RoundPhase.RECOVERY, RoundPhase.HALTED}
        ),
        RoundPhase.RECOVERY: frozenset(
            {
                RoundPhase.LOBBY,
                RoundPhase.FREEPLAY,
                RoundPhase.THEME_BRIEF,
                RoundPhase.DRESSING,
                RoundPhase.RUNWAY_READY,
                RoundPhase.RUNWAY,
                RoundPhase.VOTING,
                RoundPhase.RESULTS,
                RoundPhase.HALTED,
            }
        ),
        RoundPhase.HALTED: frozenset({RoundPhase.HALTED}),
    }

    def __init__(self, *, stable_samples: int = 2, unknown_limit: int = 3) -> None:
        if stable_samples < 1:
            raise ValueError("stable_samples must be positive")
        if unknown_limit < 1:
            raise ValueError("unknown_limit must be positive")
        self.stable_samples = int(stable_samples)
        self.unknown_limit = int(unknown_limit)
        self.current = RoundPhase.UNKNOWN
        self._candidate: Optional[RoundPhase] = None
        self._candidate_count = 0
        self.unknown_streak = 0
        self.history: list[PhaseTransition] = []

    def update(self, observation: DTIObservation) -> PhaseTransition:
        previous = self.current
        observed = observation.phase

        if self.current is RoundPhase.HALTED:
            transition = PhaseTransition(
                previous=previous,
                observed=observed,
                current=self.current,
                accepted=False,
                reason="tracker is halted",
                unknown_streak=self.unknown_streak,
            )
            self.history.append(transition)
            return transition

        if observation.unknown or observed is RoundPhase.UNKNOWN:
            self.unknown_streak += 1
            self._candidate = None
            self._candidate_count = 0
            if self.unknown_streak >= self.unknown_limit:
                self.current = RoundPhase.HALTED
                reason = "unknown-screen budget exhausted"
                accepted = True
            else:
                reason = "unknown observation retained without action authority"
                accepted = False
            transition = PhaseTransition(
                previous=previous,
                observed=observed,
                current=self.current,
                accepted=accepted,
                reason=reason,
                unknown_streak=self.unknown_streak,
            )
            self.history.append(transition)
            return transition

        self.unknown_streak = 0
        if observed is self.current:
            self._candidate = None
            self._candidate_count = 0
            transition = PhaseTransition(
                previous=previous,
                observed=observed,
                current=self.current,
                accepted=True,
                reason="phase remained stable",
                unknown_streak=0,
            )
            self.history.append(transition)
            return transition

        if self._candidate is observed:
            self._candidate_count += 1
        else:
            self._candidate = observed
            self._candidate_count = 1

        required = 1 if self.current is RoundPhase.UNKNOWN else self.stable_samples
        if self._candidate_count < required:
            transition = PhaseTransition(
                previous=previous,
                observed=observed,
                current=self.current,
                accepted=False,
                reason=f"phase candidate requires {required} stable samples",
                unknown_streak=0,
            )
            self.history.append(transition)
            return transition

        allowed = observed in self.ALLOWED[self.current]
        if allowed:
            self.current = observed
            reason = "declared transition accepted"
            accepted = True
        else:
            self.current = RoundPhase.RECOVERY
            reason = f"illegal transition {previous.value}->{observed.value}; entered recovery"
            accepted = False
        self._candidate = None
        self._candidate_count = 0
        transition = PhaseTransition(
            previous=previous,
            observed=observed,
            current=self.current,
            accepted=accepted,
            reason=reason,
            unknown_streak=0,
        )
        self.history.append(transition)
        return transition

