"""Thin DTI domain cartridge over ScreenGhost's perception and motor floors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from experiments.dti.profile import DTIProfile
from experiments.dti.rounds import DTIRoundTracker, PhaseClassifier, RoundSignals
from experiments.dti.schema import DTIObservation, OutfitPlan
from experiments.dti.theme_kernel import OutfitPlanner, ThemeCatalog, ThemeResolution, seed_theme_cards


@dataclass(frozen=True)
class Interpretation:
    observation: DTIObservation
    theme: Optional[ThemeResolution]
    plan: Optional[OutfitPlan]
    reason: str


class DTICartridge:
    """Interpret visible round state and prepare a feasible local-atlas plan.

    The cartridge does not execute actions on its own.  It emits observations and
    plans to the bounded controller, which remains the sole motor authority.
    """

    def __init__(
        self,
        profile: DTIProfile,
        *,
        themes: Optional[ThemeCatalog] = None,
        planner: Optional[OutfitPlanner] = None,
        tracker: Optional[DTIRoundTracker] = None,
    ) -> None:
        self.profile = profile
        self.themes = themes or ThemeCatalog(seed_theme_cards())
        self.planner = planner or OutfitPlanner()
        self.classifier = PhaseClassifier()
        self.tracker = tracker or DTIRoundTracker()

    def interpret(self, signals: RoundSignals) -> Interpretation:
        observation = self.classifier.classify(signals)
        transition = self.tracker.update(observation)
        if observation.theme_text is None:
            return Interpretation(
                observation=observation,
                theme=None,
                plan=None,
                reason=transition.reason,
            )
        resolution = self.themes.resolve(observation.theme_text)
        if not resolution.resolved or resolution.card is None:
            return Interpretation(
                observation=observation,
                theme=resolution,
                plan=None,
                reason=resolution.reason,
            )
        plan = self.planner.plan(resolution.card, self.profile.wardrobe)
        return Interpretation(
            observation=observation,
            theme=resolution,
            plan=plan,
            reason=transition.reason,
        )
