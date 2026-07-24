"""Generic phone-grammar operators and semantic target resolution."""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from experiments.generic_utility.schema import (
    EvidenceSource,
    Operator,
    ResolvedAction,
    SemanticGoal,
    StudentObservation,
    VisibleElement,
    clean_text,
    sha256_json,
)


GRAMMAR_VERSION = "phone_grammar_v1"


class GrammarResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class GrammarResolution:
    resolved: bool
    action: Optional[ResolvedAction]
    reason: str
    generic_transfer: bool
    candidates_considered: int


_ROLE_COMPATIBILITY = {
    Operator.OPEN: {"button", "icon_button", "menu_item", "tab", "link", "list_item"},
    Operator.ACTIVATE: {"button", "icon_button", "menu_item", "tab", "link"},
    Operator.TOGGLE: {"switch", "checkbox", "radio", "toggle"},
    Operator.FILL: {"text_field", "input", "textbox", "searchbox"},
    Operator.START: {"button", "icon_button"},
    Operator.STOP: {"button", "icon_button"},
    Operator.BACK: {"navigation", "icon_button", "button"},
    Operator.CHECK: set(),
}


def _center(bounds: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, bounds)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _label_score(wanted: Optional[str], actual: Optional[str]) -> float:
    if not wanted:
        return 0.5 if actual is None else 0.55
    if not actual:
        return 0.0
    wanted_cf = wanted.casefold().strip()
    actual_cf = actual.casefold().strip()
    if wanted_cf == actual_cf:
        return 1.0
    if wanted_cf in actual_cf or actual_cf in wanted_cf:
        return 0.88
    return difflib.SequenceMatcher(a=wanted_cf, b=actual_cf).ratio()


class PhoneGrammar:
    """Resolves normalized goals against visible semantic elements.

    Coordinates are resolved from the current observation and are never cached.
    Cross-app transfer is permitted only when the role/operator relation is
    unambiguous enough to act safely.
    """

    version = GRAMMAR_VERSION

    def resolve(
        self,
        goal: SemanticGoal,
        observation: StudentObservation,
        *,
        app_specific_memory: bool,
        minimum_confidence: float = 0.80,
    ) -> GrammarResolution:
        if goal.operator is Operator.BACK:
            action = ResolvedAction(
                action_id="action_" + sha256_json({"goal": goal.goal_id, "operator": "back"}),
                operator=goal.operator,
                target_element_id=None,
                target_role="navigation",
                target_label="Back",
                normalized_point=None,
                text_value=None,
                expected_screen=goal.expected_screen,
                expected_state_key=goal.expected_state_key,
                expected_state_value=goal.expected_state_value,
                confidence=1.0,
                evidence_sources=(EvidenceSource.PHONE_GRAMMAR.value,),
            )
            return GrammarResolution(True, action, "system back operator", not app_specific_memory, 0)

        compatible_roles = _ROLE_COMPATIBILITY[goal.operator]
        candidates = [
            element
            for element in observation.elements
            if element.interactive
            and element.enabled
            and (not compatible_roles or element.role in compatible_roles)
        ]
        if not candidates:
            return GrammarResolution(False, None, "no compatible enabled element", False, 0)

        scored: list[tuple[float, VisibleElement]] = []
        for element in candidates:
            role_score = 1.0 if not goal.target_role or element.role == goal.target_role else 0.55
            label_score = _label_score(goal.target_label, element.label)
            # A unique role-only target is a legitimate generic transfer.  Multiple
            # anonymous candidates are ambiguous and must not be guessed.
            if goal.target_label is None and len(candidates) == 1:
                label_score = 0.90
            state_bonus = 0.0
            if goal.expected_state_key and goal.expected_state_key in element.states:
                current = element.states[goal.expected_state_key].casefold()
                desired = (goal.expected_state_value or "").casefold()
                if desired and current == desired:
                    # Already complete.  Keep it a strong candidate so the caller
                    # can decide not to inject another action.
                    state_bonus = 0.04
            score = min(1.0, 0.42 * role_score + 0.58 * label_score + state_bonus)
            scored.append((score, element))
        scored.sort(key=lambda row: (-row[0], row[1].element_id))
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        required_margin = 0.08 if len(scored) > 1 else 0.0
        if best_score < minimum_confidence or margin < required_margin:
            return GrammarResolution(
                False,
                None,
                f"semantic target ambiguous: best={best_score:.3f}, margin={margin:.3f}",
                False,
                len(scored),
            )

        generic_transfer = not app_specific_memory
        evidence = [EvidenceSource.PHONE_GRAMMAR.value]
        evidence.append(
            EvidenceSource.APP_GRAPH.value
            if app_specific_memory
            else EvidenceSource.SMALL_GROUNDER.value
        )
        point = None if goal.operator is Operator.FILL else _center(best.normalized_bounds)
        action = ResolvedAction(
            action_id="action_" + sha256_json(
                {
                    "goal": goal.goal_id,
                    "operator": goal.operator.value,
                    "element_id": best.element_id,
                    "screen": observation.screen_key,
                }
            ),
            operator=goal.operator,
            target_element_id=best.element_id,
            target_role=best.role,
            target_label=best.label,
            normalized_point=point,
            text_value=goal.value,
            expected_screen=goal.expected_screen,
            expected_state_key=goal.expected_state_key,
            expected_state_value=goal.expected_state_value,
            confidence=round(best_score, 6),
            evidence_sources=tuple(evidence),
        )
        return GrammarResolution(
            True,
            action,
            "resolved from current semantic elements",
            generic_transfer,
            len(scored),
        )
