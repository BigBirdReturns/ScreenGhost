from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.contracts import RunPlan, RunStep
from core.policy import SafetyPolicy
from screenghost import ScreenState, observe


@dataclass
class StepResult:
    ok: bool
    reason: str
    app: str
    screen: str


@dataclass
class RunResult:
    completed: bool
    steps: List[StepResult] = field(default_factory=list)


class FastHandsExecutor:
    """Deterministic step runner with policy checks and postcondition verification."""

    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self.policy = policy or SafetyPolicy()

    def _verify(self, state: ScreenState, step: RunStep) -> tuple[bool, str]:
        if step.verify_label is None:
            return True, "no verification required"
        element = state.get_element(step.verify_label)
        if element is None:
            return False, f"verify label not found: {step.verify_label}"
        if step.verify_value is None:
            return True, "verify label found"
        if (element.value or "").lower() == step.verify_value.lower():
            return True, "verify value matched"
        return False, f"verify value mismatch: expected={step.verify_value} got={element.value}"

    def run(self, plan: RunPlan, *, device: Optional[str] = None, driver=None) -> RunResult:
        results: List[StepResult] = []

        for step in plan.steps:
            if not self.policy.allows_app(step.app_hint):
                results.append(StepResult(False, f"blocked by app policy: {step.app_hint}", "unknown", "unknown"))
                return RunResult(False, results)
            if not self.policy.allows_action(step.action.action):
                results.append(StepResult(False, f"blocked by action policy: {step.action.action}", "unknown", "unknown"))
                return RunResult(False, results)

            state = observe(device=device, driver=driver)
            ok, reason = self._verify(state, step)
            results.append(StepResult(ok, reason, state.app, state.screen))
            if not ok:
                return RunResult(False, results)

        return RunResult(True, results)
