from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from core.contracts import CanonicalAction, RunPlan, RunStep
from core.policy import SafetyPolicy


@dataclass
class StepResult:
    ok: bool
    reason: str
    app: str
    screen: str
    acted: bool = False


@dataclass
class RunResult:
    completed: bool
    steps: List[StepResult] = field(default_factory=list)


def goal_for_action(action: CanonicalAction) -> str:
    """Translate a canonical action into a natural-language navigator goal."""
    name = action.target.name
    where = action.target.location
    verb = action.action.lower()

    if verb == "open":
        return f"Open {name}"
    if verb == "toggle":
        return f"Toggle {name} in {where}"
    if verb == "set":
        return f"Set {name} to {action.value}" if action.value else f"Adjust {name}"
    if verb == "check":
        return f"View {name}"
    return f"{action.action} {name}"


class FastHandsExecutor:
    """Deterministic step runner: move the hands, then verify the outcome.

    "Fast hands" only means something if the hands actually move. For every
    step this enforces policy, drives the canonical action into the live UI
    through the device driver, and then confirms the postcondition. A
    ``check`` action is observe-only and skips the acting phase.

    The screen-understanding hooks (``observer``, ``act_step``,
    ``is_goal_done``) are injectable so this module imports and unit-tests
    without the vision model / imaging stack; unset hooks resolve lazily to
    the real Screen Ghost implementations at call time.
    """

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        *,
        execute: bool = True,
        max_steps_per_action: int = 8,
        delay: float = 0.4,
        observer: Optional[Callable[..., Any]] = None,
        act_step: Optional[Callable[[str, Optional[str], Any], None]] = None,
        is_goal_done: Optional[Callable[[str, Optional[str], Any], bool]] = None,
    ):
        self.policy = policy or SafetyPolicy()
        self.execute = execute
        self.max_steps_per_action = max_steps_per_action
        self.delay = delay
        self._observer = observer
        self._act_step = act_step
        self._is_goal_done = is_goal_done

    # -- screen-understanding hooks (lazy real defaults) -----------------

    def _observe(self, device: Optional[str], driver: Any):
        if self._observer is not None:
            return self._observer(device=device, driver=driver)
        from screenghost import observe

        return observe(device=device, driver=driver)

    def _goal_done(self, goal: str, device: Optional[str], driver: Any) -> bool:
        if self._is_goal_done is not None:
            return self._is_goal_done(goal, device, driver)
        from screenghost import check_goal_complete, get_driver

        d = get_driver(driver)
        img = d.screencap(device)
        done, _reason = check_goal_complete(img, goal)
        return done

    def _act_once(self, goal: str, device: Optional[str], driver: Any) -> None:
        if self._act_step is not None:
            self._act_step(goal, device, driver)
            return
        from screenghost import execute_action, get_driver, pick_action

        d = get_driver(driver)
        img = d.screencap(device)
        action = pick_action(img, goal)
        execute_action(action, device, (img.width, img.height), driver=d)

    # -- verification ----------------------------------------------------

    def _verify(self, state, step: RunStep) -> tuple[bool, str]:
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

    # -- acting ----------------------------------------------------------

    def _drive(self, step: RunStep, device: Optional[str], driver: Any) -> bool:
        """Move the hands until the step's goal is reached or steps run out.

        Returns True if at least one action was performed.
        """
        goal = goal_for_action(step.action)
        acted = False
        for _ in range(self.max_steps_per_action):
            if self._goal_done(goal, device, driver):
                break
            self._act_once(goal, device, driver)
            acted = True
            if self.delay:
                time.sleep(self.delay)
        return acted

    # -- run -------------------------------------------------------------

    def run(self, plan: RunPlan, *, device: Optional[str] = None, driver: Any = None) -> RunResult:
        results: List[StepResult] = []

        for step in plan.steps:
            if not self.policy.allows_app(step.app_hint):
                results.append(StepResult(False, f"blocked by app policy: {step.app_hint}", "unknown", "unknown"))
                return RunResult(False, results)
            if not self.policy.allows_action(step.action.action):
                results.append(StepResult(False, f"blocked by action policy: {step.action.action}", "unknown", "unknown"))
                return RunResult(False, results)

            acted = False
            if self.execute and step.action.action.lower() != "check":
                acted = self._drive(step, device, driver)

            state = self._observe(device, driver)
            ok, reason = self._verify(state, step)
            results.append(StepResult(ok, reason, state.app, state.screen, acted=acted))
            if not ok:
                return RunResult(False, results)

        return RunResult(True, results)
