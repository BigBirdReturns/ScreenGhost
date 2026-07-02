from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from core.contracts import CanonicalAction, RunPlan, RunStep
from core.policy import SafetyPolicy
from core.skills import SemanticSkill, SkillStore, Waypoint


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

    An optional ``skill_store`` makes execution adaptive the right way
    around: semantic skills (label-anchored waypoints, resolved against the
    live screen) are tried first as an accelerator; a skill that no longer
    matches reality is demoted and the VLM re-derives the path from the
    current screens. Skills speed things up; they never gate capability.

    The screen-understanding hooks (``observer``, ``act_step``,
    ``is_goal_done``, ``follow_waypoint``) are injectable so this module
    imports and unit-tests without the vision model / imaging stack; unset
    hooks resolve lazily to the real Screen Ghost implementations at call
    time.
    """

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        *,
        execute: bool = True,
        max_steps_per_action: int = 8,
        delay: float = 0.4,
        skill_store: Optional[SkillStore] = None,
        observer: Optional[Callable[..., Any]] = None,
        act_step: Optional[Callable[[str, Optional[str], Any], None]] = None,
        is_goal_done: Optional[Callable[[str, Optional[str], Any], bool]] = None,
        follow_waypoint: Optional[Callable[[Waypoint, Optional[str], Any], bool]] = None,
    ):
        self.policy = policy or SafetyPolicy()
        self.execute = execute
        self.max_steps_per_action = max_steps_per_action
        self.delay = delay
        self.skill_store = skill_store
        self._observer = observer
        self._act_step = act_step
        self._is_goal_done = is_goal_done
        self._follow_waypoint = follow_waypoint

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

    def _resolve_waypoint(self, wp: Waypoint, device: Optional[str], driver: Any) -> bool:
        """Execute one semantic waypoint against the live screen.

        Coordinates are never stored: a tap waypoint names a label, and we
        look up where that label lives *right now*. Returns False when the
        screen no longer matches the skill — the staleness signal.
        """
        if self._follow_waypoint is not None:
            return self._follow_waypoint(wp, device, driver)

        from screenghost import get_driver

        d = get_driver(driver)
        state = self._observe(device, driver)

        if wp.action == "tap":
            element = state.get_element(wp.label) if wp.label else None
            if element is None or element.bounds is None:
                return False
            x1, y1, x2, y2 = element.bounds
            d.tap((x1 + x2) // 2, (y1 + y2) // 2, device)
        elif wp.action == "type":
            d.type_text(wp.value or "", device)
        elif wp.action == "swipe":
            width = state.width or 1080
            height = state.height or 1920
            cx, cy = width // 2, height // 2
            direction = (wp.value or "up").lower()
            deltas = {
                "up": (cx, cy + 300, cx, cy - 300),
                "down": (cx, cy - 300, cx, cy + 300),
                "left": (cx + 300, cy, cx - 300, cy),
                "right": (cx - 300, cy, cx + 300, cy),
            }
            if direction not in deltas:
                return False
            d.swipe(*deltas[direction], device=device)
        elif wp.action == "back":
            d.keyevent(4, device)  # KEYCODE_BACK
        else:
            return False

        if self.delay:
            time.sleep(self.delay)

        if wp.expect_screen:
            after = self._observe(device, driver)
            if wp.expect_screen.lower() not in (after.screen or "").lower():
                return False
        return True

    def _follow_skill(self, skill: SemanticSkill, device: Optional[str], driver: Any) -> bool:
        for wp in skill.waypoints:
            if not self._resolve_waypoint(wp, device, driver):
                return False
        return True

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

        A known semantic skill is tried first (fast path). If it no longer
        matches the live UI it is demoted and we fall back to VLM
        derivation — churn costs speed, never capability. Returns True if
        at least one action was performed.
        """
        goal = goal_for_action(step.action)
        acted = False

        if self.skill_store is not None:
            skill = self.skill_store.lookup(step)
            if skill is not None:
                followed = self._follow_skill(skill, device, driver)
                acted = acted or followed
                if followed and self._goal_done(goal, device, driver):
                    self.skill_store.record_success(skill)
                    return True
                self.skill_store.record_failure(skill)

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
