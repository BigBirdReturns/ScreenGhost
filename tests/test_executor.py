from dataclasses import dataclass
from typing import Optional

from core.contracts import CanonicalAction, CanonicalTarget, RunPlan, RunStep
from core.executor import FastHandsExecutor, goal_for_action
from core.policy import SafetyPolicy


@dataclass
class FakeElement:
    label: str
    value: Optional[str] = None


class FakeState:
    """Minimal stand-in for ScreenState so tests need no vision model."""

    def __init__(self, app="Home", screen="Main", elements=None):
        self.app = app
        self.screen = screen
        self._elements = {e.label: e for e in (elements or [])}

    def get_element(self, label):
        return self._elements.get(label)


def _step(action="toggle", name="Dark Mode", app="Settings", verify_label=None, verify_value=None):
    target = CanonicalTarget(domain="display", location="Settings", name=name)
    return RunStep(
        action=CanonicalAction(action=action, target=target),
        app_hint=app,
        verify_label=verify_label,
        verify_value=verify_value,
    )


def test_goal_translation():
    assert goal_for_action(_step("open", "Settings").action) == "Open Settings"
    assert goal_for_action(_step("toggle", "Dark Mode").action) == "Toggle Dark Mode in Settings"
    set_action = CanonicalAction("set", CanonicalTarget("hvac", "Home", "thermostat"), value="72")
    assert goal_for_action(set_action) == "Set thermostat to 72"


def test_executor_actually_moves_the_hands():
    """The whole point: run() must drive the action, not just verify it."""
    taps = []

    def fake_act(goal, device, driver):
        taps.append(goal)

    # Goal reads as "not done" once (forcing an action) then "done".
    done_seq = iter([False, True])

    executor = FastHandsExecutor(
        SafetyPolicy(),
        delay=0,
        observer=lambda device, driver: FakeState(app="Settings", screen="Display"),
        act_step=fake_act,
        is_goal_done=lambda goal, device, driver: next(done_seq),
    )

    result = executor.run(RunPlan(intent="dark mode", steps=[_step()]))

    assert result.completed
    assert taps == ["Toggle Dark Mode in Settings"], "hands never moved"
    assert result.steps[0].acted is True


def test_check_action_is_observe_only():
    taps = []
    executor = FastHandsExecutor(
        SafetyPolicy(),
        delay=0,
        observer=lambda device, driver: FakeState(),
        act_step=lambda *a: taps.append(a),
        is_goal_done=lambda *a: True,
    )
    result = executor.run(RunPlan(intent="peek", steps=[_step(action="check")]))
    assert result.completed
    assert taps == []
    assert result.steps[0].acted is False


def test_policy_blocks_before_acting():
    taps = []
    executor = FastHandsExecutor(
        SafetyPolicy(blocked_actions={"factory_reset"}),
        delay=0,
        observer=lambda device, driver: FakeState(),
        act_step=lambda *a: taps.append(a),
        is_goal_done=lambda *a: False,
    )
    result = executor.run(RunPlan(intent="wipe", steps=[_step(action="factory_reset")]))
    assert not result.completed
    assert taps == [], "blocked action must never reach the hands"


def test_verify_value_mismatch_fails_run():
    executor = FastHandsExecutor(
        SafetyPolicy(),
        delay=0,
        observer=lambda device, driver: FakeState(elements=[FakeElement("Dark Mode", "off")]),
        act_step=lambda *a: None,
        is_goal_done=lambda *a: True,
    )
    result = executor.run(
        RunPlan(intent="dark mode", steps=[_step(verify_label="Dark Mode", verify_value="on")])
    )
    assert not result.completed
    assert "mismatch" in result.steps[0].reason
