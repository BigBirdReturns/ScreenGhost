from core.contracts import CanonicalAction, CanonicalTarget, RunPlan, RunStep
from core.executor import FastHandsExecutor
from core.policy import SafetyPolicy
from core.skills import (
    MAX_CONSECUTIVE_FAILURES,
    SemanticSkill,
    SkillStore,
    Waypoint,
    intent_key_for_step,
)


def _step(action="toggle", name="Dark Mode", app="Settings"):
    target = CanonicalTarget(domain="display", location="Settings", name=name)
    return RunStep(action=CanonicalAction(action=action, target=target), app_hint=app)


def _skill(step=None):
    step = step or _step()
    return SemanticSkill(
        intent_key=intent_key_for_step(step),
        app=step.app_hint,
        waypoints=[
            Waypoint(action="tap", label="Display", expect_screen="Display"),
            Waypoint(action="tap", label="Dark Mode"),
        ],
    )


class FakeState:
    def __init__(self, app="Settings", screen="Display"):
        self.app = app
        self.screen = screen

    def get_element(self, label):
        return None


def test_skills_are_semantic_not_positional():
    """The contract: waypoints anchor to labels, never to coordinates."""
    skill = _skill()
    for wp in skill.waypoints:
        assert not hasattr(wp, "x") and not hasattr(wp, "y")
        assert wp.label is not None


def test_store_roundtrip(tmp_path):
    path = tmp_path / "skills.json"
    store = SkillStore(path)
    store.save_skill(_skill())

    reloaded = SkillStore(path)
    assert len(reloaded) == 1
    found = reloaded.lookup(_step())
    assert found is not None
    assert found.waypoints[0].label == "Display"
    assert found.waypoints[0].expect_screen == "Display"


def test_stale_skill_is_forgotten_after_consecutive_failures(tmp_path):
    store = SkillStore(tmp_path / "skills.json")
    skill = _skill()
    store.save_skill(skill)

    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        store.record_failure(skill)
    assert store.lookup(_step()) is not None, "not stale yet"

    # A success resets the counter — flaky screens don't kill good skills.
    store.record_success(skill)
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        store.record_failure(skill)
    assert store.lookup(_step()) is not None

    store.record_failure(skill)
    assert store.lookup(_step()) is None, "stale skill must be forgotten"


def test_executor_prefers_skill_and_skips_vlm(tmp_path):
    store = SkillStore(tmp_path / "skills.json")
    store.save_skill(_skill())

    followed, vlm_calls = [], []
    executor = FastHandsExecutor(
        SafetyPolicy(),
        delay=0,
        skill_store=store,
        observer=lambda device, driver: FakeState(),
        act_step=lambda *a: vlm_calls.append(a),
        is_goal_done=lambda *a: True,
        follow_waypoint=lambda wp, device, driver: followed.append(wp.label) or True,
    )
    result = executor.run(RunPlan(intent="dark mode", steps=[_step()]))

    assert result.completed
    assert followed == ["Display", "Dark Mode"], "skill fast-path not taken"
    assert vlm_calls == [], "VLM should not run when the skill works"
    assert store.lookup(_step()).successes == 1


def test_stale_skill_falls_back_to_live_derivation(tmp_path):
    store = SkillStore(tmp_path / "skills.json")
    store.save_skill(_skill())

    vlm_calls = []
    done_seq = iter([False, True])  # not done once (VLM acts), then done
    executor = FastHandsExecutor(
        SafetyPolicy(),
        delay=0,
        skill_store=store,
        observer=lambda device, driver: FakeState(),
        act_step=lambda *a: vlm_calls.append(a),
        is_goal_done=lambda *a: next(done_seq),
        follow_waypoint=lambda wp, device, driver: False,  # UI churned: label gone
    )
    result = executor.run(RunPlan(intent="dark mode", steps=[_step()]))

    assert result.completed, "churn must cost speed, not capability"
    assert len(vlm_calls) == 1, "live derivation must take over"
    assert store.lookup(_step()).consecutive_failures == 1
