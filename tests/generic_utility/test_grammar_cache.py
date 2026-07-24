from __future__ import annotations

from experiments.generic_utility.decision_cache import SemanticDecisionCache
from experiments.generic_utility.generic_detector import SyntheticPixelDetector
from experiments.generic_utility.phone_grammar import PhoneGrammar
from experiments.generic_utility.phone_world import PhoneWorld
from experiments.generic_utility.schema import Operator, SemanticGoal


def test_generic_detector_finds_holdout_switch():
    world = PhoneWorld(start_app="connectivity", start_screen="home")
    observation, receipt = SyntheticPixelDetector().detect(world.capture_png())
    assert not observation.unknown
    assert receipt.elements_found >= 1
    assert any(element.role == "switch" for element in observation.elements)


def test_generic_phone_grammar_transfers_by_unique_role():
    world = PhoneWorld(start_app="connectivity", start_screen="home")
    observation, _ = SyntheticPixelDetector().detect(world.capture_png())
    goal = SemanticGoal(
        "toggle-connectivity",
        Operator.TOGGLE,
        target_role="switch",
        expected_state_key="checked",
        expected_state_value="true",
    )
    resolution = PhoneGrammar().resolve(goal, observation, app_specific_memory=False)
    assert resolution.resolved and resolution.generic_transfer and resolution.action is not None


def test_phone_grammar_refuses_ambiguous_role():
    from experiments.generic_utility.schema import StudentObservation, VisibleElement
    observation = StudentObservation(
        "ambiguous", "screen", "surface", "app", 1.0, False,
        (
            VisibleElement("a", "button", "A", (0.1, 0.1, 0.3, 0.2), True),
            VisibleElement("b", "button", "B", (0.6, 0.1, 0.8, 0.2), True),
        ),
        ("pixels",),
    )
    goal = SemanticGoal("ambiguous", Operator.ACTIVATE, target_role="button")
    result = PhoneGrammar().resolve(goal, observation, app_specific_memory=False)
    assert not result.resolved


def test_decision_cache_stores_semantics_not_coordinates(tmp_path):
    cache = SemanticDecisionCache(tmp_path / "cache.json")
    world = PhoneWorld(start_app="settings", start_screen="home")
    frame = world.teacher_snapshot()
    from experiments.generic_utility.campaign import EmulatedCampaignRunner
    observation = EmulatedCampaignRunner()._teacher_observation(frame.runtime_projection)
    goal = PhoneWorld.task_catalog()["settings_dark_mode"].goals[0]
    action = PhoneGrammar().resolve(goal, observation, app_specific_memory=True).action
    key = cache.make_key(grammar_version="v1", app_family="settings", screen_family=observation.screen_key, goal=goal, app_version="1.0")
    cache.store(key, action)
    text = (tmp_path / "cache.json").read_text()
    assert "normalized_point" not in text
    assert cache.lookup(key) is not None


def test_cache_failure_demotes_after_threshold(tmp_path):
    cache = SemanticDecisionCache(tmp_path / "cache.json", failure_limit=2)
    world = PhoneWorld(start_app="settings", start_screen="home")
    frame = world.teacher_snapshot()
    from experiments.generic_utility.campaign import EmulatedCampaignRunner
    observation = EmulatedCampaignRunner()._teacher_observation(frame.runtime_projection)
    goal = PhoneWorld.task_catalog()["settings_dark_mode"].goals[0]
    action = PhoneGrammar().resolve(goal, observation, app_specific_memory=True).action
    key = cache.make_key(grammar_version="v1", app_family="settings", screen_family=observation.screen_key, goal=goal, app_version="1.0")
    cache.store(key, action)
    cache.record_failure(key)
    assert cache.lookup(key) is not None
    cache.record_failure(key)
    assert cache.lookup(key) is None
