from __future__ import annotations

import pytest
from PIL import Image
import io

from experiments.generic_utility.phone_world import DisplayVariant, PhoneWorld
from experiments.generic_utility.schema import Operator, SemanticGoal, StudentObservation, VisibleElement


def test_semantic_goal_requires_target_for_non_back():
    with pytest.raises(ValueError):
        SemanticGoal("bad", Operator.OPEN)
    SemanticGoal("back", Operator.BACK)


def test_visible_element_rejects_bad_bounds():
    with pytest.raises(ValueError):
        VisibleElement("x", "button", "X", (0.5, 0.5, 0.4, 0.9))


def test_unknown_observation_cannot_claim_screen_key():
    with pytest.raises(ValueError):
        StudentObservation("obs", "screen", None, None, 0.2, True, (), ("pixels",))


def test_phoneworld_png_and_teacher_plane_are_separate():
    world = PhoneWorld()
    png = world.capture_png()
    assert world.pixel_reads == 1 and world.teacher_reads == 0
    with Image.open(io.BytesIO(png)) as image:
        assert image.size == (360, 720)
    frame = world.teacher_snapshot()
    assert world.teacher_reads == 1
    assert frame.runtime_projection["provenance"]["runtime_visibility"] == "teacher_blind"


def test_phoneworld_async_action_requires_advance():
    world = PhoneWorld(start_app="settings", start_screen="home")
    frame = world.teacher_snapshot()
    target = next(node for node in frame.nodes if node.label == "Display")
    x1, y1, x2, y2 = target.bounds_px
    result = world.tap((x1 + x2) // 2, (y1 + y2) // 2)
    assert result.injected and world.screen_name == "home" and world.pending
    world.advance(300)
    assert world.screen_name == "display" and not world.pending


def test_phoneworld_rejects_action_while_transition_pending():
    world = PhoneWorld(start_app="settings", start_screen="home")
    target = next(node for node in world.teacher_snapshot().nodes if node.label == "Display")
    x1, y1, x2, y2 = target.bounds_px
    assert world.tap((x1 + x2) // 2, (y1 + y2) // 2).injected
    second = world.tap((x1 + x2) // 2, (y1 + y2) // 2)
    assert not second.injected and "pending" in second.reason


def test_phoneworld_variants_change_dimensions_and_surface():
    world = PhoneWorld(start_app="settings", start_screen="display")
    normal = world.capture_png()
    world.set_variant(DisplayVariant(orientation="landscape", density=1.1, variant_id="land"))
    changed = world.capture_png()
    assert normal != changed
    with Image.open(io.BytesIO(changed)) as image:
        assert image.size == (792, 396)


def test_phoneworld_task_catalog_has_cold_warm_and_holdout_tasks():
    catalog = PhoneWorld.task_catalog()
    assert {"settings_dark_mode", "profile_display_name", "timer_start_stop", "holdout_connectivity"} <= set(catalog)
    assert catalog["settings_dark_mode"].start_screen == "home"
