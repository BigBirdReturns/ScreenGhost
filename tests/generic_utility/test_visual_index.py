from __future__ import annotations

from experiments.generic_utility.phone_world import DisplayVariant, PhoneWorld
from experiments.generic_utility.visual_index import VisualIndexPolicy, VisualStateIndex


def add(world, index, *, family_id=None):
    frame = world.teacher_snapshot()
    return index.add(frame.png_bytes, frame.runtime_projection, family_id=family_id)


def test_exact_visual_variant_matches():
    world = PhoneWorld(start_app="settings", start_screen="home")
    index = VisualStateIndex(policy=VisualIndexPolicy(minimum_margin=0.05))
    add(world, index)
    match = index.match(world.capture_png(), app_family_hint="settings", screen_name_hint="home")
    assert match.known and match.confidence == 1.0


def test_screen_hint_turns_action_history_into_narrow_verification():
    world = PhoneWorld(start_app="settings", start_screen="home")
    index = VisualStateIndex(policy=VisualIndexPolicy(minimum_margin=0.05))
    add(world, index)
    world.reset(app_family="settings", screen_name="display")
    add(world, index)
    match = index.match(
        world.capture_png(), app_family_hint="settings", screen_name_hint="display"
    )
    assert match.known and match.projection["screen_name"] == "display"


def test_screen_key_hint_verifies_same_screen_state_change():
    world = PhoneWorld(start_app="settings", start_screen="display")
    index = VisualStateIndex(policy=VisualIndexPolicy(minimum_margin=0.05))
    variant = add(world, index)
    world._dark_mode = True
    add(world, index, family_id=variant.family_id)
    match = index.match(
        world.capture_png(), app_family_hint="settings", screen_key_hint=variant.screen_key
    )
    assert match.known and match.screen_key == variant.screen_key


def test_renamed_control_is_unknown_without_transition_hint():
    world = PhoneWorld(start_app="settings", start_screen="display")
    index = VisualStateIndex(policy=VisualIndexPolicy(minimum_margin=0.05))
    add(world, index)
    world.reset(app_family="settings", screen_name="home")
    add(world, index)
    world.reset(app_family="settings", screen_name="display")
    world.set_variant(DisplayVariant(rename_control=True, variant_id="renamed"))
    assert not index.match(world.capture_png(), app_family_hint="settings").known


def test_moved_control_is_unknown():
    world = PhoneWorld(start_app="settings", start_screen="display")
    index = VisualStateIndex(policy=VisualIndexPolicy(minimum_margin=0.05))
    add(world, index)
    world.reset(app_family="settings", screen_name="home")
    add(world, index)
    world.reset(app_family="settings", screen_name="display")
    world.set_variant(DisplayVariant(move_controls=True, variant_id="moved"))
    assert not index.match(world.capture_png(), app_family_hint="settings").known


def test_unknown_and_lookalike_do_not_match():
    world = PhoneWorld(start_app="settings", start_screen="display")
    index = VisualStateIndex(policy=VisualIndexPolicy(minimum_margin=0.05))
    add(world, index)
    for app in ("unknown", "lookalike"):
        world.reset(app_family=app, screen_name="home")
        assert not index.match(world.capture_png()).known


def test_visual_index_roundtrip(tmp_path):
    path = tmp_path / "index.json"
    world = PhoneWorld(start_app="profile", start_screen="home")
    index = VisualStateIndex(path)
    add(world, index)
    loaded = VisualStateIndex(path)
    assert loaded.family_count == 1 and len(loaded.variants) == 1
