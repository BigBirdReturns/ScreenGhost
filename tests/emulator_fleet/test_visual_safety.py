from pathlib import Path

from experiments.emulator_fleet.campaign import (
    _teach_variant,
    record_coordinate_demonstration,
)
from experiments.emulator_fleet.simulated import SimulatedFleetSpec, SimulatedInstanceAdapter
from experiments.generic_utility.phone_world import DisplayVariant
from experiments.generic_utility.visual_index import VisualIndexPolicy, VisualStateIndex


def _index(tmp_path: Path):
    leader = SimulatedInstanceAdapter(
        SimulatedFleetSpec("leader", "Leader", DisplayVariant()), seed=20
    )
    _macro, snapshots = record_coordinate_demonstration(leader)
    index = VisualStateIndex(
        tmp_path / "index.json",
        policy=VisualIndexPolicy(
            minimum_confidence=0.92,
            minimum_margin=0.03,
            minimum_crop_confidence=0.965,
            maximum_variants_per_family=48,
        ),
    )
    return snapshots, index


def test_exact_pixels_can_bypass_cross_family_margin(tmp_path: Path):
    snapshots, index = _index(tmp_path)
    dark = SimulatedInstanceAdapter(
        SimulatedFleetSpec("dark", "Dark", DisplayVariant(theme="dark")), seed=21
    )
    _teach_variant(dark, snapshots, index, variant_label="dark")
    dark.reset_task("settings_dark_mode")
    match = index.match(dark.capture_png())
    assert match.known and match.confidence == 1.0


def test_crop_gate_rejects_deceptive_lookalike(tmp_path: Path):
    snapshots, index = _index(tmp_path)
    taught = SimulatedInstanceAdapter(
        SimulatedFleetSpec("teach", "Teach", DisplayVariant()), seed=22
    )
    _teach_variant(taught, snapshots, index, variant_label="default")
    lookalike = SimulatedInstanceAdapter(
        SimulatedFleetSpec("look", "Look", DisplayVariant()), seed=23
    )
    lookalike.world.reset(app_family="lookalike", screen_name="home")
    match = index.match(lookalike.capture_png())
    assert not match.known
    assert match.confidence > 0.90  # confirms rejection came from the stricter gate


def test_state_hint_selects_checked_variant(tmp_path: Path):
    snapshots, index = _index(tmp_path)
    landscape = SimulatedInstanceAdapter(
        SimulatedFleetSpec(
            "landscape", "Landscape", DisplayVariant(orientation="landscape")
        ),
        seed=24,
    )
    _teach_variant(landscape, snapshots, index, variant_label="landscape")
    landscape.restore_state(snapshots[2])
    match = index.match(
        landscape.capture_png(),
        target_label_hint="Dark mode",
        target_role_hint="switch",
        state_key_hint="checked",
        state_value_hint="true",
    )
    assert match.known
    element = match.to_student_observation().get_element("Dark mode")
    assert element.states["checked"] == "true"
