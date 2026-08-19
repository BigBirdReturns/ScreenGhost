import json

import pytest

from experiments.dti.profile import DTIProfile, PROFILE_SCHEMA
from experiments.dti.schema import GameAction, GameActionKind
from experiments.dti.windows_driver import WindowTarget, WindowsGameDriver


def profile_mapping():
    return {
        "schema": PROFILE_SCHEMA,
        "app_version": "fixture",
        "window_title_pattern": "Roblox",
        "client_size": [1600, 900],
        "observation_regions": {"theme": [0.25, 0.05, 0.75, 0.15]},
        "anchors": [
            {
                "anchor_id": "teleport",
                "semantic_label": "Quick Teleport",
                "role": "button",
            }
        ],
        "quick_teleport_labels": ["Dressing Room", "Freeplay Runway"],
        "wardrobe": [
            {
                "item_id": "test_dress",
                "slot": "dress",
                "tags": ["gothic"],
                "palettes": ["black"],
                "anchor": "dressing_room",
            }
        ],
    }


def test_profile_is_content_addressed() -> None:
    first = DTIProfile.from_mapping(profile_mapping())
    second = DTIProfile.from_mapping(profile_mapping())
    assert first.profile_id == second.profile_id
    assert first.client_size == (1600, 900)
    assert len(first.wardrobe.items) == 1


def test_profile_rejects_invalid_region() -> None:
    value = profile_mapping()
    value["observation_regions"] = {"bad": [0.8, 0.1, 0.2, 0.9]}
    with pytest.raises(ValueError):
        DTIProfile.from_mapping(value)


def test_windows_driver_imports_safely_off_windows() -> None:
    driver = WindowsGameDriver(WindowTarget())
    if not driver.is_windows():
        assert not driver.available()
        doctor = driver.doctor()
        assert not doctor.ready
        assert not doctor.windows


def test_action_contract_rejects_unbounded_hold() -> None:
    with pytest.raises(ValueError):
        GameAction(
            action_id="too-long",
            kind=GameActionKind.HOLD,
            semantic_label="move",
            keys=("w",),
            duration_ms=5000,
        )


def test_window_target_requires_primary_capture_output_by_default() -> None:
    target = WindowTarget()
    assert target.require_primary_output
    assert not WindowTarget(require_primary_output=False).require_primary_output


def test_capture_output_boundary_is_fail_closed() -> None:
    driver = WindowsGameDriver(WindowTarget(require_primary_output=True))
    driver._primary_monitor_rect = lambda: (0, 0, 1920, 1080)
    assert driver._capture_output_allowed((100, 100, 1700, 1000))
    assert not driver._capture_output_allowed((-1, 100, 1599, 1000))
    assert not driver._capture_output_allowed((100, 100, 2000, 1000))


def test_capture_output_boundary_can_be_explicitly_relaxed() -> None:
    driver = WindowsGameDriver(WindowTarget(require_primary_output=False))
    driver._primary_monitor_rect = lambda: (_ for _ in ()).throw(AssertionError("not used"))
    assert driver._capture_output_allowed((-500, 0, 1100, 900))
