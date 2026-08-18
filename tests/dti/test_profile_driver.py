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
