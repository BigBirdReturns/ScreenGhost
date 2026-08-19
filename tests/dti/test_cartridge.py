from experiments.dti.cartridge import DTICartridge
from experiments.dti.profile import DTIProfile, PROFILE_SCHEMA
from experiments.dti.rounds import RoundSignals


def profile() -> DTIProfile:
    return DTIProfile.from_mapping(
        {
            "schema": PROFILE_SCHEMA,
            "app_version": "fixture",
            "window_title_pattern": "Roblox",
            "client_size": [1600, 900],
            "observation_regions": {},
            "anchors": [],
            "quick_teleport_labels": ["Dressing Room"],
            "wardrobe": [
                {
                    "item_id": "lace_dress",
                    "slot": "dress",
                    "tags": ["gothic", "romantic", "lace"],
                    "palettes": ["black", "burgundy"],
                    "anchor": "dressing_room",
                },
                {
                    "item_id": "long_hair",
                    "slot": "hair",
                    "tags": ["romantic", "flowing"],
                    "palettes": ["black"],
                    "anchor": "dressing_room",
                },
                {
                    "item_id": "heels",
                    "slot": "shoes",
                    "tags": ["formal"],
                    "palettes": ["black"],
                    "anchor": "dressing_room",
                },
            ],
        }
    )


def test_cartridge_interprets_and_plans_visible_theme() -> None:
    cartridge = DTICartridge(profile())
    result = cartridge.interpret(
        RoundSignals(
            theme_text="Gothic Romance",
            timer_seconds=300,
            labels=("Theme", "Timer"),
        )
    )
    assert result.theme is not None and result.theme.resolved
    assert result.plan is not None
    assert result.plan.theme_name == "Gothic Romance"
    assert "lace_dress" in result.plan.item_ids


def test_cartridge_refuses_weak_theme_without_guessing() -> None:
    cartridge = DTICartridge(profile())
    result = cartridge.interpret(
        RoundSignals(theme_text="zzq orbital accountant", timer_seconds=300)
    )
    assert result.theme is not None
    assert not result.theme.resolved
    assert result.plan is None
