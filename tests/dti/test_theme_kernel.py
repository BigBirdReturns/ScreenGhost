from experiments.dti.schema import AccessTier, ThemeCard, WardrobeItem
from experiments.dti.theme_kernel import (
    OutfitPlanner,
    ThemeCatalog,
    WardrobeAtlas,
    seed_theme_cards,
)


def test_exact_alias_resolves() -> None:
    result = ThemeCatalog(seed_theme_cards()).resolve("Theme: Greek Mythology")
    assert result.resolved
    assert result.card is not None
    assert result.card.canonical_name == "Greek God or Goddess"
    assert result.confidence == 1.0


def test_fuzzy_ocr_error_resolves_with_margin() -> None:
    result = ThemeCatalog(seed_theme_cards()).resolve("Victoran")
    assert result.resolved
    assert result.card is not None
    assert result.card.canonical_name == "Victorian"


def test_weak_unknown_theme_refuses() -> None:
    result = ThemeCatalog(seed_theme_cards()).resolve("zzq orbital accountant")
    assert not result.resolved
    assert result.card is None
    assert result.alternatives


def _atlas() -> WardrobeAtlas:
    return WardrobeAtlas(
        [
            WardrobeItem(
                item_id="lace_dress",
                slot="dress",
                tags=("gothic", "romantic", "lace", "flowing"),
                palettes=("black", "burgundy"),
                anchor="dressing_room",
                route_cost=1.0,
            ),
            WardrobeItem(
                item_id="vip_couture",
                slot="dress",
                tags=("gothic", "romantic", "formal", "lace"),
                palettes=("black", "burgundy"),
                anchor="vip_room",
                access=AccessTier.VIP,
                route_cost=3.0,
            ),
            WardrobeItem(
                item_id="long_waves",
                slot="hair",
                tags=("romantic", "flowing"),
                palettes=("black",),
                anchor="dressing_room",
                route_cost=1.0,
            ),
            WardrobeItem(
                item_id="black_heels",
                slot="shoes",
                tags=("formal",),
                palettes=("black",),
                anchor="dressing_room",
                route_cost=1.0,
            ),
            WardrobeItem(
                item_id="rose_choker",
                slot="accessory",
                tags=("gothic", "choker", "roses"),
                palettes=("black", "burgundy"),
                anchor="dressing_room",
                route_cost=1.0,
            ),
        ]
    )


def test_planner_filters_vip_and_charges_anchor_once() -> None:
    theme = ThemeCatalog(seed_theme_cards()).resolve("Gothic Romance").card
    assert theme is not None
    plan = OutfitPlanner().plan(theme, _atlas())
    assert "vip_couture" not in plan.item_ids
    assert "lace_dress" in plan.item_ids
    assert "rose_choker" in plan.item_ids
    assert plan.route_cost == 1.0
    assert not plan.unresolved


def test_plan_is_content_deterministic() -> None:
    theme = ThemeCatalog(seed_theme_cards()).resolve("Gothic Romance").card
    assert theme is not None
    first = OutfitPlanner().plan(theme, _atlas())
    second = OutfitPlanner().plan(theme, _atlas())
    assert first.plan_id == second.plan_id
    assert first.to_dict() == second.to_dict()
