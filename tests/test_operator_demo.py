"""Operator Demo Pack v1 + ledger hardening: fixture, resolver, replay, receipt."""
import os

from core.fixtures import load_seller_worlds
from core.ledger_store import (
    ACCEPTED, LedgerStore, NEEDS_INFO, PROPOSED_INCOMPLETE,
)
from core.population import Sku, build_population, resolve_sku
from core.resolver import resolve, variant_missing
from core.review import (
    auto_review, populate_world, replay_ledger, run_review_session,
)

FIXTURE = "examples/fixtures/seller_world_demo_seed.json"


def test_demo_fixture_loads_all_scenarios():
    worlds = load_seller_worlds(FIXTURE)
    ids = {w.profile.seller_id for w in worlds}
    assert {"clean-shop", "messy-shop", "adversarial-shop"} <= ids
    suites = {lm.adversarial for w in worlds for lm in w.messages}
    for s in ("spelling", "alias", "variant_ambig", "repeat_cf",
              "stockout_sub", "payment_ref", "cancel", "modify"):
        assert s in suites


def test_operator_demo_runs_ten_sellers(tmp_path):
    from examples.operator_demo import run
    receipt, paths, receipt_file, _store = run("op", 10, str(tmp_path))
    assert receipt["sellers"] == 10
    assert receipt["replay_matched"] is True
    for k in ("orders.csv", "corrections.csv", "captures.csv", "receipt.txt"):
        assert os.path.exists(paths[k])
    assert os.path.exists(receipt_file)


def test_missing_variant_becomes_incomplete_not_completed():
    # A SKU that needs a variant, ordered without one, must NOT become an
    # accepted (completed) order — it routes to a human.
    store = LedgerStore(":memory:")
    adv = next(w for w in load_seller_worlds(FIXTURE)
               if w.profile.seller_id == "adversarial-shop")
    populate_world(store, adv)
    incomplete = store.events("adversarial-shop", status=PROPOSED_INCOMPLETE)
    assert len(incomplete) >= 1                     # the CF กระเป๋า (no color)
    auto_review(store, "adversarial-shop")
    # every incomplete event ended in needs_info, never auto-accepted
    for ev in incomplete:
        assert store.get_event(ev["event_id"])["status"] == NEEDS_INFO


def test_alias_normalization_improves_recall_without_hiding_failures():
    catalog = [Sku("A01", "เซรั่มหน้าใส", ["serum", "เซรั่ม"], [], 390),
               Sku("A02", "ครีมกันแดด", ["กันแดด"], [], 290)]
    # a misspelled Thai alias the baseline resolver misses, the product recovers
    garbled = "CF เซรัม x1 ค่ะ"          # เซรั่ม with a dropped char
    assert resolve_sku(garbled, catalog) is None       # baseline: honest miss
    assert resolve(garbled, catalog) == "A01"          # product: recovered
    # but genuine ambiguity is still NOT invented
    clash = [Sku("Z01", "สินค้า", ["item"], [], 1), Sku("Z02", "สินค้า", ["item"], [], 1)]
    assert resolve("cf item x1", clash) is None


def test_variant_missing_detection():
    with_variants = Sku("D01", "กระเป๋า", ["bag"], ["ดำ", "แดง"], 500)
    assert variant_missing("CF กระเป๋า x1 ค่ะ", with_variants) is True
    assert variant_missing("CF กระเป๋า ดำ x1 ค่ะ", with_variants) is False
    no_variants = Sku("C01", "เค้ก", ["cake"], [], 300)
    assert variant_missing("CF เค้ก x1", no_variants) is False


def test_transitions_are_append_only_records():
    store = LedgerStore(":memory:")
    store.add_seller("s1")
    store.add_capture("c1", "s1", "Nok", "t1", "CF A01 x1", "c1", "x")
    store.propose_event("c1", "s1", "c1", "Nok", "A01", 1, None, "order", 0.9, "c1")
    store.transition("c1", ACCEPTED, source="human")
    store.transition("c1", "fulfilled", source="human")
    log = store.transitions("c1")
    assert [(t["from_status"], t["to_status"]) for t in log] == \
        [("proposed", "accepted"), ("accepted", "fulfilled")]


def test_replay_reproduces_final_ledger():
    store = LedgerStore(":memory:")
    worlds = build_population(n=8, seed=4)
    run_review_session(store, worlds, seed="r")
    for w in worlds:
        _replayed, matched = replay_ledger(store, w.profile.seller_id)
        assert matched is True


def test_operator_receipt_makes_no_hardware_or_business_claim(tmp_path):
    from examples.operator_demo import run
    _r, _p, receipt_file, _s = run("nc", 5, str(tmp_path))
    txt = open(receipt_file, encoding="utf-8").read().lower()
    assert "not hardware proof" in txt and "not business proof" in txt
