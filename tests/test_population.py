"""Population-harness receipts, run offline.

Proves the same capture->event pipeline against labeled merchant reality:
generation + ground-truth ledger, exact-text preservation, dedupe, order
correction (modify/cancel) accounting, catalog resolution, honest adversarial
failure, and cohort reporting — plus that a too-small window still fails openly.
"""
from core.orders import EventType, classify_event, reduce_ledger
from core.population import (
    build_population, generate_population, resolve_sku,
)
from core.eval_population import evaluate_world, run_population


def test_population_generates_all_cohorts():
    sellers = generate_population(n=60, seed=1337)
    assert len(sellers) == 60
    cohorts = {s.cohort for s in sellers}
    assert len(cohorts) == 10
    for s in sellers:
        assert s.catalog and s.buyers
        assert s.traffic in {"normal", "busy", "hot", "burst", "quiet"}


def test_event_classification():
    assert classify_event("CF A01 x2 ค่ะ") == EventType.ORDER
    assert classify_event("เปลี่ยนเป็น 3 ชิ้น A01 ค่ะ") == EventType.MODIFY
    assert classify_event("ยกเลิก A01 ค่ะ") == EventType.CANCEL
    assert classify_event("สวัสดีค่ะ ราคาเท่าไหร่") == EventType.CHATTER


def test_ledger_reducer_applies_corrections():
    seq = [
        (EventType.ORDER, "b1", "A01", 2),
        (EventType.MODIFY, "b1", "A01", 5),   # replaces
        (EventType.ORDER, "b2", "A01", 1),
        (EventType.CANCEL, "b2", "A01", None),  # removes
    ]
    assert reduce_ledger(seq) == {("b1", "A01"): 5}


def test_resolve_sku_and_alias_ambiguity():
    from core.population import Sku
    catalog = [Sku("A01", "เซรั่ม", ["serum"], [], 100),
               Sku("A02", "ครีมกันแดด", ["กันแดด"], [], 200)]
    assert resolve_sku("CF A01 x1", catalog) == "A01"       # literal code
    assert resolve_sku("cf serum x2 ค่ะ", catalog) == "A01"  # unique alias
    assert resolve_sku("cf x1 ค่ะ", catalog) is None          # no match
    clash = [Sku("Z01", "สินค้า", ["item"], [], 1),
             Sku("Z02", "สินค้า", ["item"], [], 1)]
    assert resolve_sku("cf item x1", clash) is None           # ambiguous -> None


def test_pipeline_preserves_unicode_and_never_double_emits():
    worlds = build_population(n=80, seed=7)
    corruptions = sum(evaluate_world(w).unicode_corruptions for w in worlds)
    dup_events = sum(evaluate_world(w).dup_actionable for w in worlds)
    assert corruptions == 0     # exact Thai text through the whole pipeline
    assert dup_events == 0      # each comment becomes at most one order event


def test_clean_cohorts_reproduce_the_ledger():
    out = run_population(build_population(n=100, seed=1337))
    by = {c.cohort: c for c in out["cohorts"]}
    # bakery/homegoods carry only clean orders + modify/cancel -> full recall.
    assert by["bakery"].recall == 1.0 and by["bakery"].precision == 1.0
    assert by["homegoods"].recall == 1.0
    assert by["bakery"].keeps_up is True


def test_adversarial_suites_fail_in_the_open():
    out = run_population(build_population(n=100, seed=1337))
    suites = {name: (rec, prec, fp) for name, _n, rec, prec, fp in out["adversarial"]}
    # misspelled aliases don't resolve -> real recall miss, reported not hidden.
    assert suites["spelling"][0] < 0.6
    # stock-out orders and "CF CF" spam are emitted but shouldn't stand -> FP.
    assert suites["stockout_sub"][1] == 0.0 and suites["stockout_sub"][2] > 0
    assert suites["repeat_cf"][1] == 0.0 and suites["repeat_cf"][2] > 0
    # ledger isn't magically perfect across collisions/substitutions.
    assert 0.0 < out["ledger_match_rate"] < 1.0


def test_cohort_receipt_shape():
    out = run_population(build_population(n=40, seed=99))
    assert len(out["cohorts"]) == 10
    c = out["cohorts"][0]
    for field in ("recall", "precision", "order_msgs_min", "total_msgs_min",
                  "p95_latency_s", "backlog_rate", "keeps_up"):
        assert hasattr(c, field)


def test_too_small_window_still_fails_honestly():
    # Squeeze the visible window below arrival volume: comments scroll off
    # unseen and the harness must report it, not paper over it.
    # hot/burst arrive >2 comments per poll interval; window=2 is smaller, so
    # some must scroll off unseen.
    world = next(w for w in build_population(n=60, seed=1)
                 if w.profile.traffic in {"hot", "burst"})
    tight = evaluate_world(world, window=2)
    assert tight.missed > 0
