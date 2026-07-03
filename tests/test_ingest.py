"""Keep-up + recall receipts, run offline.

Proves the pipeline the scale objection said couldn't exist: at realistic live
volume, order-event recall is complete, latency is seller-useful, Thai text is
never corrupted, and no order is double-counted — and the *boundary* where it
starts missing is detected, not hidden.
"""
from core.ingest import ReplayDriver, simulate_keepup


def test_replay_driver_serves_frames_then_repeats():
    d = ReplayDriver(["a", "b"])
    assert [d.dump_ui_xml(), d.dump_ui_xml(), d.dump_ui_xml()] == ["a", "b", "b"]


def test_keeps_up_at_realistic_volume():
    # 50 comments/sec (3,000/min) for 30s, 0.3s poll, window 120.
    # arrivals/interval = 15 << 120 -> full recall with ~8x headroom.
    r = simulate_keepup(arrival_rate=50, poll_interval_s=0.3, window=120, duration_s=30)
    assert r.missed == 0
    assert r.order_recall == 1.0
    assert r.unicode_corruptions == 0
    assert r.keeps_up is True
    assert r.headroom >= 5
    # dedup actually did work — the same comments were re-read many times.
    assert r.duplicates_suppressed > r.arrived


def test_latency_is_seller_useful():
    r = simulate_keepup(arrival_rate=50, poll_interval_s=0.3, window=120, duration_s=30)
    # An order should become a structured event within a couple of polls.
    assert r.p95_latency_s <= 1.0
    assert r.max_latency_s <= 2.0


def test_two_buyers_same_text_not_merged_at_volume():
    # Recall counts distinct order messages; identical text from distinct
    # (sender, ts) must not collapse, or recall would silently drop.
    r = simulate_keepup(arrival_rate=40, poll_interval_s=0.25, window=100, duration_s=20)
    assert r.order_captured == r.order_arrived
    assert r.order_precision == 1.0


def test_boundary_is_reported_not_hidden():
    # Firehose past the window: 1,000/sec, 0.3s poll, window 100 ->
    # 300 arrive per interval but only 100 fit -> misses, honestly counted.
    r = simulate_keepup(arrival_rate=1000, poll_interval_s=0.3, window=100, duration_s=3)
    assert r.missed > 0
    assert r.order_recall < 1.0
    assert r.keeps_up is False
    assert r.headroom < 1
