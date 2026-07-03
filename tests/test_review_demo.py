"""Review workflow: replay, catalog import, export, receipt, server, discipline."""
import json
import os
import threading
import urllib.request as u

import pytest

from core.catalog_io import read_catalog_csv, rows_to_skus, write_catalog_csv
from core.ledger_store import LedgerStore
from core.population import build_population, resolve_sku
from core.review import (
    _RECEIPT_ORDER, auto_review, export_session, populate_world,
    replay_event_values, reviewer_resolve, run_review_session,
)
from core.review_server import make_server


def test_review_session_reproduction_improves():
    store = LedgerStore(":memory:")
    worlds = build_population(n=10, seed=1337)
    r = run_review_session(store, worlds, seed="t")
    assert r["ledger_reproduction_after_correction"] >= r["ledger_reproduction_before_correction"]
    assert r["corrected_events"] > 0


def test_replay_with_correction_log_reproduces_values():
    store = LedgerStore(":memory:")
    worlds = build_population(n=6, seed=9)
    for w in worlds:
        populate_world(store, w)
        auto_review(store, w.profile.seller_id)
        reviewer_resolve(store, w)
    for w in worlds:
        sid = w.profile.seller_id
        replay = replay_event_values(store, sid)
        live = {e["event_id"]: (e["buyer"], e["sku"], e["qty"])
                for e in store.events(sid)}
        assert replay == live


def test_demo_runs_ten_sellers():
    store = LedgerStore(":memory:")
    worlds = build_population(n=10, seed=1)
    r = run_review_session(store, worlds, seed="demo")
    assert r["sellers"] == 10 and r["proposed_events"] > 0


def test_catalog_csv_import_and_resolution(tmp_path):
    p = str(tmp_path / "cat.csv")
    write_catalog_csv(p, [{"sku": "A01", "name": "เซรั่ม",
                           "aliases": ["serum", "เซรั่ม"], "variants": [],
                           "price": 390, "stock": 1}])
    rows = read_catalog_csv(p)
    assert rows[0]["aliases"] == ["serum", "เซรั่ม"]           # exact Thai alias
    assert resolve_sku("cf serum x2 ค่ะ", rows_to_skus(rows)) == "A01"


def test_catalog_missing_columns_rejected(tmp_path):
    p = str(tmp_path / "bad.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        read_catalog_csv(p)


def test_export_generates_files_and_receipt(tmp_path):
    store = LedgerStore(":memory:")
    worlds = build_population(n=5, seed=2)
    r = run_review_session(store, worlds, seed="exp")
    paths = export_session(store, worlds, r, str(tmp_path))
    for k in ("orders.csv", "corrections.csv", "captures.csv", "receipt.txt"):
        assert os.path.exists(paths[k])
    txt = open(paths["receipt.txt"], encoding="utf-8").read()
    assert "NOT hardware proof" in txt and "NOT business proof" in txt


def test_receipt_keeps_denominators_separate():
    store = LedgerStore(":memory:")
    worlds = build_population(n=5, seed=3)
    r = run_review_session(store, worlds, seed="d")
    assert "total_messages" in r and "order_bearing_messages" in r
    assert r["total_messages"] >= r["order_bearing_messages"]
    for k in _RECEIPT_ORDER:
        assert k in r


def test_no_category_borrowing_from_proof_matrix(tmp_path):
    # the product slice must not launder synthetic/[1] proof into hardware/business.
    store = LedgerStore(":memory:")
    worlds = build_population(n=3, seed=7)
    r = run_review_session(store, worlds, seed="nc")
    paths = export_session(store, worlds, r, str(tmp_path))
    txt = open(paths["receipt.txt"], encoding="utf-8").read().lower()
    assert "synthetic" in txt
    assert "not hardware proof" in txt and "not business proof" in txt


def test_local_review_server_serves_and_acts():
    store = LedgerStore(":memory:")
    worlds = build_population(n=2, seed=5)
    for w in worlds:
        populate_world(store, w)          # leave events 'proposed' so accept is legal
    srv = make_server(store, worlds, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    base = f"http://{host}:{port}"
    try:
        assert "Review Ledger" in u.urlopen(base + "/").read().decode()
        sellers = json.loads(u.urlopen(base + "/api/sellers").read())
        sid = sellers[0]["seller_id"]
        evs = json.loads(u.urlopen(base + f"/api/events?seller={sid}").read())
        assert len(evs) > 0
        req = u.Request(base + "/api/action", method="POST",
                        data=json.dumps({"event_id": evs[0]["event_id"],
                                         "action": "accept"}).encode())
        assert json.loads(u.urlopen(req).read())["ok"] is True
    finally:
        srv.shutdown()
