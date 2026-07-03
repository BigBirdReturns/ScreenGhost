"""Release-candidate guards: setup, determinism, schema, verifier, errors, UI."""
import json
import os
import subprocess
import sys
import threading
import urllib.request as u

import pytest

from core.catalog_io import read_catalog_csv
from core.ledger_store import LedgerStore, SchemaError
from core.population import build_population
from core.review import populate_world, run_review_session
from core.review_server import make_server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_setup_smoke_runs():
    r = subprocess.run([sys.executable, "-m", "tools.setup_demo"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "smoke test... ok" in r.stdout


def test_population_is_deterministic_across_processes():
    code = ("from core.population import build_population;"
            "w=build_population(n=8,seed=1337);"
            "print(sum(len(x.messages) for x in w))")
    a = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    b = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert a.stdout.strip() == b.stdout.strip() and a.stdout.strip()


def test_schema_version_present():
    s = LedgerStore(":memory:")
    assert s.schema_version() == 1
    assert "created_at_utc" in s.meta() and "app_version" in s.meta()


def test_unknown_future_schema_fails_closed(tmp_path):
    db = str(tmp_path / "l.db")
    LedgerStore(db).close()
    # simulate a store written by a newer build
    s = LedgerStore(db)
    s.db.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    s.db.commit()
    s.close()
    with pytest.raises(SchemaError):
        LedgerStore(db)


def test_raw_captures_survive_reopen(tmp_path):
    db = str(tmp_path / "l.db")
    s = LedgerStore(db)
    s.add_seller("s1")
    s.add_capture("c1", "s1", "Nok", "t1", "CF บี12 x3 ค่ะ", "c1", "in_process")
    s.close()
    s2 = LedgerStore(db)
    assert s2.get_capture("c1")["raw_text"] == "CF บี12 x3 ค่ะ"


def test_demo_exits_clean_and_prints_replay_command(tmp_path):
    r = subprocess.run(
        [sys.executable, "examples/operator_demo.py", "--seed", "op",
         "--sellers", "5", "--out", str(tmp_path / "o")],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "python examples/replay_ledger.py --store" in r.stdout
    assert "NOT hardware proof" in r.stdout


def test_receipt_verifier_matches_canonical():
    r = subprocess.run(
        [sys.executable, "examples/verify_demo_receipt.py", "--receipt",
         "examples/receipts/operator_demo_seed_op.txt"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MATCH" in r.stdout


def test_malformed_catalog_csv_gives_actionable_error(tmp_path):
    p = str(tmp_path / "bad.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("foo,bar\n1,2\n")
    with pytest.raises(ValueError) as e:
        read_catalog_csv(p)
    assert "missing required columns" in str(e.value)


def test_missing_fixture_gives_actionable_error(tmp_path):
    from examples.operator_demo import DemoError, run
    with pytest.raises(DemoError) as e:
        run(fixture="does/not/exist.json", out=str(tmp_path))
    assert "missing fixture" in str(e.value)


def test_invalid_seed_rejected(tmp_path):
    from examples.operator_demo import DemoError, run
    with pytest.raises(DemoError):
        run(seed="", out=str(tmp_path))


def test_ui_routes_render():
    store = LedgerStore(":memory:")
    worlds = build_population(n=2, seed=5)
    for w in worlds:
        populate_world(store, w)
    srv = make_server(store, worlds, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://{srv.server_address[0]}:{srv.server_address[1]}"
    try:
        assert "Review Ledger" in u.urlopen(base + "/").read().decode()
        sellers = json.loads(u.urlopen(base + "/api/sellers").read())
        sid = sellers[0]["seller_id"]
        evs = json.loads(u.urlopen(base + f"/api/events?seller={sid}").read())
        ev = json.loads(u.urlopen(base + f"/api/event/{evs[0]['event_id']}").read())
        # enriched detail: raw text + parser path + transitions present
        assert "raw_text" in ev and "parser_path" in ev and "transitions" in ev
    finally:
        srv.shutdown()


def test_committed_receipts_carry_no_forbidden_claim():
    for name in ("operator_demo_seed_op.txt", "demo_rc0_seed_op.txt"):
        txt = open(os.path.join(ROOT, "examples", "receipts", name),
                   encoding="utf-8").read().lower()
        assert "not hardware proof" in txt and "not business proof" in txt
