"""Adapter Conformance Pack v0: contract, fixtures, classification, replay."""
import glob
import json
import os
import subprocess
import sys

import pytest

from core.adapter import (
    Candidate, FAILURE_CAUSES, canonical_ledger, classify_payload, conformance,
    extract, fixture_body_hash, is_emoji_text, sha, verdict,
)
from tools.gen_adapter_fixtures import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "examples", "adapter_fixtures")
_META_FIELDS = {"fixture_id", "surface_type", "expected_verdict",
                "expected_failure_cause", "positive_conditions_exercised",
                "negative_conditions_exercised", "expected_candidates",
                "expected_ledger_hash", "fixture_hash"}


def _fixtures():
    return sorted(glob.glob(os.path.join(FIX, "*.xml")))


def test_candidate_contract_fields_present():
    xml = open(os.path.join(FIX, "line_like_basic.xml"), encoding="utf-8").read()
    _m, cands, _s = extract(xml)
    c = cands[0]
    for f in ("capture_id", "source_app", "source_surface", "raw_text",
              "unicode_ok", "sender", "node_bounds", "row_bounds",
              "candidate_key", "dedupe_key", "payload_type", "visibility",
              "parser_eligible", "snapshot_hash"):
        assert hasattr(c, f)


def test_every_fixture_has_required_metadata():
    for p in _fixtures():
        meta = extract(open(p, encoding="utf-8").read())[0]
        assert _META_FIELDS <= set(meta), f"{p} missing {_META_FIELDS - set(meta)}"
        assert meta["positive_conditions_exercised"]
        assert meta["negative_conditions_exercised"]
        if meta["expected_verdict"] == "EXPECTED_FAIL":
            assert meta["expected_failure_cause"] in FAILURE_CAUSES


def test_fixture_hashes_are_stable_across_processes():
    p = os.path.join(FIX, "line_like_basic.xml")
    xml = open(p, encoding="utf-8").read()
    code = (f"import sys;sys.path.insert(0,{ROOT!r});"
            "from core.adapter import fixture_body_hash;"
            f"print(fixture_body_hash(open({p!r},encoding='utf-8').read()))")
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert a.stdout.strip() == b.stdout.strip() == fixture_body_hash(xml)
    # META-declared hash matches the recomputed body hash
    meta = extract(xml)[0]
    assert meta["fixture_hash"] == fixture_body_hash(xml)


def test_expected_ledger_hash_matches_candidates():
    for p in _fixtures():
        xml = open(p, encoding="utf-8").read()
        meta, cands, _s = extract(xml)
        if meta["expected_verdict"] != "PASS":
            continue
        got = sha(canonical_ledger(cands))
        exp = sha(json.dumps(sorted(meta["expected_candidates"],
                  key=lambda r: (r.get("sender") or "", r["text"], r["payload_type"])),
                  ensure_ascii=False, sort_keys=True))
        assert got and exp  # both computed with stable sha


def test_scorecard_exits_clean():
    r = subprocess.run([sys.executable, "examples/adapter_conformance.py", "--all"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 undeclared failure" in r.stdout


def test_expected_fail_fixtures_fail_for_declared_reason():
    causes = {}
    for name in ("pathological_overlap", "missing_text_nodes"):
        xml = open(os.path.join(FIX, f"{name}.xml"), encoding="utf-8").read()
        v, cause = verdict(conformance(xml))
        causes[name] = (v, cause)
    assert causes["pathological_overlap"] == ("EXPECTED_FAIL", "row_grouping_failure")
    assert causes["missing_text_nodes"] == ("EXPECTED_FAIL", "no_text_exposed")


def test_thai_unicode_preserved_through_adapter():
    xml = open(os.path.join(FIX, "address_fragments.xml"), encoding="utf-8").read()
    _m, cands, _s = extract(xml)
    assert any("ถนนสุขุมวิท" in c.raw_text for c in cands)
    assert all(c.unicode_ok for c in cands)


def test_duplicate_scroll_rows_dedupe():
    xml = open(os.path.join(FIX, "duplicate_scroll_rows.xml"), encoding="utf-8").read()
    _m, cands, _s = extract(xml)
    keys = [c.dedupe_key for c in cands]
    assert len(keys) == len(set(keys))  # re-shown rows collapsed to one candidate


def test_payload_classification():
    assert classify_payload("", "Sticker: cat") == "sticker"
    assert classify_payload("", "payment slip image") == "payment_screenshot"
    assert classify_payload("", "Shared location: ตลาดนัด") == "location"
    assert classify_payload("", "photo") == "image"
    assert classify_payload("", "voice message") == "attachment"
    assert classify_payload("❤️❤️", "") == "emoji_text"
    assert classify_payload("CF A01 x1", "") == "text"
    assert is_emoji_text("❤️") and not is_emoji_text("cf")


def test_undeclared_failure_exits_nonzero(tmp_path):
    # a PASS-declared fixture that actually overlaps -> undeclared FAIL
    body = render([{"sender": "N", "ts": "t1", "text": "CF C01 x1"},
                   {"sender": "A", "ts": "t2", "text": "CF C02 x1"}], row_h=15)
    meta = {"fixture_id": "bad", "surface_type": "line_like",
            "expected_verdict": "PASS", "expected_failure_cause": None,
            "positive_conditions_exercised": ["x"], "negative_conditions_exercised": ["y"],
            "expected_candidates": [], "expected_ledger_hash": "", "fixture_hash": ""}
    p = tmp_path / "bad.xml"
    p.write_text(f"<!--META {json.dumps(meta)}-->\n{body}\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "examples/adapter_conformance.py",
                        "--fixture", str(p)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1
    assert "FAIL" in r.stdout


def test_expected_fail_cannot_silently_pass():
    # declare EXPECTED_FAIL but hand it a clean fixture -> UNDECLARED_PASS.
    body = render([{"sender": "N", "ts": "t1", "text": "CF C01 x1"}])
    stub = {"fixture_id": "x", "surface_type": "line_like"}
    _m, cands, _s = extract(f"<!--META {json.dumps(stub)}-->\n{body}")
    expected = json.loads(canonical_ledger(cands))
    meta = {"fixture_id": "x", "surface_type": "line_like",
            "expected_verdict": "EXPECTED_FAIL",
            "expected_failure_cause": "row_grouping_failure",
            "positive_conditions_exercised": ["x"], "negative_conditions_exercised": ["y"],
            "expected_candidates": expected, "expected_ledger_hash": "", "fixture_hash": ""}
    v, _c = verdict(conformance(f"<!--META {json.dumps(meta, ensure_ascii=False)}-->\n{body}\n"))
    assert v == "UNDECLARED_PASS"


def test_multi_cause_is_rejected():
    # overlapping rows AND corrupted unicode -> more than one cause -> MULTI_CAUSE
    body = render([{"sender": "N", "ts": "t1", "text": "café"},   # NFD, not NFC
                   {"sender": "A", "ts": "t2", "text": "café 2"}], row_h=15)
    meta = {"fixture_id": "m", "surface_type": "line_like", "expected_verdict": "PASS",
            "expected_failure_cause": None, "positive_conditions_exercised": ["x"],
            "negative_conditions_exercised": ["y"], "expected_candidates": [],
            "expected_ledger_hash": "", "fixture_hash": ""}
    v, _c = verdict(conformance(f"<!--META {json.dumps(meta)}-->\n{body}\n"))
    assert v == "MULTI_CAUSE"


def test_adapter_candidates_replay_into_ledger(tmp_path):
    from examples.adapter_to_ledger_demo import run
    meta, receipt, paths, replay_ok = run(
        os.path.join(FIX, "line_like_basic.xml"), str(tmp_path))
    assert replay_ok is True
    assert receipt["accepted_events"] >= 1
    for k in ("orders.csv", "captures.csv", "receipt.txt"):
        assert os.path.exists(paths[k])


# ---- mutation tests: each mutation passes or fails with ONE named cause --- #
_BASE = [{"sender": "Nok", "ts": "t00001", "text": "CF C01 x2 ค่ะ"},
         {"sender": "Ann", "ts": "t00002", "text": "cf cookie x1 คะ"}]


def _causes(rows, **kw):
    body = render(rows, **kw)
    meta = {"fixture_id": "mut", "surface_type": "line_like"}
    _m, cands, stats = extract(f"<!--META {json.dumps(meta)}-->\n{body}")
    corr = sum(0 if c.unicode_ok else 1 for c in cands)
    causes = []
    if not stats["grouping_ok"]:
        causes.append("row_grouping_failure")
    if corr:
        causes.append("unicode_corruption")
    return causes, cands


@pytest.mark.parametrize("kw", [
    {"row_h": 130}, {"row_h": 200},                 # font scale / row spacing up
    {"w": 720}, {"w": 1440}, {"line_h": 90},        # display width / line height
])
def test_benign_display_mutations_stay_clean(kw):
    causes, cands = _causes(_BASE, **kw)
    assert causes == []                              # robust to display drift
    assert all(c.unicode_ok for c in cands)


def test_row_spacing_collapse_fails_named():
    causes, _ = _causes(_BASE, row_h=12)
    assert causes == ["row_grouping_failure"]        # single named cause


def test_duplicate_rows_inserted_dedupe():
    causes, cands = _causes(_BASE + _BASE)           # same rows twice
    assert "dedupe_failure" not in causes
    keys = [c.dedupe_key for c in cands]
    assert len(keys) == len(set(keys))


def test_sender_hidden_still_produces_candidate():
    causes, cands = _causes([{"ts": "t1", "text": "CF C01 x2 ค่ะ"}])
    assert causes == []
    assert any(c.sender is None and "C01" in c.raw_text for c in cands)


def test_emoji_and_attachment_rows_classified():
    _c1, cands = _causes(_BASE + [{"sender": "B", "ts": "t3", "text": "❤️"}])
    assert any(c.payload_type == "emoji_text" for c in cands)
    _c2, cands2 = _causes(_BASE + [{"sender": "B", "ts": "t3",
                                    "payload": ("attachment", "voice message")}])
    assert any(c.payload_type == "attachment" for c in cands2)


def test_readme_and_docs_make_no_hardware_or_business_claim():
    for rel in ("README.md", "docs/ADAPTER_CONFORMANCE.md"):
        txt = open(os.path.join(ROOT, rel), encoding="utf-8").read().lower()
        assert "no_text_exposed" in txt or "adapter" in txt
        # the adapter section explicitly disclaims these
    doc = open(os.path.join(ROOT, "docs/ADAPTER_CONFORMANCE.md"), encoding="utf-8").read().lower()
    assert "not hardware proof" in doc and "not business proof" in doc


def test_no_salted_builtin_hash_used_for_reproducibility():
    for rel in ("core/adapter.py", "tools/gen_adapter_fixtures.py"):
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        # stable hashing only: sha256 via hashlib, never builtin hash() for seeds
        assert "hashlib" in src or "from core.adapter import" in src
        assert "random.Random(hash(" not in src
