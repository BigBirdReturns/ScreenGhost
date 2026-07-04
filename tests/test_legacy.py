"""Legacy surface ladder: green screen -> same Candidate contract, frozen floor."""
import json
import os

from core.capture import STRATEGIES, capture
from core.legacy import (
    candidates_from_screen_buffer, field_value, load_screen_fixture,
    screen_to_order_line,
)
from core.orders import ChatMessage, OrderBook, classify_event
from core.surfaces import SURFACES, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "examples", "legacy_fixtures", "green_screen_order_panel.json")


def _fx():
    return load_screen_fixture(FIX)


def test_green_screen_yields_expected_candidates():
    fx = _fx()
    got = sorted((c.sender, c.raw_text, c.payload_type)
                 for c in candidates_from_screen_buffer(fx))
    exp = sorted((r["sender"], r["text"], r["payload_type"])
                 for r in fx["expected_candidates"])
    assert got == exp


def test_green_screen_text_is_exact():
    cands = candidates_from_screen_buffer(_fx())
    assert cands and all(c.unicode_ok for c in cands)
    # the SKU value survives byte-for-byte, no OCR, no normalization drift
    assert any(c.raw_text == "C01" and c.sender == "ITEM" for c in cands)


def test_grid_grouping_is_exact():
    # ITEM and QTY share row 5 -> identical row band; there is no y-band guess
    cands = {c.sender: c for c in candidates_from_screen_buffer(_fx())}
    assert cands["ITEM"].node_bounds[1] == cands["QTY"].node_bounds[1] == 5 * 20
    assert cands["ITEM"].row_bounds == cands["QTY"].row_bounds


def test_protected_chrome_is_not_a_candidate():
    cands = candidates_from_screen_buffer(_fx())
    texts = {c.raw_text for c in cands}
    assert not any(t.startswith("ORDER ENTRY") for t in texts)   # title
    assert not any("PF1=HELP" in t for t in texts)               # function keys
    # labels appear only as senders, never as bodies
    assert "CUST" in {c.sender for c in cands}
    assert "CUST:" not in texts


def test_field_value_reads_labeled_data():
    fx = _fx()
    assert field_value(fx, "ITEM") == "C01"
    assert field_value(fx, "QTY") == "0002"
    assert field_value(fx, "NONEXISTENT") is None


def test_cross_source_ledger_parity():
    # a mainframe-sourced order reaches the SAME OrderBook a LINE order would
    line = screen_to_order_line(_fx())
    assert line == {"buyer": "NOK RATANA", "text": "CF C01 x2", "screen_id": "ORDR01"}
    book = OrderBook()
    ev = book.ingest([ChatMessage(line["buyer"], "ORDR01:submit", line["text"])])[0]
    assert classify_event(ev.text) == "order" and "C01" in ev.text


def test_registry_has_green_screen_rung():
    validate()
    gs = next(s for s in SURFACES if s.key == "green_screen_3270")
    assert gs.strategies == ("view_tree",) and gs.proof == "fixture"


def test_physical_rung_is_declared_and_frozen():
    assert "physical" in STRATEGIES
    phys = next(s for s in SURFACES if s.key == "physical_actuation")
    assert phys.strategies[0] == "physical"
    assert phys.proof == "gap"   # frozen, unbuilt — named, not hidden


def test_capture_of_physical_surface_is_unsupported_not_pretended():
    # capture() implements api + view_tree only; a physical-only surface with no
    # readable buffer must resolve to the named floor, never a faked pixel path
    r = capture("physical_actuation")
    assert r.strategy == "none" and r.unsupported_reason == "unsupported_surface"


def test_ladder_doc_names_the_three_landmines():
    doc = open(os.path.join(ROOT, "docs", "LEGACY_SURFACE_LADDER.md"),
               encoding="utf-8").read().lower()
    assert "hypothesis" in doc and "oracle" in doc          # docs are a prior
    assert "characterization testing" in doc
    assert "differential testing" in doc
    assert "automata learning" in doc or "l\\*" in doc or "l*" in doc
    assert "read-back" in doc                                # webcam OCR loop
    assert "production" in doc and "frozen" in doc


def test_fixture_is_wellformed_json():
    with open(FIX, encoding="utf-8") as f:
        data = json.load(f)
    assert data["surface_type"] == "green_screen_3270"
    assert data["fields"] and data["expected_candidates"]
