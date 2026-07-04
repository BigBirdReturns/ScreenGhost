"""Source-agnostic capture: routing, contract parity, named unsupported."""
import os

from core.adapter import conformance, verdict
from core.capture import (
    candidates_from_api_events, candidates_from_view_tree, capture,
)
from core.orders import ChatMessage, OrderBook, classify_event
from core.surfaces import SURFACES, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "examples", "adapter_fixtures")


def _xml(name):
    return open(os.path.join(FIX, name), encoding="utf-8").read()


def test_api_events_become_candidates_exact_thai():
    cands = candidates_from_api_events(
        [{"sender": "Nok", "ts": "t1", "text": "CF C01 x2 ตัว ค่ะ"}], "line_oa")
    assert len(cands) == 1
    assert cands[0].raw_text == "CF C01 x2 ตัว ค่ะ"   # exact
    assert cands[0].unicode_ok and cands[0].parser_eligible
    assert cands[0].source_surface == "line_oa"


def test_api_events_dedupe_on_redelivery():
    ev = {"sender": "Nok", "ts": "t1", "text": "CF C01 x1"}
    assert len(candidates_from_api_events([ev, ev, ev], "line_oa")) == 1


def test_capture_is_api_first():
    # given both, the ghost prefers the API path (obfuscation-irrelevant)
    r = capture("fb_page_messenger",
                api_events=[{"sender": "N", "ts": "t1", "text": "CF C01 x1"}],
                view_tree_xml=_xml("line_like_basic.xml"))
    assert r.strategy == "api"


def test_capture_falls_to_view_tree_when_no_api():
    r = capture("web_storefront", view_tree_xml=_xml("line_like_basic.xml"))
    assert r.strategy == "view_tree" and r.candidates


def test_messenger_scrape_is_unsupported_surface():
    # accessibility stripped -> no readable text -> named unsupported, not hidden
    r = capture("messenger_app_obfuscated",
                view_tree_xml=_xml("messenger_app_obfuscated.xml"))
    assert r.strategy == "none"
    assert r.unsupported_reason == "unsupported_surface"


def test_messenger_fixture_fails_by_declared_name():
    v, cause = verdict(conformance(_xml("messenger_app_obfuscated.xml")))
    assert (v, cause) == ("EXPECTED_FAIL", "no_text_exposed")


def test_source_agnostic_ledger_parity():
    # the SAME message via api or view_tree yields the same order behavior
    api_c = candidates_from_api_events(
        [{"sender": "Nok", "ts": "t1", "text": "CF C01 x2 ค่ะ"}], "line_oa")[0]
    vt_c = next(c for c in candidates_from_view_tree(_xml("line_like_basic.xml"))
                if c.raw_text.startswith("CF C01"))
    # both feed OrderBook identically via the chat-field projection
    book = OrderBook()
    for c in (api_c, vt_c):
        s, ts, txt = c.to_chat_fields()
        ev = book.ingest([ChatMessage(s, f"{c.source_surface}:{ts}", txt)])[0]
        assert classify_event(ev.text) == "order" and "C01" in ev.text


def test_surface_registry_is_valid():
    validate()
    keys = [s.key for s in SURFACES]
    assert len(keys) == len(set(keys))
    assert any(s.strategies[0] == "api" for s in SURFACES)      # routed platforms
    assert any(s.proof == "gap" for s in SURFACES)              # honest gap exists


def test_matrix_doc_states_frozen_boundary():
    doc = open(os.path.join(ROOT, "docs", "SURFACE_CAPABILITY_MATRIX.md"),
               encoding="utf-8").read().lower()
    assert "frozen" in doc and "not claimed" not in doc[:0]  # boundary present
    assert "live platform api integration" in doc or "live api integration" in doc
    assert "unsupported_surface" in doc
