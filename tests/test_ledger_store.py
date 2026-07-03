"""Local store invariants: schema, immutable captures, states, corrections."""
import pytest

from core.ledger_store import (
    ACCEPTED, LedgerStore, PROPOSED, REJECTED, TransitionError,
)


def _store_with_event():
    s = LedgerStore(":memory:")
    s.add_seller("s1", "bakery")
    s.add_capture("c1", "s1", "Nok", "t00001", "CF A01 x1 ค่ะ", "c1", "in_process")
    s.propose_event("c1", "s1", "c1", "Nok", "A01", 1, None, "order", 0.9, "c1")
    return s


def test_schema_creation():
    s = LedgerStore(":memory:")
    tables = {r[0] for r in s.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("sellers", "buyers", "catalog_items", "captures", "order_events",
              "order_event_sources", "corrections", "ledger_snapshots", "exports"):
        assert t in tables


def test_raw_capture_is_immutable():
    s = LedgerStore(":memory:")
    s.add_seller("s1")
    s.add_capture("c1", "s1", "Nok", "t1", "CF A01 x1", "c1", "in_process")
    s.add_capture("c1", "s1", "Nok", "t1", "TAMPERED", "c1", "in_process")  # same id
    assert s.get_capture("c1")["raw_text"] == "CF A01 x1"


def test_thai_preserved_through_storage():
    s = LedgerStore(":memory:")
    s.add_seller("s1")
    s.add_capture("c1", "s1", "Nok", "t1", "CF บี12 x3 ชิ้น ค่ะ", "c1", "x")
    assert s.get_capture("c1")["raw_text"] == "CF บี12 x3 ชิ้น ค่ะ"


def test_state_transitions_enforced():
    s = _store_with_event()
    s.transition("c1", ACCEPTED)
    with pytest.raises(TransitionError):        # accepted -> proposed is illegal
        s.transition("c1", PROPOSED)


def test_rejected_is_terminal():
    s = _store_with_event()
    s.transition("c1", REJECTED)
    with pytest.raises(TransitionError):
        s.transition("c1", ACCEPTED)


def test_corrections_append_only_and_leave_capture_untouched():
    s = LedgerStore(":memory:")
    s.add_seller("s1")
    s.add_capture("c1", "s1", "Nok", "t1", "cf item x1", "c1", "x")
    s.propose_event("c1", "s1", "c1", "Nok", None, 1, None, "order", 0.35, "c1")
    s.correct("c1", "sku", "A01", "reviewer knew item", "human")
    s.correct("c1", "qty", 2, "reviewer confirmed qty", "human")
    ev = s.get_event("c1")
    assert ev["sku"] == "A01" and ev["qty"] == 2 and ev["status"] == "corrected"
    assert len(ev["corrections"]) == 2                     # append-only history
    assert s.get_capture("c1")["raw_text"] == "cf item x1"  # raw fact preserved


def test_uncorrectable_field_rejected():
    s = _store_with_event()
    with pytest.raises(TransitionError):
        s.correct("c1", "raw_text", "hax", "no", "human")
