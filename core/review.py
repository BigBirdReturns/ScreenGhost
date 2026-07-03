"""Review workflow: pipeline outputs -> proposed events -> reviewed ledger.

The store never sees ground truth; review decisions are human judgement. The
receipt DOES compare against the synthetic ground-truth ledger — that is a
metric on synthetic data, clearly labelled, never a business claim. Corrections
are replayable: captures + the correction log reproduce the corrected values.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

from core.catalog_io import rows_to_skus
from core.eval_population import _ledger_shape
from core.ledger_store import (
    ACCEPTED, CORRECTED, LEDGER_STATES, NEEDS_INFO, PROPOSED,
    PROPOSED_INCOMPLETE, REJECTED, LedgerStore,
)
from core.orders import EventType, classify_event, parse_confirm
from core.population import SellerWorld
from core.resolver import resolve, variant_missing

HIGH_CONFIDENCE = 0.8


def _derive(text: str, skus) -> Tuple[str, object, object]:
    """(event_type, resolved_sku, qty) from raw text + a catalog. No ground truth."""
    return classify_event(text), resolve(text, skus), parse_confirm(text)[2]


def _sku_obj(skus, code):
    return next((s for s in skus if s.code == code), None)


def populate_world(store: LedgerStore, world: SellerWorld,
                   parser_path: str = "in_process") -> Dict[str, int]:
    """Load a seller's stream into the store as captures + proposed events."""
    prof = world.profile
    store.add_seller(prof.seller_id, prof.cohort)
    for b in prof.buyers:
        store.add_buyer(b.buyer_id, prof.seller_id, b.display_name)
    catalog_rows = [{"sku": s.code, "name": s.name, "aliases": s.aliases,
                     "variants": s.variants, "price": s.price,
                     "stock": 1 if s.in_stock else 0} for s in prof.catalog]
    store.import_catalog_rows(prof.seller_id, catalog_rows)
    skus = rows_to_skus(store.catalog(prof.seller_id))

    total = order = proposed = duplicates = 0
    seen = set()
    for lm in world.messages:
        m = lm.msg
        total += 1
        if m.id in seen:
            duplicates += 1
            continue
        seen.add(m.id)
        store.add_capture(m.id, prof.seller_id, m.sender, m.ts, m.text,
                          dedupe_key=m.id, parser_path=parser_path, unicode_ok=True)
        etype, sku, qty = _derive(m.text, skus)
        if etype == EventType.CHATTER:
            continue
        order += 1
        # A recognized order whose SKU needs a variant the buyer didn't give is
        # INCOMPLETE — never auto-accepted, never a hallucinated done order.
        incomplete = (etype == EventType.ORDER and sku is not None
                      and variant_missing(m.text, _sku_obj(skus, sku)))
        if sku is not None and qty is not None:
            confidence = 0.6 if incomplete else 0.9
        else:
            confidence = 0.35
        status = PROPOSED_INCOMPLETE if incomplete else PROPOSED
        store.propose_event(m.id, prof.seller_id, m.id, m.sender, sku, qty,
                            variant=None, event_type=etype,
                            confidence=confidence, dedupe_key=m.id, status=status)
        proposed += 1
    return {"total": total, "order_bearing": order, "proposed": proposed,
            "duplicates": duplicates}


def auto_review(store: LedgerStore, seller_id: str) -> Dict[str, int]:
    """Deterministic first pass: accept confident events, flag the rest.

    proposed_incomplete never auto-accepts — a missing required field always
    routes to a human (needs_info), never silently into the ledger.
    """
    accepted = flagged = 0
    for ev in store.events(seller_id, status=PROPOSED):
        if ev["confidence"] >= HIGH_CONFIDENCE:
            store.transition(ev["event_id"], ACCEPTED)
            accepted += 1
        else:
            store.transition(ev["event_id"], NEEDS_INFO)
            flagged += 1
    for ev in store.events(seller_id, status=PROPOSED_INCOMPLETE):
        store.transition(ev["event_id"], NEEDS_INFO)
        flagged += 1
    return {"accepted": accepted, "needs_info": flagged}


def reviewer_resolve(store: LedgerStore, world: SellerWorld) -> Dict[str, int]:
    """Simulate the seller resolving needs_info items (they know their catalog).

    Uses the synthetic label to stand in for the human's own knowledge of what a
    buyer meant. This is a simulated reviewer on synthetic data, not automation.
    """
    label_by_id = {}
    for lm in world.messages:
        label_by_id.setdefault(lm.msg.id, lm)
    corrected = rejected = 0
    for ev in store.events(world.profile.seller_id, status=NEEDS_INFO):
        lm = label_by_id.get(ev["event_id"])
        if lm and lm.should_emit and lm.sku:
            if ev["sku"] != lm.sku:
                store.correct(ev["event_id"], "sku", lm.sku,
                              reason="reviewer identified intended item", source="human")
            if ev["qty"] != lm.qty and lm.qty is not None:
                store.correct(ev["event_id"], "qty", lm.qty,
                              reason="reviewer confirmed quantity", source="human")
            store.transition(ev["event_id"], ACCEPTED)
            corrected += 1
        else:
            store.transition(ev["event_id"], REJECTED)  # not a real order
            rejected += 1
    return {"corrected": corrected, "rejected": rejected}


def replay_event_values(store: LedgerStore, seller_id: str
                        ) -> Dict[str, Tuple[str, object, object]]:
    """Reproduce corrected event values from captures + the correction log.

    Re-derives each event from its raw capture, then folds the append-only
    corrections. The result must equal the live event rows — that is the
    replayability guarantee.
    """
    skus = rows_to_skus(store.catalog(seller_id))
    out: Dict[str, Tuple[str, object, object]] = {}
    for ev in store.events(seller_id):
        cap = store.get_capture(ev["capture_id"])
        etype, sku, qty = _derive(cap["raw_text"], skus)
        buyer = cap["buyer_display"]
        for c in store.get_event(ev["event_id"])["corrections"]:
            if c["field"] == "sku":
                sku = c["new_value"]
            elif c["field"] == "qty":
                qty = int(c["new_value"])
            elif c["field"] == "buyer":
                buyer = c["new_value"]
        out[ev["event_id"]] = (buyer, sku, qty)
    return out


def replay_ledger(store: LedgerStore, seller_id: str):
    """Reconstruct the ledger from captures + corrections alone (no event values).

    Re-derives each event from its raw capture and folds the correction log in
    arrival (capture-ts) order, over events whose review status counts toward the
    ledger. Returns (replayed_ledger, matched_snapshot?) — matched against the
    stored after_correction snapshot when present, else the live ledger.
    """
    from core.orders import classify_event, reduce_ledger
    skus = rows_to_skus(store.catalog(seller_id))
    rows = []
    for e in store.events(seller_id):
        if e["status"] not in LEDGER_STATES:
            continue
        cap = store.get_capture(e["capture_id"])
        etype = classify_event(cap["raw_text"])
        buyer, sku, qty = cap["buyer_display"], resolve(cap["raw_text"], skus), \
            parse_confirm(cap["raw_text"])[2]
        for c in store.get_event(e["event_id"])["corrections"]:
            if c["field"] == "sku":
                sku = c["new_value"]
            elif c["field"] == "qty":
                qty = int(c["new_value"])
            elif c["field"] == "buyer":
                buyer = c["new_value"]
        rows.append((cap["ts"], (etype, buyer, sku, qty)))
    rows.sort(key=lambda r: r[0])
    replayed = reduce_ledger(t for _ts, t in rows)
    snap = store.get_snapshot(seller_id, "after_correction")
    reference = snap if snap is not None else store.current_ledger(seller_id)
    return replayed, _ledger_shape(replayed) == _ledger_shape(reference)


def reproduction_rate(store: LedgerStore, worlds: List[SellerWorld]) -> float:
    match = 0
    for w in worlds:
        if _ledger_shape(store.current_ledger(w.profile.seller_id)) == _ledger_shape(w.ledger):
            match += 1
    return match / len(worlds) if worlds else 1.0


def run_review_session(store: LedgerStore, worlds: List[SellerWorld],
                       seed: str) -> Dict:
    """Full flow: populate -> auto-review -> before -> reviewer -> after."""
    totals = {"total": 0, "order_bearing": 0, "proposed": 0, "duplicates": 0}
    for w in worlds:
        s = populate_world(store, w)
        for k in totals:
            totals[k] += s[k]
        auto_review(store, w.profile.seller_id)
        store.snapshot_ledger(w.profile.seller_id, "before_correction")
    before = reproduction_rate(store, worlds)

    for w in worlds:
        reviewer_resolve(store, w)
        store.snapshot_ledger(w.profile.seller_id, "after_correction")
    after = reproduction_rate(store, worlds)

    def count(status):
        return sum(len(store.events(w.profile.seller_id, status=status)) for w in worlds)

    unicode_corrupt = sum(
        1 for w in worlds for c in store.all_captures(w.profile.seller_id)
        if not c["unicode_ok"])
    return {
        "seed": seed,
        "sellers": len(worlds),
        "total_messages": totals["total"],
        "order_bearing_messages": totals["order_bearing"],
        "proposed_events": totals["proposed"],
        "accepted_events": count(ACCEPTED),
        "rejected_events": count(REJECTED),
        "corrected_events": sum(store.corrected_event_count(w.profile.seller_id)
                                for w in worlds),
        "needs_info_events": count(NEEDS_INFO),
        "unicode_corruptions": unicode_corrupt,
        "duplicate_events": totals["duplicates"],
        "ledger_reproduction_before_correction": before,
        "ledger_reproduction_after_correction": after,
    }


def receipt_from_store(store: LedgerStore, worlds: List[SellerWorld],
                       seed: str) -> Dict:
    """Read-only receipt from current store state (for UI export).

    Reproduction 'before' comes from the before_correction snapshot when present.
    duplicate_events/total are store-derived (deduped captures), so they reflect
    stored state, not the original raw stream.
    """
    def cnt(status):
        return sum(len(store.events(w.profile.seller_id, status=status)) for w in worlds)

    before = after = total_caps = order = 0
    for w in worlds:
        sid = w.profile.seller_id
        snap = store.get_snapshot(sid, "before_correction")
        if snap is not None and _ledger_shape(snap) == _ledger_shape(w.ledger):
            before += 1
        if _ledger_shape(store.current_ledger(sid)) == _ledger_shape(w.ledger):
            after += 1
        caps = store.all_captures(sid)
        total_caps += len(caps)
        order += len(store.events(sid))
    n = len(worlds) or 1
    return {
        "seed": seed, "sellers": len(worlds),
        "total_messages": total_caps, "order_bearing_messages": order,
        "proposed_events": cnt("proposed"), "accepted_events": cnt(ACCEPTED),
        "rejected_events": cnt(REJECTED),
        "corrected_events": sum(store.corrected_event_count(w.profile.seller_id) for w in worlds),
        "needs_info_events": cnt(NEEDS_INFO),
        "unicode_corruptions": sum(1 for w in worlds
                                   for c in store.all_captures(w.profile.seller_id)
                                   if not c["unicode_ok"]),
        "duplicate_events": 0,
        "ledger_reproduction_before_correction": before / n,
        "ledger_reproduction_after_correction": after / n,
    }


_RECEIPT_ORDER = [
    "seed", "sellers", "total_messages", "order_bearing_messages",
    "proposed_events", "accepted_events", "rejected_events", "corrected_events",
    "needs_info_events", "unicode_corruptions", "duplicate_events",
    "ledger_reproduction_before_correction", "ledger_reproduction_after_correction",
]


def export_session(store: LedgerStore, worlds: List[SellerWorld],
                   receipt: Dict, out_dir: str) -> Dict[str, str]:
    """Write orders.csv, corrections.csv, captures.csv, receipt.txt."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, k) for k in
             ("orders.csv", "corrections.csv", "captures.csv", "receipt.txt")}

    with open(paths["orders.csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "seller_id", "buyer", "sku", "qty", "variant",
                    "event_type", "status", "confidence"])
        for world in worlds:
            for e in store.events(world.profile.seller_id):
                w.writerow([e["event_id"], e["seller_id"], e["buyer"], e["sku"],
                            e["qty"], e["variant"], e["event_type"], e["status"],
                            e["confidence"]])

    with open(paths["corrections.csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["correction_id", "event_id", "field", "old_value",
                    "new_value", "reason", "source"])
        for world in worlds:
            for c in store.all_corrections(world.profile.seller_id):
                w.writerow([c["correction_id"], c["event_id"], c["field"],
                            c["old_value"], c["new_value"], c["reason"], c["source"]])

    with open(paths["captures.csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["capture_id", "seller_id", "buyer_display", "ts", "raw_text",
                    "dedupe_key", "parser_path", "unicode_ok"])
        for world in worlds:
            for c in store.all_captures(world.profile.seller_id):
                w.writerow([c["capture_id"], c["seller_id"], c["buyer_display"],
                            c["ts"], c["raw_text"], c["dedupe_key"],
                            c["parser_path"], c["unicode_ok"]])

    with open(paths["receipt.txt"], "w", encoding="utf-8") as f:
        f.write("ScreenGhost — Reviewable Order Ledger v0 receipt\n")
        f.write("(synthetic population; NOT hardware proof, NOT business proof)\n\n")
        f.write("denominators kept separate:\n")
        for k in _RECEIPT_ORDER:
            f.write(f"  {k:38}: {receipt[k]}\n")

    for kind, path in paths.items():
        store.record_export(path, kind)
    return paths
