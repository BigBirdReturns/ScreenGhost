"""Replay adapter candidates into the Reviewable Order Ledger.

    python examples/adapter_to_ledger_demo.py --fixture examples/adapter_fixtures/line_like_basic.xml

Parses a fixture, produces candidates (adapter), feeds eligible ones through
OrderBook into a local ledger store, exports, replays, and prints a receipt.
The adapter emits candidates; OrderBook emits events. Not hardware/business proof.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adapter import conformance, extract, verdict
from core.ledger_store import LedgerStore
from core.orders import EventType, classify_event, parse_confirm
from core.population import Sku
from core.resolver import resolve
from core.review import (
    auto_review, export_by_sellers, receipt_from_store_ids, replay_ledger,
)

# demo catalog covering the codes/aliases used in the fixtures
DEMO_CATALOG = [
    Sku("C01", "เค้ก", ["cake", "เค้ก"], [], 300), Sku("C02", "คุกกี้", ["cookie"], [], 120),
    Sku("A01", "เซรั่ม", ["serum", "เซรั่ม"], [], 390), Sku("A03", "บลัช", [], [], 200),
    Sku("B02", "กางเกง", [], [], 590), Sku("D01", "กระเป๋า", ["bag"], ["ดำ", "แดง"], 500),
]


def run(fixture_path, out):
    xml = open(fixture_path, encoding="utf-8").read()
    meta, cands, _stats = extract(xml)
    sid = meta["fixture_id"]
    os.makedirs(out, exist_ok=True)
    store = LedgerStore(os.path.join(out, "ledger.db"))
    store.add_seller(sid, meta["surface_type"])
    store.import_catalog_rows(sid, [
        {"sku": s.code, "name": s.name, "aliases": s.aliases, "variants": s.variants,
         "price": s.price, "stock": 1 if s.in_stock else 0} for s in DEMO_CATALOG])

    for c in cands:
        store.add_capture(c.capture_id, sid, c.sender or "?", c.first_seen_at,
                          c.raw_text, dedupe_key=c.dedupe_key,
                          parser_path=f"adapter:{meta['surface_type']}",
                          unicode_ok=c.unicode_ok)
        if not c.parser_eligible:
            continue
        etype = classify_event(c.raw_text)
        if etype == EventType.CHATTER:
            continue
        sku = resolve(c.raw_text, DEMO_CATALOG)
        qty = parse_confirm(c.raw_text)[2]
        store.propose_event(c.capture_id, sid, c.capture_id, c.sender or "?",
                            sku, qty, None, etype,
                            0.9 if (sku and qty) else 0.35, c.dedupe_key)
    auto_review(store, sid)
    store.snapshot_ledger(sid, "after_correction")
    replay_ok = replay_ledger(store, sid)[1]

    receipt = receipt_from_store_ids(store, [sid], seed=sid)
    receipt["replay_matched"] = replay_ok
    receipt["source_surface"] = meta["surface_type"]
    receipt["candidates"] = len(cands)
    receipt["eligible_candidates"] = sum(1 for c in cands if c.parser_eligible)
    paths = export_by_sellers(store, [sid], receipt, out)
    return meta, receipt, paths, replay_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", default="log/adapter_demo")
    args = ap.parse_args()
    if not os.path.exists(args.fixture):
        print(f"error: missing fixture {args.fixture}", file=sys.stderr)
        sys.exit(2)

    v, cause = verdict(conformance(open(args.fixture, encoding="utf-8").read()))
    meta, receipt, paths, replay_ok = run(args.fixture, args.out)
    print(f"adapter -> ledger : {os.path.basename(args.fixture)}")
    print(f"conformance       : {v}" + (f" ({cause})" if cause else ""))
    print("NOT hardware proof. NOT business proof. NOT live-seller proof.\n")
    for k in ("source_surface", "candidates", "eligible_candidates",
              "total_messages", "order_bearing_messages", "accepted_events",
              "needs_info_events", "unicode_corruptions", "duplicate_events",
              "replay_matched"):
        if k in receipt:
            print(f"  {k:24}: {receipt[k]}")
    print("\nExports:")
    for kind, p in paths.items():
        print(f"  {kind:16}: {p}")
    sys.exit(0 if replay_ok else 1)


if __name__ == "__main__":
    main()
