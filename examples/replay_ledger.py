"""Replay a stored ledger from captures + corrections.

    python examples/replay_ledger.py --store log/operator_demo/ledger.db

Re-derives each seller's ledger from the raw captures and the append-only
correction log, then reports whether the replay matches the stored ledger
snapshot. Proves the ledger is reproducible from the durable record.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ledger_store import LedgerStore
from core.review import replay_ledger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="path to ledger.db")
    args = ap.parse_args()
    if not os.path.exists(args.store):
        print(f"no store at {args.store}", file=sys.stderr)
        sys.exit(2)

    store = LedgerStore(args.store)
    sellers = [r["seller_id"] for r in
               store.db.execute("SELECT seller_id FROM sellers ORDER BY seller_id")]
    all_matched = True
    print(f"Replaying {len(sellers)} sellers from {args.store}\n")
    for sid in sellers:
        _replayed, matched = replay_ledger(store, sid)
        all_matched = all_matched and matched
        print(f"  {sid:22} replay matched snapshot: {matched}")
    print(f"\nreplay matched (all sellers): {all_matched}")
    sys.exit(0 if all_matched else 1)


if __name__ == "__main__":
    main()
