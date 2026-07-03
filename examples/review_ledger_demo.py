"""Reviewable Order Ledger v0 — the first usable seller workflow.

    python examples/review_ledger_demo.py --seed "op" --sellers 10
    python examples/review_ledger_demo.py --seed "op" --sellers 10 --serve

Generates a small seller world, runs the pipeline into a local SQLite store,
reviews it (auto + a simulated reviewer), exports the ledger, and prints where
the review UI lives. `--serve` leaves auto-review in place and opens the UI for
manual accept/reject/correct. No phone, no live app, no [2b].
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ledger_store import LedgerStore
from core.population import build_population
from core.review import (
    auto_review, export_session, populate_world, receipt_from_store,
    run_review_session,
)
from core.review_server import make_server


def seed_to_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--sellers", type=int, default=10)
    ap.add_argument("--out", default="log/review")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    worlds = build_population(n=args.sellers, seed=seed_to_int(args.seed))
    store = LedgerStore(os.path.join(args.out, "ledger.db"))

    print(f"Reviewable Order Ledger v0  (seed={args.seed!r}, sellers={args.sellers})")
    print("NOT hardware proof. NOT business proof. First usable workflow on the proof spine.\n")

    if args.serve:
        # populate + first-pass auto-review only; leave the rest for the human UI
        for w in worlds:
            populate_world(store, w)
            auto_review(store, w.profile.seller_id)
            store.snapshot_ledger(w.profile.seller_id, "before_correction")
        srv = make_server(store, worlds, port=args.port, export_dir=args.out)
        host, port = srv.server_address
        print(f"Review UI: http://{host}:{port}/   (Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
        return

    receipt = run_review_session(store, worlds, seed=args.seed)
    paths = export_session(store, worlds, receipt, args.out)
    print("Receipt (denominators separate):")
    for k, v in receipt.items():
        print(f"  {k:38}: {v}")
    print("\nExports:")
    for kind, path in paths.items():
        print(f"  {kind:16}: {path}")
    print(f"\nReview UI available with --serve (http://127.0.0.1:{args.port}/)")


if __name__ == "__main__":
    main()
