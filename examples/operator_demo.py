"""Operator Demo — run the Reviewable Order Ledger without a developer narrating.

    python examples/operator_demo.py --seed "op" --sellers 10
    python examples/operator_demo.py --fixture examples/fixtures/seller_world_demo_seed.json

Generates (or loads) a small seller world, runs the pipeline into a local ledger
store, reviews it, exports CSVs + receipt, and prints the review URL to open with
--serve. Not hardware proof, not business proof. Exits cleanly.
"""
import argparse
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fixtures import load_seller_worlds
from core.ledger_store import (
    ACCEPTED, CANCELLED, CORRECTED, FULFILLED, NEEDS_INFO, PROPOSED,
    PROPOSED_INCOMPLETE, REJECTED, LedgerStore,
)
from core.population import build_population
from core.review import export_session, replay_ledger, run_review_session


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-") or "seed"


def seed_to_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)


def run(seed, sellers, out, fixture=None):
    os.makedirs(out, exist_ok=True)
    if fixture:
        worlds = load_seller_worlds(fixture)
        label = f"fixture:{os.path.basename(fixture)}"
    else:
        worlds = build_population(n=sellers, seed=seed_to_int(seed))
        label = f"seed:{seed}"
    store = LedgerStore(os.path.join(out, "ledger.db"))
    receipt = run_review_session(store, worlds, seed=seed)

    replay_ok = all(replay_ledger(store, w.profile.seller_id)[1] for w in worlds)
    receipt["replay_matched"] = replay_ok

    def total(status):
        return sum(len(store.events(w.profile.seller_id, status=status)) for w in worlds)
    for st, key in [(PROPOSED, "proposed_events"), (PROPOSED_INCOMPLETE, "proposed_incomplete_events"),
                    (NEEDS_INFO, "needs_info_events"), (CANCELLED, "cancelled_events"),
                    (FULFILLED, "fulfilled_events")]:
        receipt[key] = total(st)

    paths = export_session(store, worlds, receipt, out)
    receipt["export_paths"] = paths

    rslug = _slug(os.path.basename(fixture).replace(".json", "")) if fixture else _slug(seed)
    receipt_file = os.path.join("examples", "receipts", f"operator_demo_seed_{rslug}.txt")
    os.makedirs(os.path.dirname(receipt_file), exist_ok=True)
    with open(receipt_file, "w", encoding="utf-8") as f:
        f.write("ScreenGhost — Operator Demo receipt\n")
        f.write("NOT hardware proof. NOT business proof. Product slice on the proof spine.\n\n")
        f.write(f"source: {label}\n")
        for k, v in receipt.items():
            if k == "export_paths":
                continue
            f.write(f"  {k:40}: {v}\n")
        f.write("  export_paths:\n")
        for kind, p in paths.items():
            f.write(f"     {kind:16}: {p}\n")
    return receipt, paths, receipt_file, store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="operator")
    ap.add_argument("--sellers", type=int, default=10)
    ap.add_argument("--out", default="log/operator_demo")
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    receipt, paths, receipt_file, store = run(args.seed, args.sellers, args.out, args.fixture)
    print("ScreenGhost — Operator Demo (Reviewable Order Ledger v0)")
    print("NOT hardware proof. NOT business proof.\n")
    for k in ("sellers", "total_messages", "order_bearing_messages",
              "proposed_events", "proposed_incomplete_events", "accepted_events",
              "corrected_events", "rejected_events", "needs_info_events",
              "cancelled_events", "fulfilled_events", "unicode_corruptions",
              "duplicate_events", "ledger_reproduction_before_correction",
              "ledger_reproduction_after_correction", "replay_matched"):
        print(f"  {k:40}: {receipt[k]}")
    print("\nExports:")
    for kind, p in paths.items():
        print(f"  {kind:16}: {p}")
    print(f"  receipt         : {receipt_file}")
    print(f"\nReview UI: run with --serve  -> http://127.0.0.1:{args.port}/")

    if args.serve:
        from core.review_server import make_server
        srv = make_server(store, load_seller_worlds(args.fixture) if args.fixture
                          else build_population(n=args.sellers, seed=seed_to_int(args.seed)),
                          port=args.port, export_dir=args.out)
        host, port = srv.server_address
        print(f"\nserving http://{host}:{port}/  (Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
