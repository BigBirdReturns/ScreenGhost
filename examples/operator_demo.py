"""Operator Demo — run the Reviewable Order Ledger without a developer narrating.

    python examples/operator_demo.py --seed "op" --sellers 10
    python examples/operator_demo.py --fixture examples/fixtures/seller_world_demo_seed.json
    python examples/operator_demo.py --seed "op" --sellers 10 --serve

Generates (or loads) a small seller world, runs the pipeline into a local ledger
store, reviews it, exports CSVs + receipt, replays, and prints every path plus
the replay command. Not hardware proof, not business proof, not live-seller
proof. Exits cleanly. Runtime artifacts go under --out (ignored by git).
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fixtures import load_seller_worlds
from core.ledger_store import (
    ACCEPTED, CANCELLED, FULFILLED, LedgerStore, NEEDS_INFO, PROPOSED,
    PROPOSED_INCOMPLETE, REJECTED,
)
from core.population import build_population
from core.review import export_session, replay_ledger, run_review_session

PROOF_FORBIDDEN = "hardware proof, business proof, live-seller proof"


class DemoError(RuntimeError):
    """Actionable operator-facing error (never a bare stack trace)."""


def seed_to_int(seed: str) -> int:
    if not seed or not seed.strip():
        raise DemoError("invalid seed: provide a non-empty --seed string.")
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)


def _load_worlds(seed, sellers, fixture):
    if fixture:
        if not os.path.exists(fixture):
            raise DemoError(f"missing fixture: {fixture} does not exist.")
        return load_seller_worlds(fixture), f"fixture:{os.path.basename(fixture)}"
    if sellers < 1:
        raise DemoError("invalid --sellers: must be >= 1.")
    return build_population(n=sellers, seed=seed_to_int(seed)), f"seed:{seed}"


def build_receipt(store, worlds, seed, label, base_receipt):
    def total(status):
        return sum(len(store.events(w.profile.seller_id, status=status)) for w in worlds)
    r = dict(base_receipt)
    r["source"] = label
    r["proposed_incomplete_events"] = total(PROPOSED_INCOMPLETE)
    r["proposed_events"] = total(PROPOSED)
    r["needs_info_events"] = total(NEEDS_INFO)
    r["cancelled_events"] = total(CANCELLED)
    r["fulfilled_events"] = total(FULFILLED)
    r["replay_matched"] = all(replay_ledger(store, w.profile.seller_id)[1] for w in worlds)
    r["schema_version"] = store.schema_version()
    r["proof_claims_forbidden"] = PROOF_FORBIDDEN
    return r


_ORDER = ["seed", "source", "sellers", "total_messages", "order_bearing_messages",
          "proposed_events", "proposed_incomplete_events", "accepted_events",
          "corrected_events", "rejected_events", "needs_info_events",
          "cancelled_events", "fulfilled_events", "unicode_corruptions",
          "duplicate_events", "ledger_reproduction_before_correction",
          "ledger_reproduction_after_correction", "replay_matched",
          "schema_version", "proof_claims_forbidden"]


def write_receipt(receipt, paths, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ScreenGhost — Reviewable Order Ledger (Demo RC0) receipt\n")
        f.write("NOT hardware proof. NOT business proof. NOT live-seller proof.\n\n")
        for k in _ORDER:
            if k in receipt:
                f.write(f"  {k:38}: {receipt[k]}\n")
        f.write("  export_paths:\n")
        for kind, p in paths.items():
            f.write(f"     {kind:16}: {p}\n")


def run(seed="op", sellers=10, out="log/operator_demo", fixture=None):
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        raise DemoError(f"export directory not writable: {out} ({e})")
    worlds, label = _load_worlds(seed, sellers, fixture)
    store = LedgerStore(os.path.join(out, "ledger.db"))
    base = run_review_session(store, worlds, seed=seed)
    receipt = build_receipt(store, worlds, seed, label, base)
    paths = export_session(store, worlds, receipt, out)
    receipt_path = os.path.join(out, "receipt_full.txt")
    write_receipt(receipt, paths, receipt_path)
    return receipt, paths, receipt_path, store, worlds


def main() -> None:
    ap = argparse.ArgumentParser(description="ScreenGhost operator demo")
    ap.add_argument("--seed", default="op")
    ap.add_argument("--sellers", type=int, default=10)
    ap.add_argument("--out", default="log/operator_demo")
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    try:
        receipt, paths, receipt_path, store, worlds = run(
            args.seed, args.sellers, args.out, args.fixture)
    except DemoError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    store_path = os.path.join(args.out, "ledger.db")
    print("ScreenGhost — Operator Demo (Reviewable Order Ledger, RC0)")
    print("NOT hardware proof. NOT business proof. NOT live-seller proof.\n")
    for k in _ORDER:
        if k in receipt:
            print(f"  {k:38}: {receipt[k]}")
    print("\nArtifacts:")
    print(f"  store           : {store_path}")
    for kind, p in paths.items():
        print(f"  {kind:16}: {p}")
    print(f"  receipt         : {receipt_path}")
    print("\nNext:")
    print(f"  replay          : python examples/replay_ledger.py --store {store_path}")
    print(f"  review UI       : python examples/operator_demo.py "
          f"--{'fixture ' + args.fixture if args.fixture else 'seed ' + args.seed} --serve")

    if args.serve:
        from core.review_server import make_server
        srv = make_server(store, worlds, port=args.port, export_dir=args.out)
        host, port = srv.server_address
        print(f"\nserving http://{host}:{port}/  (Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
