"""Reproducibility check: re-run a saved receipt's seed and compare.

    python examples/verify_demo_receipt.py --receipt examples/receipts/operator_demo_seed_op.txt

Parses the saved receipt, re-runs the same seed + seller count, compares the
denominator counts and export existence, runs replay, and reports match or
mismatch. Wall-clock time is not compared. Exit 0 = match, 1 = mismatch.
"""
import argparse
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.operator_demo import DemoError, run

# denominators that must reproduce exactly for the same seed
_COMPARED = [
    "sellers", "total_messages", "order_bearing_messages", "proposed_events",
    "proposed_incomplete_events", "accepted_events", "corrected_events",
    "rejected_events", "needs_info_events", "cancelled_events",
    "fulfilled_events", "unicode_corruptions", "duplicate_events",
    "replay_matched", "schema_version",
]


def parse_receipt(path: str) -> dict:
    if not os.path.exists(path):
        raise DemoError(f"missing receipt: {path}")
    fields = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*([a-z_]+)\s*:\s*(.+?)\s*$", line)
        if m and m.group(1) in _COMPARED + ["seed", "source"]:
            fields[m.group(1)] = m.group(2)
    if "seed" not in fields:
        raise DemoError(f"receipt has no seed field: {path}")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    args = ap.parse_args()
    try:
        saved = parse_receipt(args.receipt)
    except DemoError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    seed = saved["seed"]
    sellers = int(saved.get("sellers", "10"))
    mismatches = []
    with tempfile.TemporaryDirectory() as tmp:
        receipt, paths, _rp, _store, _worlds = run(seed, sellers, tmp)
        for k in _COMPARED:
            if k in saved and str(receipt.get(k)) != saved[k]:
                mismatches.append(f"{k}: saved={saved[k]} rerun={receipt.get(k)}")
        # check exports while the tempdir still exists
        for kind, p in paths.items():
            if not os.path.exists(p):
                mismatches.append(f"missing export {kind}")
        if str(receipt.get("replay_matched")) != "True":
            mismatches.append("replay did not match on rerun")

    if mismatches:
        print("MISMATCH:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)
    print(f"MATCH: seed {seed!r} ({sellers} sellers) reproduced all "
          f"{len(_COMPARED)} denominators + exports + replay.")
    sys.exit(0)


if __name__ == "__main__":
    main()
