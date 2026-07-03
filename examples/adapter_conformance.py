"""Adapter conformance verifier.

    python examples/adapter_conformance.py --fixture examples/adapter_fixtures/line_like_basic.xml
    python examples/adapter_conformance.py --all

Runs the candidate contract against a fixture (or all), prints the per-check
results and verdict, and exits nonzero on any undeclared failure, undeclared
pass of an EXPECTED_FAIL fixture, or multi-cause attribution.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adapter import conformance, verdict

FIX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "examples", "adapter_fixtures")

ALLOWED = "adapter candidate conformance on a fixture surface (exact text, grouping, dedupe, payloads, window)"
FORBIDDEN = "hardware proof, live-seller proof, business proof"


def _pf(ok):
    return "PASS" if ok else "FAIL"


def one(path, quiet=False):
    xml = open(path, encoding="utf-8").read()
    res = conformance(xml)
    v, cause = verdict(res)
    if not quiet:
        m = res.meta
        print(f"fixture              : {os.path.basename(path)}")
        print(f"fixture_hash         : {res.fixture_hash}")
        print(f"expected_ledger_hash : {res.expected_ledger_hash}")
        print(f"computed_ledger_hash : {res.computed_ledger_hash}")
        print(f"node_count           : {res.node_count}")
        print(f"text_node_count      : {res.text_node_count}")
        print(f"thai_text_node_count : {res.thai_text_node_count}")
        print(f"unicode_corruptions  : {res.unicode_corruptions}")
        print(f"candidate_count      : {len(res.candidates)}")
        print(f"expected_candidates  : {len(m.get('expected_candidates', []))}")
        print(f"row_grouping         : {_pf(res.grouping_ok)}")
        print(f"dedupe               : {_pf(res.dedupe_ok)}")
        print(f"payload_classify     : {_pf(res.payloads_ok)}")
        print(f"visible_window_loss  : {res.window_loss}")
        print(f"verdict              : {v}" + (f"  ({cause})" if cause else ""))
        print(f"ALLOWED              : {ALLOWED}")
        print(f"FORBIDDEN            : {FORBIDDEN}")
    return v, cause, res


# verdicts that are acceptable (declared); anything else is a hard failure
_OK = {"PASS", "EXPECTED_FAIL"}


def run_all():
    paths = sorted(glob.glob(os.path.join(FIX_DIR, "*.xml")))
    hdr = f"{'fixture':38} {'unicode':>7} {'group':>6} {'dedupe':>6} {'payloads':>8} {'window':>6} {'verdict':>14}"
    print(hdr)
    print("-" * len(hdr))
    bad = 0
    for p in paths:
        v, cause, res = one(p, quiet=True)
        if v not in _OK:
            bad += 1
        print(f"{os.path.basename(p):38} {_pf(res.unicode_corruptions == 0):>7} "
              f"{_pf(res.grouping_ok):>6} {_pf(res.dedupe_ok):>6} {_pf(res.payloads_ok):>8} "
              f"{res.window_loss:>6} {v:>14}" + (f" ({cause})" if cause else ""))
    print(f"\n{len(paths)} fixtures, {bad} undeclared failure(s).")
    print(f"FORBIDDEN: {FORBIDDEN}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixture")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        sys.exit(1 if run_all() else 0)
    v, _cause, _res = one(args.fixture)
    sys.exit(0 if v in _OK else 1)


if __name__ == "__main__":
    main()
