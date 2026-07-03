"""Objection receipt — every objection, its status, and its claim boundaries.

    python examples/objection_receipt.py            # static registry view
    python examples/objection_receipt.py --live     # run the cheap checks too

Prints, per objection: current evidence category, latest bench artifact,
pass/partial/architectural/frozen, the allowed claim, the forbidden claim, and
the next required proof. It reads core/objections.py, so it can never disagree
with the matrix or the tests.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.objections import OBJECTIONS, gather_evidence

_MARK = {"PASS": "PASS ✓", "PARTIAL": "PARTIAL ◐", "ARCH": "ARCHITECTURAL ◇",
         "FROZEN": "FROZEN ✕"}

RULES = """\
HARD RULES
  • Denominator discipline: never a rate without saying total msgs vs
    order-bearing msgs vs UI nodes vs emitted events.
  • No category borrows trust: synthetic [1] != view-tree [2a] != device
    [2b-1] != real-app [2b-2] != business [3]. A status is scoped to what ran.
"""


def main() -> None:
    live = "--live" in sys.argv
    evidence = gather_evidence() if live else None
    mode = "LIVE (cheap checks executed)" if live else "STATIC (registry view)"
    print(f"SCREENGHOST OBJECTION RECEIPT — {mode}\n")
    print(RULES)
    for o in OBJECTIONS:
        status = _MARK.get(o.status(evidence), o.status(evidence))
        print(f"[{o.id}]  category {o.category}   status: {status}")
        print(f"   objection : {o.objection}")
        print(f"   answer    : {o.answer}")
        print(f"   bench     : {o.bench}")
        print(f"   receipt   : {o.receipt}")
        print(f"   ALLOWED   : {o.allowed}")
        print(f"   FORBIDDEN : {o.forbidden}")
        print(f"   adapter   : {o.adapter_risk}")
        print(f"   next proof: {o.next_proof}")
        print()
    if live:
        e = evidence
        print("live signals: "
              f"pop_corruptions={e['pop_corrupt']}  seam_corruptions={e['seam_corrupt']}  "
              f"seam_recall_delta={e['seam_delta']:.3f}  "
              f"busy_keeps_up={e['busy_keeps_up']}  firehose_fails_openly={e['firehose_fails']}")


if __name__ == "__main__":
    main()
