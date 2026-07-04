"""Surface Capability Matrix — print the decided capture path per surface.

    python examples/capture_matrix.py

Shows, per surface, the preferred capture strategy and a live routing demo: an
API surface captures from a representative event payload; a view-tree surface
from a fixture; the obfuscated Messenger scrape resolves to unsupported_surface.
The ghost inherits the user's access; the same candidate contract flows to the
ledger regardless of source. Not hardware/live-seller/business proof.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.capture import capture
from core.surfaces import SURFACES

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "examples", "adapter_fixtures")

# representative authenticated-session events (the shape a webhook yields)
_API_EVENTS = [
    {"sender": "Nok", "ts": "t1", "text": "CF C01 x2 ค่ะ"},
    {"sender": "Ann", "ts": "t2", "text": "เอาเซรั่ม 2 ขวดค่ะ"},
    {"sender": "Aum", "ts": "t3", "text": "ยกเลิกก่อนนะคะ"},
]


def _demo(surface):
    if surface.strategies[0] == "api":
        return capture(surface.key, api_events=_API_EVENTS)
    if surface.key == "messenger_app_obfuscated":
        xml = open(os.path.join(FIX, "messenger_app_obfuscated.xml"), encoding="utf-8").read()
        return capture(surface.key, view_tree_xml=xml)
    xml = open(os.path.join(FIX, "line_like_basic.xml"), encoding="utf-8").read()
    return capture(surface.key, view_tree_xml=xml)


def main():
    print("SURFACE CAPABILITY MATRIX — the ghost inherits the user's access")
    print("api-first -> view_tree -> vision -> unsupported_surface (named)\n")
    hdr = f"{'surface':30} {'prefer':>10} {'proof':>13} {'routed→':>10} {'candidates':>11}"
    print(hdr)
    print("-" * len(hdr))
    for s in SURFACES:
        r = _demo(s)
        n = r.unsupported_reason or f"{len(r.candidates)}"
        print(f"{s.label:30} {s.strategies[0]:>10} {s.proof:>13} "
              f"{r.strategy:>10} {n:>11}")
    print("\nNOT hardware/live-seller/business proof. Live API integration is a "
          "frozen [2b] item; api rows are decided strategy + event schema.")
    print("Messenger scrape fails by name; the routed answer is the Page API — "
          "see docs/SURFACE_CAPABILITY_MATRIX.md")


if __name__ == "__main__":
    main()
