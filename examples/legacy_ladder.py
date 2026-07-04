"""Legacy Surface Ladder — the ghost inherits the user's terminal session.

    python examples/legacy_ladder.py

Shows the capture ladder extended to legacy: a decoded mainframe green-screen
buffer becomes the SAME Candidate contract the phone adapters emit, feeds the
SAME OrderBook, and the physical(robot+webcam) rung is declared FROZEN. Not a
live TN3270 connection and not hardware proof — see docs/LEGACY_SURFACE_LADDER.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.legacy import (
    candidates_from_screen_buffer, load_screen_fixture, screen_to_order_line,
)
from core.orders import ChatMessage, OrderBook, classify_event
from core.surfaces import SURFACES

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "examples", "legacy_fixtures", "green_screen_order_panel.json")


def main():
    print("LEGACY SURFACE LADDER — the ghost inherits the user's terminal session")
    print("api -> view_tree -> vision -> physical(robot+webcam) -> unsupported\n")

    fx = load_screen_fixture(FIX)
    cands = candidates_from_screen_buffer(fx)
    print(f"green-screen panel {fx['screen_id']} ({fx['source_app']}) — "
          f"{len(cands)} candidates from the decoded field buffer:")
    print(f"  {'label':6} {'value':20} {'grid':>7} {'exact':>6}")
    for c in cands:
        r, cc = c.node_bounds[1] // 20, c.node_bounds[0] // 10
        print(f"  {(c.sender or ''):6} {c.raw_text:20} {f'{r},{cc}':>7} "
              f"{str(c.unicode_ok):>6}")

    # same candidate contract -> same ledger a LINE order would reach
    line = screen_to_order_line(fx)
    book = OrderBook()
    ev = book.ingest([ChatMessage(line["buyer"], f"{line['screen_id']}:submit",
                                  line["text"])])[0]
    print(f"\nprojected order line: {line['buyer']!r} -> {line['text']!r}")
    print(f"OrderBook event: type={classify_event(ev.text)}  text={ev.text!r}")
    print("  -> a mainframe-sourced order and a LINE-sourced order reach the "
          "identical ledger.")

    print("\nladder routing (core/surfaces.py):")
    for s in SURFACES:
        if s.key in ("green_screen_3270", "physical_actuation"):
            tag = "FROZEN" if s.proof == "gap" else s.proof
            print(f"  {s.label:38} {s.strategies[0]:>9}  [{tag}]")

    print("\nNOT a live TN3270 connection, NOT hardware proof. The green-screen "
          "buffer->contract mapping is exercised against a fixture; the "
          "physical(robot+webcam) rung is declared, frozen, and unbuilt.")
    print("See docs/LEGACY_SURFACE_LADDER.md")


if __name__ == "__main__":
    main()
