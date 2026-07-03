"""Rendered seller-world parity: category [2a] device-seam receipt.

    python examples/parity_bench.py [n]

Renders the synthetic population into device-format UiAutomator XML and recovers
the ledger through the real view-tree extraction code, then compares to the
in-process result. A near-zero delta means the seam is clean and the remaining
failures are parser/resolver logic, not capture.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.android_fixture import (
    RenderParams, capture_world, run_parity, score_capture,
)
from core.population import build_population

BANNER = """\
EVIDENTIARY CATEGORIES
  [1] pipeline    — in-process synthetic population           (proved earlier)
  [2a] view-tree  — SAME population rendered to UiAutomator XML and recovered
                    through parse_ui_dump + geometric row grouping   (THIS bench)
  [2b] hardware   — real adb/uiautomator on a real LINE/FB app   (FROZEN: no
                    device in this environment; not claimed)
  [3] business    — seller-hour lift from a live seller           (FROZEN)
No OCR. No live accounts. Synthetic sellers are not live sellers.
A parity delta of ~0 means failures are parser/resolver, not the capture seam.
"""


def parity_table(rows) -> None:
    print("RENDERED SELLER-WORLD PARITY  [1] in-process  vs  [2a] view-tree seam")
    hdr = (f"{'cohort':13} {'sell':>4} {'ip_recall':>9} {'seam_recall':>11} "
           f"{'d':>5} {'ip_prec':>8} {'seam_prec':>9} {'corrupt':>7} {'seam_ledgr':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        d = (r.seam_recall - r.inproc_recall) * 100
        print(f"{r.cohort:13} {r.sellers:4d} {r.inproc_recall*100:8.1f}% "
              f"{r.seam_recall*100:10.1f}% {d:+5.1f} {r.inproc_precision*100:7.1f}% "
              f"{r.seam_precision*100:8.1f}% {r.corruptions:7d} {r.seam_ledger_rate*100:9.0f}%")


def drift_table(worlds) -> None:
    print("\nDRIFT / HOSTILITY  [does the seam survive display change?]")
    w = next(x for x in worlds if x.profile.cohort == "bakery")
    se = sum(1 for lm in w.messages if lm.should_emit)
    print(f"{'condition':26} {'recall':>7} {'corrupt':>8} {'ledger':>7}")
    print("-" * 50)
    conds = [
        ("baseline", RenderParams()),
        ("font 1.5x", RenderParams(font_scale=1.5)),
        ("font 0.85x", RenderParams(font_scale=0.85)),
        ("narrow bubble 280px", RenderParams(bubble_width=280)),
        ("dark mode", RenderParams(dark=True)),
        ("small display 720w", RenderParams(display_w=720)),
        ("PATHOLOGICAL row overlap", RenderParams(row_h_base=15)),
    ]
    for label, p in conds:
        sc = score_capture(w, capture_world(w, p))
        rec = sc["tp"] / se if se else 1.0
        print(f"{label:26} {rec*100:6.1f}% {sc['corruptions']:8d} {str(sc['ledger_match']):>7}")


def boundary(worlds) -> None:
    print("\nSEAM BOUNDARY  [must still fail openly past the visible window]")
    # burst arrives ~6 comments per poll interval; a 2-row window can't hold them.
    w = next((x for x in worlds if x.profile.traffic == "burst"),
             next(x for x in worlds if x.profile.traffic == "hot"))
    se = sum(1 for lm in w.messages if lm.should_emit)
    for win in (400, 2):
        sc = score_capture(w, capture_world(w, window=win))
        rec = sc["tp"] / se if se else 1.0
        print(f"  window={win:<4} recall={rec*100:6.1f}%  "
              f"({'keeps up' if sc['tp'] == se else 'scroll-loss, reported'})")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(BANNER)
    worlds = build_population(n=n, seed=1337)
    parity_table(run_parity(worlds))
    drift_table(worlds)
    boundary(worlds)


if __name__ == "__main__":
    main()
