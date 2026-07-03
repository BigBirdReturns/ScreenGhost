"""Population receipt: the same pipeline against 1,000 labeled seller worlds.

    python examples/population_bench.py [n]

Prints two tables — a per-cohort receipt and a boundary/firehose table — and
keeps the evidentiary categories explicit. Rate columns state their denominator
(total messages vs order-bearing messages) so no one can recast merchant work as
"tokens per minute."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eval_population import run_population
from core.ingest import simulate_keepup
from core.population import build_population

BANNER = """\
EVIDENTIARY CATEGORIES (kept separate; do not conflate)
  [1] PIPELINE proof   — this bench: exact text, dedupe, ledger reproduced
                         across a broad LABELED SYNTHETIC population.
  [2] DEVICE-SEAM proof — NOT here: real Android view-tree capture.
  [3] BUSINESS proof    — NOT here: seller-hour lift from a LIVE seller.
Synthetic sellers are not live sellers. No business lift is claimed.
The parser shares a grammar with the generator, so clean recall is high by
construction — the signal is in the ADVERSARIAL suites, where it must fail open.
"""


def cohort_table(out) -> None:
    hdr = (f"{'cohort':13} {'sellers':>7} {'total/min':>9} {'order/min':>9} "
           f"{'recall':>7} {'prec':>7} {'p95 lat':>8} {'corrupt':>7} "
           f"{'dupevt':>7} {'backlog':>7} {'keeps':>6}")
    print("POPULATION RECEIPT  [category 1: pipeline]")
    print(hdr)
    print("-" * len(hdr))
    for c in out["cohorts"]:
        print(f"{c.cohort:13} {c.sellers:7d} {c.total_msgs_min:9.0f} "
              f"{c.order_msgs_min:9.0f} {c.recall*100:6.1f}% {c.precision*100:6.1f}% "
              f"{c.p95_latency_s:7.2f}s {c.unicode_corruptions:7d} "
              f"{c.dup_event_rate:7.3f} {c.backlog_rate:7.3f} {str(c.keeps_up):>6}")
    print(f"\nledger reproduced exactly (line count + per-SKU units): "
          f"{out['ledger_match_rate']*100:.1f}% of sellers")


def adversarial_table(out) -> None:
    print("\nADVERSARIAL SUITES  [where it must fail in the open]")
    print(f"{'suite':16} {'should-emit':>11} {'recall':>8} {'precision':>10} {'false-pos':>10}")
    print("-" * 58)
    for suite, n, recall, precision, fp in out["adversarial"]:
        print(f"{suite:16} {n:11d} {recall*100:7.1f}% {precision*100:9.1f}% {fp:10d}")


def firehose_table() -> None:
    print("\nBOUNDARY / FIREHOSE  [keep-up geometry, honest failure past window]")
    hdr = (f"{'scenario':16} {'total/min':>9} {'order/min':>9} {'recall':>7} "
           f"{'p95 lat':>8} {'head':>6} {'keeps up':>9}")
    print(hdr)
    print("-" * len(hdr))
    # (label, total msgs/sec, poll interval, window, duration)
    rows = [
        ("busy stream", 50, 0.30, 120, 30),
        ("hot sale", 200, 0.30, 400, 15),
        ("firehose", 1000, 0.30, 100, 3),
    ]
    for label, rate, interval, window, dur in rows:
        r = simulate_keepup(rate, interval, window, dur)
        order_min = r.order_arrived / dur * 60
        head = "inf" if r.headroom == float("inf") else f"{r.headroom:.1f}x"
        print(f"{label:16} {rate*60:9.0f} {order_min:9.0f} {r.order_recall*100:6.1f}% "
              f"{r.p95_latency_s:7.2f}s {head:>6} {str(r.keeps_up):>9}")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(BANNER)
    out = run_population(build_population(n=n, seed=1337))
    cohort_table(out)
    adversarial_table(out)
    firehose_table()


if __name__ == "__main__":
    main()
