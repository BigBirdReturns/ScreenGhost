"""Print the keep-up receipt across arrival rates.

    python examples/keepup_bench.py

Simulation proves the pipeline (recall, latency, dedup, keep-up boundary).
One live seller proves the claim. This is the first half — the number that
answers "it can't keep up at real volume."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingest import simulate_keepup

# (label, comments/sec, poll interval s, visible window, duration s)
SCENARIOS = [
    ("busy stream   (3k/min)", 50, 0.30, 120, 30),
    ("hot sale     (12k/min)", 200, 0.30, 400, 15),
    ("firehose  (past window)", 1000, 0.30, 100, 3),
]

HEAD = f"{'scenario':26} {'recall':>7} {'p95 lat':>8} {'corrupt':>8} {'head':>6} {'keeps up':>9}"


def main() -> None:
    print(HEAD)
    print("-" * len(HEAD))
    for label, rate, interval, window, dur in SCENARIOS:
        r = simulate_keepup(rate, interval, window, dur)
        head = "inf" if r.headroom == float("inf") else f"{r.headroom:.1f}x"
        print(f"{label:26} {r.order_recall*100:6.1f}% {r.p95_latency_s:7.2f}s "
              f"{r.unicode_corruptions:8d} {head:>6} {str(r.keeps_up):>9}")


if __name__ == "__main__":
    main()
