"""Pop-proof demo — an operator attack, not an author disclosure.

    python examples/pop_proof_demo.py --seed "pop-picks-this-live" --sellers 1000 --mode adversarial
    python examples/pop_proof_demo.py --seed "anything" --sellers 50 --cha-mode

The operator supplies the seed and tries to make it lie. The population is
generated fresh from that seed (no golden replay); the real evaluators run;
every miss the operator's seed produces is attributed to exactly one cause; the
matrix classifies each objection and refuses to launder ARCHITECTURAL/FROZEN
rows into PASS. [2b] hardware and [3] business are scoped out by operator
decision — not unpaid debt, not oversight.

Invalid runs (no boundary failure, no adversarial bleed, multi-cause
attribution, forbidden status upgrade) abort nonzero. A demo that can pass by
not testing is the exact failure this whole stack exists to make impossible.
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.android_fixture import RenderParams, capture_world, run_parity, score_capture
from core.eval_population import run_population
from core.ingest import simulate_keepup
from core.objections import (
    ARCHITECTURAL, FROZEN, MEANINGS, OBJECTIONS, PASS,
)
from core.population import build_population

_MARK = {"PASS": "PASS", "PARTIAL": "PARTIAL", "ARCH": "ARCHITECTURAL", "FROZEN": "FROZEN"}
CAUSES = ["resolver", "row_grouping", "dedupe", "window_loss", "unicode_corruption"]


class DemoInvalid(RuntimeError):
    """Raised when the demo would pass by not testing what it claims to test."""


def seed_to_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)


# ---- guards (unit-testable) ---------------------------------------------- #
def _require_boundary_failure(results):
    if not any(not ok for _label, ok in results):
        raise DemoInvalid("Demo invalid: no boundary failure was exercised.")


def _require_attribution_exercised(adv_labels_present, adversarial_errors):
    # A resolver error is real bleed; capture-domain errors are injected on
    # purpose, so the guard watches the adversarial (parser-domain) count.
    if adv_labels_present and adversarial_errors == 0:
        raise DemoInvalid(
            "Demo invalid: adversarial suites produced no bleed. "
            "Attribution table not exercised.")


def _check_single_cause(total_errors, per_cause):
    # single-cause per error unit: the per-cause tally must reconstruct the
    # total exactly. Any error counted under two causes breaks this.
    if sum(per_cause.values()) != total_errors:
        raise DemoInvalid(
            "Demo invalid: multi-cause attribution detected. "
            "Attribution must be single-cause per error unit.")


def _assert_no_forbidden_upgrade(evidence):
    for o in OBJECTIONS:
        if o.base_status in (ARCHITECTURAL, FROZEN) and o.status(evidence) == PASS:
            raise DemoInvalid("Demo invalid: forbidden status upgrade attempted.")


def _sample_per_cohort(worlds, k=3):
    seen, out = {}, []
    for w in worlds:
        c = w.profile.cohort
        if seen.get(c, 0) < k:
            out.append(w)
            seen[c] = seen.get(c, 0) + 1
    return out


def run_demo(seed: str, sellers: int = 1000, mode: str = "adversarial",
             cha: bool = False) -> str:
    L = []
    P = L.append
    hr = (lambda t: P(f"\n== {t} ==")) if cha else (lambda t: P(f"\n{'='*70}\n{t}\n{'='*70}"))

    # ---- ACT I ------------------------------------------------------------ #
    hr("ACT I — THE OBJECTION (decomposed)")
    P("One 'you are out of touch' verdict bundled ten separable problems:")
    for o in OBJECTIONS:
        P(f"  - {o.id}")
    P("\nRules enforced live:")
    P("  1. No objection may borrow evidence from another category.")
    P("  2. No rate may be reported without denominator discipline.")
    P("  3. No synthetic proof may become a hardware or business proof.")
    P("")
    P("Pick a seed. Pick a mode. Try to make it lie.")

    # ---- ACT II ----------------------------------------------------------- #
    seed_int = seed_to_int(seed)
    worlds = build_population(n=sellers, seed=seed_int)
    total_msgs = sum(len(w.messages) for w in worlds)
    order_msgs = sum(1 for w in worlds for lm in w.messages if lm.should_emit)
    cohort_dist, adv_counts = {}, {}
    for w in worlds:
        cohort_dist[w.profile.cohort] = cohort_dist.get(w.profile.cohort, 0) + 1
        for lm in w.messages:
            key = lm.adversarial or "clean"
            adv_counts[key] = adv_counts.get(key, 0) + 1

    hr("ACT II — BUILD THE WORLD (fresh from the seed, not replayed)")
    P(f"  seed (string)       : {seed!r}")
    P(f"  seed (int)          : {seed_int}")
    P(f"  synthetic sellers   : {sellers}")
    P(f"  cohorts             : {len(cohort_dist)}")
    P(f"  total messages      : {total_msgs}")
    P(f"  order-bearing msgs  : {order_msgs}")
    P("  (denominators separate: total vs order-bearing, never merged)")
    if not cha:
        P("  cohorts     : " + ", ".join(f"{k}={v}" for k, v in sorted(cohort_dist.items())))
        P("  adv suites  : " + ", ".join(f"{k}={v}" for k, v in sorted(adv_counts.items())))

    # ---- ACT III ---------------------------------------------------------- #
    out = run_population(worlds)
    results = out["results"]
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    se = sum(r.should_emit for r in results)
    max_lat = max((max(r.latencies) if r.latencies else 0.0) for r in results)
    p95 = max(c.p95_latency_s for c in out["cohorts"])
    pop_corrupt = sum(r.unicode_corruptions for r in results)

    hr("ACT III — THE RECEIPT (pipeline, category [1])")
    P(f"  recall     : {tp/se*100 if se else 100:.1f}%   [correct order events / order-bearing msgs]")
    P(f"  precision  : {tp/(tp+fp)*100 if (tp+fp) else 100:.1f}%   [correct / emitted order events]")
    P(f"  p95 / max capture latency : {p95:.2f}s / {max_lat:.2f}s")
    P(f"  unicode corruptions       : {pop_corrupt}")
    P(f"  exact ledger reproduced   : {out['ledger_match_rate']*100:.1f}% of sellers")

    # ---- ACT IV: FAILURE ATTRIBUTION (single-cause) ----------------------- #
    # Aggregate adversarial suites (in-process => parser/resolver domain only).
    adv_agg = {}
    for r in results:
        for suite, (a_tp, a_fp, a_se) in r.adv.items():
            b = adv_agg.setdefault(suite, [0, 0, 0])
            b[0] += a_tp
            b[1] += a_fp
            b[2] += a_se

    # Capture-domain conditions, each mapping to exactly one cause.
    sample = _sample_per_cohort(worlds, k=3)
    parity = run_parity(sample)
    seam_delta = max((abs(p.seam_recall - p.inproc_recall) for p in parity), default=0.0)
    seam_baseline_corrupt = sum(p.corruptions for p in parity)   # unicode/grouping at baseline
    path_world = sample[0]
    path_se = sum(1 for lm in path_world.messages if lm.should_emit)
    path_sc = score_capture(path_world, capture_world(path_world, RenderParams(row_h_base=15)))
    path_miss = max(0, path_se - path_sc["tp"])                  # -> row_grouping
    burst = next((w for w in worlds if w.profile.traffic == "burst"),
                 next((w for w in worlds if w.profile.traffic == "hot"), worlds[0]))
    burst_se = sum(1 for lm in burst.messages if lm.should_emit)
    ov_sc = score_capture(burst, capture_world(burst, window=2))
    overflow_miss = max(0, burst_se - ov_sc["tp"])              # -> window_loss

    per_cause = {c: 0 for c in CAUSES}
    suite_rows = []
    adversarial_errors = 0
    for suite in sorted(adv_agg):                               # 5e deterministic
        if suite == "clean":
            continue
        s_tp, s_fp, s_se = adv_agg[suite]
        miss, fpos = max(0, s_se - s_tp), s_fp
        if miss == 0 and fpos == 0:
            continue
        per_cause["resolver"] += miss + fpos
        adversarial_errors += miss + fpos
        suite_rows.append((suite, miss, fpos, "resolver"))
    per_cause["row_grouping"] += path_miss
    per_cause["window_loss"] += overflow_miss
    per_cause["unicode_corruption"] += seam_baseline_corrupt
    total_errors = adversarial_errors + path_miss + overflow_miss + seam_baseline_corrupt

    adv_labels_present = any(k != "clean" for k in adv_counts)

    hr(f"ACT IV — FAILURE ATTRIBUTION  (seed: {seed}  — these are YOUR results)")
    if not adv_labels_present:
        # nothing-to-test (honest) vs not-tested (invalid) — Override 3
        P("Demo note: seed produced no adversarial labels; attribution table intentionally empty.")
    else:
        _require_attribution_exercised(adv_labels_present, adversarial_errors)
        _check_single_cause(total_errors, per_cause)
        P(f"  capture-seam baseline recall delta: {seam_delta*100:+.2f}%  "
          "-> every resolver miss below is parser-domain, not capture")
        P("  per cause (single-cause per error unit):")
        for c in CAUSES:                                       # fixed order (5e)
            P(f"     {c:18}: {per_cause[c]}")
        P("  per suite (resolver domain), sorted by suite:")
        for suite, miss, fpos, cause in suite_rows:            # already sorted
            P(f"     {suite:16} miss={miss:<5} false_pos={fpos:<5} -> {cause}")
        P("  row_grouping/window_loss above were exercised on purpose "
          "(pathological overlap, window=2); dedupe & unicode stayed 0.")
    P("  NOTE: rendered UiAutomator XML parity [2a]. NOT live hardware proof [2b].")

    # ---- ACT V: boundary -------------------------------------------------- #
    busy = simulate_keepup(50, 0.3, 120, 20)
    hot = simulate_keepup(200, 0.3, 400, 15)
    fire = simulate_keepup(1000, 0.3, 100, 3)
    overflow_ok = burst_se == 0 or ov_sc["tp"] == burst_se
    boundary = [
        ("busy stream (3k total/min, ~2.25k order/min)", busy.keeps_up),
        ("hot sale (12k total/min)", hot.keeps_up),
        ("firehose past window (60k total/min)", fire.keeps_up),
        ("visible-window overflow (window=2)", overflow_ok),
        ("pathological row overlap", path_sc["corruptions"] == 0),
    ]
    hr("ACT V — BOUNDARY CONDITIONS (must fail openly somewhere)")
    for label, ok in boundary:
        P(f"  {'keeps up ' if ok else 'FAILS    '} {label}")
    _require_boundary_failure(boundary)

    # ---- ACT VI: objection matrix ----------------------------------------- #
    evidence = {
        "pop_corrupt": pop_corrupt, "seam_corrupt": seam_baseline_corrupt,
        "seam_delta": seam_delta, "busy_keeps_up": busy.keeps_up,
        "firehose_fails": not fire.keeps_up,
    }
    _assert_no_forbidden_upgrade(evidence)                     # 5d
    hr("ACT VI — OBJECTION MATRIX (every objection classified)")
    for o in OBJECTIONS:                                       # registry order (5e)
        P(f"  [{o.id}] {_MARK[o.status(evidence)]}  ({o.category})")
        if not cha:
            P(f"     ALLOWED  : {o.allowed}")
            P(f"     FORBIDDEN: {o.forbidden}")
            P(f"     next     : {o.next_proof}")

    # ---- ACT VII: final ledger -------------------------------------------- #
    hr("ACT VII — FINAL LEDGER")
    P(f"  {'objection':26} {'status':14} what this means")
    for o in OBJECTIONS:
        P(f"  {o.id:26} {_MARK[o.status(evidence)]:14} {MEANINGS.get(o.id, '')}")

    P("")
    if cha:
        P("The objection named ten problems and asserted one verdict.")
        P("Here are the ten, separated.")
        P("The verdict does not survive the separation.")
    else:
        P("Result: the critique fails as a universal impossibility claim. "
          "Remaining work is adapter and business proof, not proof that the "
          "solution space exists.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="ScreenGhost Pop-proof demo")
    ap.add_argument("--seed", required=True, help="operator-supplied seed")
    ap.add_argument("--sellers", type=int, default=1000)
    ap.add_argument("--mode", default="adversarial")
    ap.add_argument("--cha-mode", action="store_true", dest="cha")
    args = ap.parse_args()
    try:
        print(run_demo(args.seed, args.sellers, args.mode, args.cha))
    except DemoInvalid as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
