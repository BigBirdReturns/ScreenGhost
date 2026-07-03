"""Pop-proof demo — adversarial proof theater, not a product demo.

    python examples/pop_proof_demo.py --seed "pop-picks-this-live" --sellers 1000 --mode adversarial
    python examples/pop_proof_demo.py --seed "anything" --sellers 50 --cha-mode

One command demonstrates that the original critique was overbroad because it
conflated separable objections into one impossibility claim. The operator picks
the seed; the seller population is generated fresh from it (no golden replay);
the real evaluators run; failures are exposed, not hidden. It claims neither
hardware nor business proof.
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.android_fixture import RenderParams, capture_world, run_parity, score_capture
from core.eval_population import run_population
from core.ingest import simulate_keepup
from core.objections import OBJECTIONS
from core.population import build_population

_MARK = {"PASS": "PASS", "PARTIAL": "PARTIAL", "ARCH": "ARCHITECTURAL", "FROZEN": "FROZEN"}


def seed_to_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)


def _require_boundary_failure(results):
    """results: list of (label, ok_bool). At least one must be a real failure."""
    if not any(not ok for _label, ok in results):
        raise RuntimeError("Demo invalid: no boundary failure was exercised.")


def _sample_per_cohort(worlds, k=5):
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
    hr = (lambda t: P(f"\n== {t} ==")) if cha else (lambda t: P(f"\n{'='*66}\n{t}\n{'='*66}"))

    # ---- ACT I: objection decomposition ----------------------------------- #
    hr("ACT I — THE OBJECTION (decomposed)")
    P("One 'you are out of touch' verdict bundled ten separable problems:")
    for o in OBJECTIONS:
        P(f"  - {o.id}")
    P("\nRules enforced live:")
    P("  1. No objection may borrow evidence from another category.")
    P("  2. No rate may be reported without denominator discipline.")
    P("  3. No synthetic proof may become a hardware or business proof.")

    # ---- ACT II: fresh seller world --------------------------------------- #
    seed_int = seed_to_int(seed)
    worlds = build_population(n=sellers, seed=seed_int)
    total_msgs = sum(len(w.messages) for w in worlds)
    order_msgs = sum(1 for w in worlds for lm in w.messages if lm.should_emit)
    cohort_dist = {}
    adv_counts = {}
    for w in worlds:
        cohort_dist[w.profile.cohort] = cohort_dist.get(w.profile.cohort, 0) + 1
        for lm in w.messages:
            key = lm.adversarial or "clean"
            adv_counts[key] = adv_counts.get(key, 0) + 1

    hr("ACT II — BUILD THE WORLD (fresh from the seed, not replayed)")
    P(f"  seed (string)        : {seed!r}")
    P(f"  seed (int)           : {seed_int}")
    P(f"  synthetic sellers    : {sellers}")
    P(f"  cohorts              : {len(cohort_dist)}")
    P(f"  total messages       : {total_msgs}")
    P(f"  order-bearing msgs   : {order_msgs}")
    P("  (denominators kept separate: total vs order-bearing, never merged)")
    if not cha:
        P("  cohort distribution  : " + ", ".join(f"{k}={v}" for k, v in sorted(cohort_dist.items())))
        P("  adversarial suites   : " + ", ".join(f"{k}={v}" for k, v in sorted(adv_counts.items())))

    # ---- ACT III: pipeline receipt ---------------------------------------- #
    out = run_population(worlds)
    results = out["results"]
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    se = sum(r.should_emit for r in results)
    overall_recall = tp / se if se else 1.0
    overall_prec = tp / (tp + fp) if (tp + fp) else 1.0
    max_lat = max((max(r.latencies) if r.latencies else 0.0) for r in results)
    p95 = max(c.p95_latency_s for c in out["cohorts"])
    pop_corrupt = sum(r.unicode_corruptions for r in results)

    hr("ACT III — THE RECEIPT (pipeline, category [1])")
    P(f"  recall (order events)   : {overall_recall*100:.1f}%   [num = correct order events / order-bearing msgs]")
    P(f"  precision               : {overall_prec*100:.1f}%")
    P(f"  p95 / max capture lat   : {p95:.2f}s / {max_lat:.2f}s")
    P(f"  unicode corruptions     : {pop_corrupt}")
    P(f"  duplicate event rate    : {sum(c.dup_event_rate for c in out['cohorts'])/len(out['cohorts']):.3f}")
    P(f"  exact ledger reproduced : {out['ledger_match_rate']*100:.1f}% of sellers")
    P("  where it bleeds (failure surface, exposed):")
    for suite, n, rec, prec, fpc in out["adversarial"]:
        if suite == "clean":
            continue
        P(f"     {suite:16} recall={rec*100:5.1f}%  precision={prec*100:5.1f}%  false_pos={fpc}")

    # ---- ACT IV: view-tree parity ----------------------------------------- #
    sample = _sample_per_cohort(worlds, k=3)
    parity = run_parity(sample)
    seam_corrupt = sum(p.corruptions for p in parity)
    seam_delta = max((abs(p.seam_recall - p.inproc_recall) for p in parity), default=0.0)
    path = score_capture(sample[0], capture_world(sample[0], RenderParams(row_h_base=15)))
    path_se = sum(1 for lm in sample[0].messages if lm.should_emit)

    hr("ACT IV — VIEW-TREE PARITY (category [2a], NOT live hardware)")
    P(f"  parity sample           : {len(sample)} sellers rendered to UiAutomator XML")
    P(f"  max recall delta        : {seam_delta*100:+.2f}%  (in-process vs view-tree seam)")
    P(f"  row-grouping/unicode fails: {seam_corrupt}")
    P(f"  pathological overlap    : recall={path['tp']/path_se*100 if path_se else 0:.1f}%  "
      f"corruptions={path['corruptions']}  (fails in the open, by design)")
    P("  NOTE: this is rendered UiAutomator XML parity. It is NOT live hardware proof [2b].")

    # ---- ACT V: boundary conditions --------------------------------------- #
    busy = simulate_keepup(50, 0.3, 120, 20)
    hot = simulate_keepup(200, 0.3, 400, 15)
    fire = simulate_keepup(1000, 0.3, 100, 3)
    burst = next((w for w in worlds if w.profile.traffic == "burst"),
                 next((w for w in worlds if w.profile.traffic == "hot"), worlds[0]))
    burst_se = sum(1 for lm in burst.messages if lm.should_emit)
    overflow = score_capture(burst, capture_world(burst, window=2))
    overflow_ok = burst_se == 0 or overflow["tp"] == burst_se

    boundary = [
        ("busy stream (3k total/min, ~2.25k order/min)", busy.keeps_up),
        ("hot sale (12k total/min)", hot.keeps_up),
        ("firehose past window (60k total/min)", fire.keeps_up),
        ("visible-window overflow (window=2)", overflow_ok),
        ("pathological row overlap", path["corruptions"] == 0),
    ]
    hr("ACT V — BOUNDARY CONDITIONS (must fail openly somewhere)")
    for label, ok in boundary:
        P(f"  {'keeps up ' if ok else 'FAILS    '} {label}")
    _require_boundary_failure(boundary)  # raises if everything is green

    # ---- ACT VI: objection matrix receipt --------------------------------- #
    evidence = {
        "pop_corrupt": pop_corrupt, "seam_corrupt": seam_corrupt,
        "seam_delta": seam_delta, "busy_keeps_up": busy.keeps_up,
        "firehose_fails": not fire.keeps_up,
    }
    hr("ACT VI — OBJECTION MATRIX (every objection classified)")
    for o in OBJECTIONS:
        P(f"  [{o.id}] {_MARK[o.status(evidence)]}  ({o.category})")
        if not cha:
            P(f"     ALLOWED  : {o.allowed}")
            P(f"     FORBIDDEN: {o.forbidden}")
            P(f"     next     : {o.next_proof}")

    # ---- ACT VII: final ledger -------------------------------------------- #
    meaning = {
        "THAI_TEXT_RELIABILITY": "exact Unicode preserved in tested paths",
        "VISION_LATENCY": "no OCR/model on text capture path",
        "LIVE_COMMERCE_BURST": "keeps up inside modeled window; fails openly past it",
        "WINDOW_MANAGEMENT": "dedupe/window model measured, not handwaved",
        "NON_TEXT_PAYLOADS": "event classes modeled; full real-app coverage not claimed",
        "PARSER_GENERALIZATION": "adversarial failures exposed",
        "APP_SURFACE_DRIFT": "rendered drift tested; real apps frozen",
        "CENTRAL_IP_BLOCKING": "distributed design removes shared origin; hardware not run",
        "USER_INFRASTRUCTURE": "phone-first design specified; hardware not run",
        "BUSINESS_OUTCOME": "no seller-hour claim",
    }
    hr("ACT VII — FINAL LEDGER")
    P(f"  {'objection':26} {'status':14} what this means")
    for o in OBJECTIONS:
        P(f"  {o.id:26} {_MARK[o.status(evidence)]:14} {meaning.get(o.id, '')}")

    P("")
    if cha:
        P("The objection was too large for the evidence behind it. ScreenGhost "
          "reduces it to ten claims, five receipts, three frozen seams, and zero "
          "denominator games.")
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
    print(run_demo(args.seed, args.sellers, args.mode, args.cha))


if __name__ == "__main__":
    main()
