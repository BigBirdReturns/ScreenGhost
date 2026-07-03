"""Objection-complete proof stack — the single source of truth.

Every objection from the original critique is a named row here: its
architectural answer, the test category that addresses it, the executable
bench, the receipt artifact, the claim it is ALLOWED to make, the claim it is
FORBIDDEN from making, its current status, and the remaining adapter risk.

Two hard rules are enforced by construction:
  1. Denominator discipline — benches separate total messages, order-bearing
     messages, UI nodes, and emitted events. No naked "per minute".
  2. No category borrows trust from a higher one — synthetic != device !=
     real-app != business. Each `status` is scoped to what actually ran.

`docs/OBJECTION_MATRIX.md`, `examples/objection_receipt.py`, and the tests all
read this module, so they cannot disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# status vocabulary — scoped to evidence that actually exists
PASS = "PASS"            # a runnable receipt confirms it
PARTIAL = "PARTIAL"      # confirmed for the tested surface; a leg remains
ARCHITECTURAL = "ARCH"   # true by design; awaits a device receipt to demonstrate
FROZEN = "FROZEN"        # deliberately not claimed yet


@dataclass
class Objection:
    id: str
    objection: str
    answer: str
    category: str            # [1] [2a] [2b-1] [2b-2] [3]
    bench: str               # executable
    receipt: str             # artifact path
    allowed: str
    forbidden: str
    base_status: str
    adapter_risk: str
    next_proof: str
    # optional live check: (evidence) -> refined status, else base_status
    refine: Optional[Callable[[Dict], str]] = None

    def status(self, evidence: Optional[Dict]) -> str:
        if self.refine and evidence is not None:
            return self.refine(evidence)
        return self.base_status


OBJECTIONS: List[Objection] = [
    Objection(
        id="THAI_TEXT_RELIABILITY",
        objection="Thai text is unreliable through vision/OCR.",
        answer="No OCR on the text path; extract exact Unicode from the view tree.",
        category="[2a]",
        bench="python examples/population_bench.py ; python examples/parity_bench.py",
        receipt="examples/receipts/parity_seam_v1.txt",
        allowed="Exact Thai text is preserved through the tested extraction path "
                "(view-tree XML -> parse -> group), corruption count 0.",
        forbidden="That every real app surface exposes perfect Thai nodes — "
                  "unproven until [2b-2].",
        base_status=PARTIAL,
        adapter_risk="A real app could render Thai as an image (rare) or split "
                     "nodes oddly; that would be an adapter fix, caught by [2b-2].",
        next_proof="[2b-2] benign Thai threads on a real LINE/FB surface.",
        refine=lambda e: PASS if e["pop_corrupt"] == 0 and e["seam_corrupt"] == 0 else PARTIAL,
    ),
    Objection(
        id="VISION_LATENCY",
        objection="Vision models are slow and token-heavy.",
        answer="The model is not load-bearing for text capture; deterministic "
               "extraction first, a model only at interpretation edges.",
        category="[1]",
        bench="python examples/keepup_bench.py",
        receipt="examples/receipts/population_baseline_v0.txt",
        allowed="The text-capture path invokes no model; p95 capture latency is "
                "sub-second at tested volumes.",
        forbidden="That interpretation of genuinely ambiguous language needs no "
                  "model ever — some edges will.",
        base_status=PASS,
        adapter_risk="None on the capture path; model cost reappears only if you "
                     "push interpretation onto it.",
        next_proof="Hold: keep the capture path model-free as features grow.",
    ),
    Objection(
        id="NON_TEXT_PAYLOADS",
        objection="Stickers, locations, attachments, reactions, payment screenshots matter.",
        answer="Classify them as event- or context-bearing non-text nodes, not "
               "as order text.",
        category="[1]/[2a]",
        bench="python examples/population_bench.py",
        receipt="examples/receipts/population_baseline_v0.txt",
        allowed="Non-text nodes are typed (sticker/location/attachment) from "
                "content-desc; payment-screenshot refs are surfaced as "
                "order-bearing-but-unstructured, not silently dropped.",
        forbidden="That the artwork/bytes behind a sticker or slip are decoded — "
                  "they are typed, not read.",
        base_status=PARTIAL,
        adapter_risk="Per-app content-desc conventions differ; a resolver table "
                     "per platform, not an architecture change.",
        next_proof="[2b-2] confirm real content-desc strings per platform.",
    ),
    Objection(
        id="LIVE_COMMERCE_BURST",
        objection="Live commerce creates burst loads and 50-100 concurrent buyers.",
        answer="Finite viewport, polling cadence, dedupe, backlog, and an "
               "explicit keeps_up boundary.",
        category="[1]/[2a]",
        bench="python examples/population_bench.py ; python examples/parity_bench.py",
        receipt="examples/receipts/parity_seam_v1.txt",
        allowed="Full order recall with headroom at busy/hot volumes; honest "
                "scroll-loss reported past the visible window.",
        forbidden="That an arbitrary firehose never loses a message — it does, "
                  "and the boundary says so.",
        base_status=PASS,
        adapter_risk="Real scroll physics and notification timing differ; tune "
                     "window/cadence, not architecture.",
        next_proof="[2b-1] measure real dump cadence on hardware.",
        refine=lambda e: PASS if e["firehose_fails"] and e["busy_keeps_up"] else PARTIAL,
    ),
    Objection(
        id="CENTRAL_IP_BLOCKING",
        objection="Shared origin/IP gets blocked.",
        answer="No central actuation; the seller uses their own phone, account, "
               "IP, and app session.",
        category="[2b-1]",
        bench="(architecture) core/android_fixture.py + on-device runner",
        receipt="docs/scaling-architecture.md",
        allowed="No shared origin is required by design; there is no central "
                "surface to fingerprint.",
        forbidden="That every platform policy permits every automation mode — a "
                  "ToS/account question, not an IP one.",
        base_status=ARCHITECTURAL,
        adapter_risk="Account-level policy, not IP; a product/consent posture.",
        next_proof="[2b-1] device-local runner on real hardware.",
    ),
    Objection(
        id="USER_INFRASTRUCTURE",
        objection="Sellers have no PCs or always-on servers.",
        answer="Phone-first execution, local append-only logs, replay, optional "
               "sync later.",
        category="[2b-1]",
        bench="(device) driver.dump_ui_xml -> group_rows on hardware",
        receipt="pending [2b-1]",
        allowed="The design assumes only the seller's own phone.",
        forbidden="That a working on-device build exists today — it is not yet "
                  "run on hardware.",
        base_status=ARCHITECTURAL,
        adapter_risk="Android accessibility/policy for the on-device agent.",
        next_proof="[2b-1] run the capture path on a physical device.",
    ),
    Objection(
        id="WINDOW_MANAGEMENT",
        objection="Too many windows and chats drown the system.",
        answer="Each device manages only its own seller surface; the finite "
               "viewport is modeled explicitly.",
        category="[2a]",
        bench="python examples/parity_bench.py",
        receipt="examples/receipts/parity_seam_v1.txt",
        allowed="Scroll replay is deduped and visible-window loss is reported; "
                "one device = one seller surface.",
        forbidden="That a single device multiplexes many sellers — it does not, "
                  "by design.",
        base_status=PASS,
        adapter_risk="Real multi-tab/live layouts vary; viewport model tuning.",
        next_proof="[2b-2] real live-sale layout sampling.",
    ),
    Objection(
        id="PARSER_GENERALIZATION",
        objection="Real merchant language is messier than generated language.",
        answer="Separate parser/resolver failure from the capture seam; expose "
               "failures instead of hiding them.",
        category="[1]",
        bench="python examples/population_bench.py",
        receipt="examples/receipts/population_baseline_v0.txt",
        allowed="Capture-seam delta is ~0, so current failures are attributed to "
                "parser/resolver and reported per adversarial suite.",
        forbidden="That the parser generalizes to real Thai merchant language — "
                  "the generator shares its grammar; unproven.",
        base_status=PARTIAL,
        adapter_risk="High: real language needs real-data parser work; this is "
                     "the biggest open item.",
        next_proof="[2b-2]/[3] evaluate on real (consented) merchant text.",
    ),
    Objection(
        id="APP_SURFACE_DRIFT",
        objection="Real LINE/Facebook/Shopee view trees may differ.",
        answer="Adapter risk, not architecture risk — unless exact text is "
               "unavailable from the tree.",
        category="[2a]->[2b-2]",
        bench="python examples/parity_bench.py",
        receipt="examples/receipts/parity_seam_v1.txt",
        allowed="Recovery survives font scale, display size, bubble width, and "
                "dark mode in the fixture; pathological overlap fails openly.",
        forbidden="That real app trees group identically to the fixture — that "
                  "is exactly what [2b-2] must test.",
        base_status=PARTIAL,
        adapter_risk="Medium: per-app node layout; adapters, unless a surface "
                     "hides text entirely.",
        next_proof="[2b-1] then [2b-2] on real surfaces.",
    ),
    Objection(
        id="BUSINESS_OUTCOME",
        objection="Technical capture does not prove seller value.",
        answer="True. Freeze the business claim until seller-hour measurement "
               "exists.",
        category="[3]",
        bench="(frozen) before/after seller-hour",
        receipt="not claimed",
        allowed="Nothing about revenue or adoption.",
        forbidden="Any seller-hour lift, adoption, or revenue claim.",
        base_status=FROZEN,
        adapter_risk="N/A until a live seller exists.",
        next_proof="[3] one consented live seller, before/after.",
    ),
]


# One-line gloss per objection — owned here so the demo never restates claims.
MEANINGS: Dict[str, str] = {
    "THAI_TEXT_RELIABILITY": "exact Unicode preserved in tested paths",
    "VISION_LATENCY": "no OCR/model on text capture path",
    "NON_TEXT_PAYLOADS": "event classes modeled; full real-app coverage not claimed",
    "LIVE_COMMERCE_BURST": "keeps up inside modeled window; fails openly past it",
    "CENTRAL_IP_BLOCKING": "distributed design removes shared origin; hardware not run",
    "USER_INFRASTRUCTURE": "phone-first design specified; hardware not run",
    "WINDOW_MANAGEMENT": "dedupe/window model measured, not handwaved",
    "PARSER_GENERALIZATION": "adversarial failures exposed",
    "APP_SURFACE_DRIFT": "rendered drift tested; real apps frozen",
    "BUSINESS_OUTCOME": "no seller-hour claim",
}


def gather_evidence(n: int = 40) -> Dict:
    """Run the cheap live checks the refinable objections depend on."""
    from core.android_fixture import run_parity
    from core.eval_population import run_population
    from core.ingest import simulate_keepup
    from core.population import build_population

    worlds = build_population(n=n, seed=1337)
    pop = run_population(worlds)
    pop_corrupt = sum(r.unicode_corruptions for r in pop["results"])
    busy = simulate_keepup(50, 0.3, 120, 20)
    fire = simulate_keepup(1000, 0.3, 100, 3)
    parity = run_parity([w for w in worlds
                         if w.profile.cohort in {"bakery", "clothing", "adversarial"}])
    seam_delta = max((abs(p.seam_recall - p.inproc_recall) for p in parity), default=0.0)
    seam_corrupt = sum(p.corruptions for p in parity)
    return {
        "pop_corrupt": pop_corrupt,
        "seam_corrupt": seam_corrupt,
        "seam_delta": seam_delta,
        "busy_keeps_up": busy.keeps_up,
        "firehose_fails": not fire.keeps_up,
    }
