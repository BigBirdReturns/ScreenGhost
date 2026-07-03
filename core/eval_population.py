"""Score the same capture->event pipeline against the labeled population.

Runs each :class:`SellerWorld` through a windowed poll replay using the exact
``OrderBook`` from the rate bench — no parser-only shortcut — and compares
emitted order events to ground-truth labels. Rolls up by seller cohort and by
adversarial suite, keeping the three evidentiary categories separate:

  * PIPELINE proof  (this file): exact text preserved, dedupe correct, ledger
    reproduced under a broad labeled population.
  * DEVICE-SEAM proof (not here): real Android view-tree capture.
  * BUSINESS proof   (not here): seller-hour improvement from a live seller.

Synthetic sellers are not live sellers; no business lift is claimed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core.orders import (
    EventType, OrderBook, classify_event, parse_confirm, reduce_ledger,
)
from core.population import LabeledMessage, SellerWorld, resolve_sku

# total messages/sec by traffic class -> honest denominators, not "tokens/min".
_RATE = {"quiet": 0.08, "normal": 0.5, "busy": 3.3, "hot": 10.0, "burst": 20.0}


def predicted_fields(ev, catalog):
    """(event_type, resolved_sku, qty) — the pipeline's read of one event.

    Shared by the in-process scorer and the Android view-tree scorer so a parity
    delta reflects the capture seam, never two different scoring rules.
    """
    return (classify_event(ev.text), resolve_sku(ev.text, catalog),
            parse_confirm(ev.text)[2])


def event_is_correct(lm, etype, psku, pqty) -> bool:
    return bool(
        lm and lm.should_emit and etype == lm.event_type
        and psku == lm.sku and pqty == lm.qty
    )


@dataclass
class WorldResult:
    cohort: str
    traffic: str
    total_msgs: int
    order_msgs: int
    duration_s: float
    tp: int
    fp: int
    should_emit: int
    unicode_corruptions: int
    dup_actionable: int
    missed: int
    latencies: List[float]
    ledger_match: bool
    headroom: float
    # adversarial breakdown: suite -> (tp, fp, should_emit)
    adv: Dict[str, List[int]] = field(default_factory=dict)


def _ledger_shape(ledger: Dict[Tuple[str, str], int]) -> Tuple[int, Dict[str, int]]:
    """(line_count, units-per-sku). Cross-namespace-safe ledger comparison."""
    units: Dict[str, int] = {}
    for (_buyer, sku), q in ledger.items():
        units[sku] = units.get(sku, 0) + q
    return len(ledger), units


def evaluate_world(world: SellerWorld, window: int = 400,
                   poll_interval_s: float = 0.3) -> WorldResult:
    msgs = world.messages
    stream = [lm.msg for lm in msgs]
    total = len(stream)
    rate = _RATE[world.profile.traffic]
    duration = total / rate

    label_by_id: Dict[str, LabeledMessage] = {}
    pos_by_id: Dict[str, int] = {}
    for j, lm in enumerate(msgs):
        label_by_id.setdefault(lm.msg.id, lm)  # first (real) label wins on dup rows
        pos_by_id.setdefault(lm.msg.id, j)     # arrival order

    catalog = world.profile.catalog
    book = OrderBook()
    first_seen: Dict[str, int] = {}
    actionable = []  # (poll_k, OrderEvent) for actionable emissions, deduped
    emitted_ids = set()
    dup_actionable = 0

    # Arrived-count at each poll, with a guaranteed final read at `total` so
    # normal traffic isn't marked as missing its tail by int() truncation.
    # A genuine firehose (step > window) still skips messages mid-stream.
    polls = int(duration / poll_interval_s) + 1
    arrived_points = [min(total, int(rate * k * poll_interval_s)) for k in range(polls)]
    arrived_points.append(total)
    for k, arrived in enumerate(arrived_points):
        lo = max(0, arrived - window)
        for ev in book.ingest(stream[lo:arrived]):
            first_seen.setdefault(ev.msg_id, k)
            if classify_event(ev.text) != EventType.CHATTER:
                if ev.msg_id in emitted_ids:
                    dup_actionable += 1
                    continue
                emitted_ids.add(ev.msg_id)
                actionable.append((k, ev))

    # correctness
    tp = fp = unicode_corruptions = 0
    latencies: List[float] = []
    adv: Dict[str, List[int]] = {}
    for k, ev in actionable:
        lm = label_by_id.get(ev.msg_id)
        etype, psku, pqty = predicted_fields(ev, catalog)
        # exact-text integrity: emitted text must equal the source line.
        if lm is not None and ev.text != lm.msg.text:
            unicode_corruptions += 1
        correct = event_is_correct(lm, etype, psku, pqty)
        if correct:
            tp += 1
            # capture latency = polls between the message appearing on screen and
            # the pipeline emitting it, in seconds. Measures the pipeline, not the
            # inter-arrival gap (which dominates at low traffic and isn't ours).
            j = pos_by_id.get(ev.msg_id, 0)
            api = rate * poll_interval_s
            arrival_poll = math.ceil((j + 1) / api) if api > 0 else 0
            latencies.append(max(0.0, (k - arrival_poll) * poll_interval_s))
        else:
            fp += 1
        suite = (lm.adversarial if lm else None) or "clean"
        bucket = adv.setdefault(suite, [0, 0, 0])
        bucket[0 if correct else 1] += 1

    should_emit_labels = [lm for lm in msgs if lm.should_emit]
    for lm in should_emit_labels:
        suite = lm.adversarial or "clean"
        adv.setdefault(suite, [0, 0, 0])[2] += 1

    # predicted ledger (display-name space) vs ground truth (buyer_id space),
    # compared by shape so a display-name collision surfaces as a mismatch.
    pred_ledger = reduce_ledger(
        (classify_event(ev.text), ev.sender, resolve_sku(ev.text, catalog),
         parse_confirm(ev.text)[2])
        for _k, ev in actionable
    )
    ledger_match = _ledger_shape(pred_ledger) == _ledger_shape(world.ledger)

    # "missed" = a unique comment never made visible in any poll (genuine
    # scroll-off), NOT a duplicate row the pipeline correctly deduped. Counting
    # unique ids keeps dup_rows from masquerading as a keep-up failure.
    unique_ids = {m.id for m in stream}
    captured = set(first_seen)
    missed = len(unique_ids - captured)
    order_msgs = len(should_emit_labels)
    a_per_int = rate * poll_interval_s
    return WorldResult(
        cohort=world.profile.cohort, traffic=world.profile.traffic,
        total_msgs=total, order_msgs=order_msgs, duration_s=duration,
        tp=tp, fp=fp, should_emit=order_msgs,
        unicode_corruptions=unicode_corruptions, dup_actionable=dup_actionable,
        missed=missed, latencies=latencies, ledger_match=ledger_match,
        headroom=float("inf") if a_per_int == 0 else window / a_per_int, adv=adv,
    )


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


@dataclass
class CohortReceipt:
    cohort: str
    sellers: int
    total_msgs: int
    total_msgs_min: float
    order_msgs: int
    order_msgs_min: float
    recall: float
    precision: float
    p95_latency_s: float
    max_latency_s: float
    unicode_corruptions: int
    dup_event_rate: float
    backlog_rate: float
    min_headroom: float
    keeps_up: bool


def _cohort_receipts(results: List[WorldResult]) -> List[CohortReceipt]:
    by: Dict[str, List[WorldResult]] = {}
    for r in results:
        by.setdefault(r.cohort, []).append(r)
    out: List[CohortReceipt] = []
    for cohort, rs in sorted(by.items()):
        tp = sum(r.tp for r in rs)
        fp = sum(r.fp for r in rs)
        se = sum(r.should_emit for r in rs)
        dur_min = sum(r.duration_s for r in rs) / 60 or 1e-9
        lat = [x for r in rs for x in r.latencies]
        actionable = tp + fp
        out.append(CohortReceipt(
            cohort=cohort, sellers=len(rs),
            total_msgs=sum(r.total_msgs for r in rs),
            total_msgs_min=sum(r.total_msgs for r in rs) / dur_min,
            order_msgs=se, order_msgs_min=se / dur_min,
            recall=tp / se if se else 1.0,
            precision=tp / actionable if actionable else 1.0,
            p95_latency_s=_p95(lat), max_latency_s=max(lat) if lat else 0.0,
            unicode_corruptions=sum(r.unicode_corruptions for r in rs),
            dup_event_rate=sum(r.dup_actionable for r in rs) / actionable if actionable else 0.0,
            backlog_rate=sum(r.missed for r in rs) / sum(r.total_msgs for r in rs),
            min_headroom=min(r.headroom for r in rs),
            keeps_up=all(r.missed == 0 for r in rs),
        ))
    return out


def _adversarial_receipts(
    results: List[WorldResult],
) -> List[Tuple[str, int, float, float, int]]:
    agg: Dict[str, List[int]] = {}
    for r in results:
        for suite, (tp, fp, se) in r.adv.items():
            b = agg.setdefault(suite, [0, 0, 0])
            b[0] += tp
            b[1] += fp
            b[2] += se
    rows = []
    for suite, (tp, fp, se) in sorted(agg.items()):
        recall = tp / se if se else 1.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        rows.append((suite, se, recall, precision, fp))  # fp: false positives
    return rows


def run_population(worlds: List[SellerWorld]) -> Dict[str, object]:
    results = [evaluate_world(w) for w in worlds]
    return {
        "cohorts": _cohort_receipts(results),
        "adversarial": _adversarial_receipts(results),
        "ledger_match_rate": sum(r.ledger_match for r in results) / len(results),
        "results": results,
    }
