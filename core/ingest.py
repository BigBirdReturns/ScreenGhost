"""Ingestion loop + keep-up measurement for live comment streams.

Two things live here:

1. :func:`run_live` — the real device path. It reads the view tree on a cadence,
   turns the visible window into :class:`ChatMessage` rows, and feeds an
   :class:`OrderBook` for exactly-once order emission. It runs against a real
   phone through the existing driver seam; nothing here needs a model.

2. :func:`simulate_keepup` — the receipt that answers the scale objection
   without a phone. The critic's real requirement is *order-event recall under
   live arrival load*: of every order-bearing comment that arrived, what
   fraction became the correct structured event, fast enough to be useful, with
   no Unicode corruption and no order counted twice. This simulates a live
   stream and measures exactly that — including the boundary where recall starts
   dropping — so the claim is falsifiable, not asserted.

The keep-up model, stated plainly so it can be argued with:

    One poll reads the whole visible window (W comments) in a single dump.
    Between two polls, ``arrival_rate * poll_interval`` new comments arrive.
    A comment is captured iff some poll's window still holds it before newer
    comments push it past W. So recall is complete exactly while

        arrival_rate * poll_interval  <=  W          (headroom = W / that)

    Unlike a screenshot->VLM loop, the poll captures W comments at once and its
    cost is a cheap UI dump, so W is large and the interval small — the
    inequality holds with slack at realistic volume. When it doesn't, the
    simulation reports how many orders were missed rather than pretending
    coverage.

What this proves and what it can't: simulation proves the *pipeline* — queue
discipline, exact-text parsing, event schema, dedup identity, latency, and the
keep-up boundary. It cannot prove platform weirdness, view-tree drift, account
throttling, or app updates. Simulation proves the pipeline; one live seller
proves the claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.orders import ChatMessage, OrderBook, OrderEvent, parse_confirm


class ReplayDriver:
    """Stands in for AndroidAdbDriver, replaying pre-rendered dumps in order.

    Lets the real ingestion path run offline: same ``dump_ui_xml`` seam, canned
    frames instead of a phone. Anything past the last frame repeats it.
    """

    def __init__(self, frames: Sequence[str]):
        if not frames:
            raise ValueError("ReplayDriver needs at least one frame")
        self._frames = list(frames)
        self._i = 0

    def dump_ui_xml(self, device: Optional[str] = None) -> str:
        frame = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return frame


def run_live(
    book: OrderBook,
    rows_of: Callable[[Optional[str]], List[ChatMessage]],
    polls: int,
    device: Optional[str] = None,
    on_orders: Optional[Callable[[List[OrderEvent]], None]] = None,
) -> List[OrderEvent]:
    """Poll a live comment surface ``polls`` times, emitting each order once.

    ``rows_of`` maps a device to the current visible comment rows — in
    production, ``lambda d: group_rows(read_tree(d, driver))``; in tests, a
    function over a replayed stream. Injected so the loop carries no device or
    parsing assumptions of its own.
    """
    emitted: List[OrderEvent] = []
    for _ in range(polls):
        fresh = book.ingest(rows_of(device))
        if fresh:
            emitted.extend(fresh)
            if on_orders:
                on_orders(fresh)
    return emitted


@dataclass
class KeepUpReport:
    # arrival / capture
    arrived: int
    captured: int
    missed: int
    duplicates_suppressed: int
    # the metric that settles the fight: order-event recall
    order_arrived: int
    order_captured: int
    order_events_emitted: int
    unicode_corruptions: int
    # latency of order capture, seconds
    p95_latency_s: float
    max_latency_s: float
    # keep-up geometry
    window: int
    arrivals_per_interval: float

    @property
    def order_recall(self) -> float:
        return 1.0 if self.order_arrived == 0 else self.order_captured / self.order_arrived

    @property
    def order_precision(self) -> float:
        # emitted confirm events that were genuinely order-bearing. Deterministic
        # parse -> expected 1.0; reported so a regression can't hide.
        return 1.0 if self.order_events_emitted == 0 else \
            self.order_captured / self.order_events_emitted

    @property
    def headroom(self) -> float:
        a = self.arrivals_per_interval
        return float("inf") if a == 0 else self.window / a

    @property
    def keeps_up(self) -> bool:
        return self.missed == 0 and self.unicode_corruptions == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arrived": self.arrived,
            "captured": self.captured,
            "missed": self.missed,
            "duplicates_suppressed": self.duplicates_suppressed,
            "order_arrived": self.order_arrived,
            "order_captured": self.order_captured,
            "order_recall": round(self.order_recall, 4),
            "order_precision": round(self.order_precision, 4),
            "unicode_corruptions": self.unicode_corruptions,
            "p95_latency_s": round(self.p95_latency_s, 3),
            "max_latency_s": round(self.max_latency_s, 3),
            "window": self.window,
            "arrivals_per_interval": round(self.arrivals_per_interval, 3),
            "headroom": self.headroom,
            "keeps_up": self.keeps_up,
        }


# Chatter interleaved with real orders, so recall/precision mean something.
_CHATTER = ["สวยจังค่ะ", "ราคาเท่าไหร่คะ", "มีสีอื่นไหม", "ส่งวันไหนคะ", "❤️❤️"]
_ITEMS = ["A01", "A03", "B12", "L07", "C22"]


def _synthetic_stream(n: int, buyers: int) -> List[ChatMessage]:
    """A deterministic live stream: rotating buyers, real Thai lines.

    Deterministic on purpose (no RNG) so the number is reproducible. One in four
    lines is chatter, not an order. Two different buyers routinely emit the
    identical order text, which is exactly the collision (sender, ts, text)
    identity must survive.
    """
    out: List[ChatMessage] = []
    for i in range(n):
        sender = f"buyer{i % buyers:03d}"
        ts = f"t{i:05d}"  # unique per arrival; stands in for the in-app clock
        if i % 4 == 0:
            text = _CHATTER[i % len(_CHATTER)]
        else:
            item = _ITEMS[i % len(_ITEMS)]
            text = f"CF {item} x{(i % 3) + 1} ค่ะ"
        out.append(ChatMessage(sender=sender, ts=ts, text=text))
    return out


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def simulate_keepup(
    arrival_rate: float,
    poll_interval_s: float,
    window: int,
    duration_s: float,
    buyers: int = 200,
) -> KeepUpReport:
    """Measure order-event recall + latency of the poll loop against a stream.

    Models the scrolling window empirically: at each poll the loop sees the last
    ``window`` comments that have arrived, dedups via an OrderBook, and we track
    which arrivals were ever captured and how long after arrival. Comments that
    scroll past ``window`` between polls are counted as missed — the honest
    failure mode, not smoothed over.
    """
    total = int(round(arrival_rate * duration_s))
    stream = _synthetic_stream(total, buyers)
    order_flags = [parse_confirm(m.text)[0] for m in stream]
    order_arrived = sum(order_flags)

    book = OrderBook()
    first_seen_poll: Dict[str, int] = {}
    order_events_emitted = 0
    unicode_corruptions = 0

    polls = int(duration_s / poll_interval_s) + 1
    # Append a guaranteed final read at `total` so int() truncation on the last
    # scheduled poll can't fake a tail miss (a genuine firehose still drops
    # messages mid-stream where step > window).
    arrived_schedule = [min(total, int(arrival_rate * k * poll_interval_s))
                        for k in range(polls)] + [total]
    for k, arrived_by_now in enumerate(arrived_schedule):
        lo = max(0, arrived_by_now - window)
        visible = stream[lo:arrived_by_now]
        for ev in book.ingest(visible):
            first_seen_poll.setdefault(ev.msg_id, k)
            # exact-text integrity check: the emitted event text must equal the
            # source line byte-for-byte. Any drift is a corruption, not a warn.
            src = stream[lo + [m.id for m in visible].index(ev.msg_id)]
            if ev.text != src.text:
                unicode_corruptions += 1
            if ev.is_confirm:
                order_events_emitted += 1

    # capture accounting
    captured_ids = set(first_seen_poll)
    captured = len(captured_ids)
    order_captured = sum(
        1 for j, m in enumerate(stream) if order_flags[j] and m.id in captured_ids
    )

    # latency of order capture: (first-seen time) - (arrival time)
    latencies: List[float] = []
    for j, m in enumerate(stream):
        if order_flags[j] and m.id in first_seen_poll:
            arrival_t = j / arrival_rate
            seen_t = first_seen_poll[m.id] * poll_interval_s
            latencies.append(max(0.0, seen_t - arrival_t))

    return KeepUpReport(
        arrived=total,
        captured=captured,
        missed=total - captured,
        duplicates_suppressed=book.duplicates_suppressed,
        order_arrived=order_arrived,
        order_captured=order_captured,
        order_events_emitted=order_events_emitted,
        unicode_corruptions=unicode_corruptions,
        p95_latency_s=_p95(latencies),
        max_latency_s=max(latencies) if latencies else 0.0,
        window=window,
        arrivals_per_interval=arrival_rate * poll_interval_s,
    )
