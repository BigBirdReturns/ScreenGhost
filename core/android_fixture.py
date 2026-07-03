"""Category [2a]: rendered seller-world parity through the view-tree seam.

The population harness (category [1]) handed the pipeline `(sender, ts, text)`
objects. This does not: it renders each synthetic seller's stream into
**device-format UiAutomator XML** — flat nodes with bounds, on a scrolling
window — then recovers the order ledger through the *real* extraction code
(`core.texttree.parse_ui_dump`) plus a geometric `group_rows` that must
reconstruct messages from position alone. The same synthetic ledger is the
answer key; the capture path is no longer in-process.

What this proves: the render -> XML -> parse -> group -> ledger path preserves
exact Thai, dedupes scrolled/replayed rows, and reproduces the ledger under
display drift (font scale, size, bubble width). A parity delta versus the
in-process result is therefore attributable to the *capture seam*, not to two
different scoring rules (both sides call the same `predicted_fields`).

What this does NOT prove (still frozen): that a real LINE/Facebook app's view
tree on real hardware matches this XML shape. There is no adb/device in this
environment. This is the extraction-code leg [2a]; the on-hardware leg [2b] and
the business leg [3] stay locked. No seller adoption or revenue is claimed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from core.eval_population import (
    _RATE, _ledger_shape, evaluate_world, event_is_correct, predicted_fields,
)
from core.orders import ChatMessage, EventType, OrderBook, classify_event, reduce_ledger
from core.population import SellerWorld
from core.texttree import TextNode, parse_ui_dump

_TS_RE = re.compile(r"^(t\d+|\d{1,2}:\d{2})$")
_PAYLOAD_TYPES = {"sticker", "location", "attachment", "image", "reaction"}


@dataclass
class RenderParams:
    """Display conditions the seam must survive. `dark` is included precisely to
    show it changes nothing — the text tree carries strings, not pixels."""

    font_scale: float = 1.0
    display_w: int = 1080
    row_h_base: int = 130
    bubble_width: int = 760
    header_offset: int = 120
    dark: bool = False


def _node(text: str, cls: str, x0: int, y0: int, x1: int, y1: int,
          content_desc: str = "") -> str:
    return (f'<node text="{escape(text)}" class="{cls}" '
            f'content-desc="{escape(content_desc)}" '
            f'clickable="false" bounds="[{x0},{y0}][{x1},{y1}]"/>')


def render_uiautomator(messages: List[ChatMessage], params: RenderParams,
                       payloads: Optional[Dict[int, Tuple[str, str]]] = None) -> str:
    """Render a visible window of messages into device-format UiAutomator XML.

    Each message is a row band: sender (left), body (middle), timestamp (right)
    as sibling text nodes at the same y — the extractor must regroup them by
    geometry. Body text is a single node regardless of visual wrapping, which is
    the view tree's structural advantage over OCR.
    """
    p = params
    row_h = max(1, int(p.row_h_base * p.font_scale))
    line_h = max(1, int(60 * p.font_scale))
    w = p.display_w
    out = ["<?xml version='1.0' encoding='UTF-8'?>", '<hierarchy rotation="0">',
           f'<node class="android.widget.FrameLayout" bounds="[0,0][{w},2400]">']
    for i, m in enumerate(messages):
        yb = p.header_offset + i * row_h + 8
        sw = min(220, 20 + len(m.sender) * 14)
        out.append(_node(m.sender, "android.widget.TextView", 40, yb, 40 + sw, yb + line_h))
        tx0 = 40 + sw + 30
        tw = min(p.bubble_width, 20 + len(m.text) * 16)
        out.append(_node(m.text, "android.widget.TextView", tx0, yb, tx0 + tw, yb + line_h))
        out.append(_node(m.ts, "android.widget.TextView", w - 140, yb, w - 20, yb + line_h))
        if payloads and i in payloads:
            _kind, desc = payloads[i]
            out.append(_node("", "android.widget.ImageView", tx0, yb + line_h,
                             tx0 + 180, yb + line_h + 150, content_desc=desc))
    out.append("</node></hierarchy>")
    return "\n".join(out)


def group_rows(nodes: List[TextNode]) -> List[ChatMessage]:
    """Reconstruct messages from flat bounded nodes — the fragile seam step.

    Clusters nodes into rows by a y-tolerance derived from the median node
    height (so ordinary font-scale drift is absorbed), then within a row reads
    timestamp (time pattern, right), sender (leftmost remaining), and body.
    Pathological drift where rows overlap will merge them — a real, reported
    failure, not a hidden one.
    """
    items = [n for n in nodes if n.bounds]
    if not items:
        return []

    def yc(n: TextNode) -> float:
        b = n.bounds
        return (b[1] + b[3]) / 2

    def xc(n: TextNode) -> float:
        b = n.bounds
        return (b[0] + b[2]) / 2

    heights = sorted(b.bounds[3] - b.bounds[1] for b in items)
    tol = 0.6 * heights[len(heights) // 2]
    items.sort(key=yc)
    rows: List[List[TextNode]] = [[items[0]]]
    for n in items[1:]:
        if abs(yc(n) - yc(rows[-1][0])) <= tol:
            rows[-1].append(n)
        else:
            rows.append([n])

    out: List[ChatMessage] = []
    for row in rows:
        row.sort(key=xc)
        ts: Optional[str] = None
        sender: Optional[str] = None
        body: List[str] = []
        payload = next((n for n in row if not n.text.strip()
                        and n.type in _PAYLOAD_TYPES), None)
        for n in row:
            t = n.text
            if not t.strip():
                continue
            if ts is None and _TS_RE.match(t.strip()):
                ts = t.strip()
            elif sender is None:
                sender = t
            else:
                body.append(t)
        body_text = " ".join(body) if body else (
            f"[{payload.type}]" if payload else "")
        if sender is not None and ts is not None and body_text:
            out.append(ChatMessage(sender=sender, ts=ts, text=body_text))
    return out


def capture_world(world: SellerWorld, params: Optional[RenderParams] = None,
                  window: int = 400) -> List:
    """Recover actionable order events for one world through the XML seam."""
    params = params or RenderParams()
    stream = [lm.msg for lm in world.messages]
    total = len(stream)
    rate = _RATE[world.profile.traffic]
    polls = int((total / rate) / 0.3) + 1
    arrived_points = [min(total, int(rate * k * 0.3)) for k in range(polls)] + [total]

    book = OrderBook()
    actionable = []
    emitted = set()
    prev: Optional[Tuple[int, int]] = None
    for arrived in arrived_points:
        lo = max(0, arrived - window)
        if (lo, arrived) == prev:
            continue  # window unchanged -> re-render would only produce dups
        prev = (lo, arrived)
        rows = group_rows(parse_ui_dump(render_uiautomator(stream[lo:arrived], params)))
        for ev in book.ingest(rows):
            if classify_event(ev.text) != EventType.CHATTER and ev.msg_id not in emitted:
                emitted.add(ev.msg_id)
                actionable.append(ev)
    return actionable


def score_capture(world: SellerWorld, actionable: List) -> Dict[str, object]:
    label_by_id = {}
    for lm in world.messages:
        label_by_id.setdefault(lm.msg.id, lm)
    catalog = world.profile.catalog
    tp = fp = corruptions = 0
    for ev in actionable:
        lm = label_by_id.get(ev.msg_id)
        if lm is None:
            # id mismatch == the seam altered sender/ts/text (e.g. bad grouping
            # or Unicode mangling). A concrete, attributable capture failure.
            fp += 1
            corruptions += 1
            continue
        etype, psku, pqty = predicted_fields(ev, catalog)
        if event_is_correct(lm, etype, psku, pqty):
            tp += 1
        else:
            fp += 1
    should_emit = sum(1 for lm in world.messages if lm.should_emit)
    pred = reduce_ledger((predicted_fields(ev, catalog)[0], ev.sender,
                          *predicted_fields(ev, catalog)[1:]) for ev in actionable)
    return {
        "tp": tp, "fp": fp, "should_emit": should_emit,
        "corruptions": corruptions,
        "ledger_match": _ledger_shape(pred) == _ledger_shape(world.ledger),
    }


@dataclass
class ParityRow:
    cohort: str
    sellers: int
    inproc_recall: float
    seam_recall: float
    inproc_precision: float
    seam_precision: float
    corruptions: int
    seam_ledger_rate: float


def run_parity(worlds: List[SellerWorld], params: Optional[RenderParams] = None
               ) -> List[ParityRow]:
    by: Dict[str, List[Tuple]] = {}
    for w in worlds:
        ip = evaluate_world(w)
        cap = capture_world(w, params)
        sc = score_capture(w, cap)
        by.setdefault(w.profile.cohort, []).append((ip, sc))
    rows = []
    for cohort, pairs in sorted(by.items()):
        ip_tp = sum(ip.tp for ip, _ in pairs)
        ip_se = sum(ip.should_emit for ip, _ in pairs)
        ip_act = sum(ip.tp + ip.fp for ip, _ in pairs)
        s_tp = sum(sc["tp"] for _, sc in pairs)
        s_se = sum(sc["should_emit"] for _, sc in pairs)
        s_act = sum(sc["tp"] + sc["fp"] for _, sc in pairs)
        rows.append(ParityRow(
            cohort=cohort, sellers=len(pairs),
            inproc_recall=ip_tp / ip_se if ip_se else 1.0,
            seam_recall=s_tp / s_se if s_se else 1.0,
            inproc_precision=ip_tp / ip_act if ip_act else 1.0,
            seam_precision=s_tp / s_act if s_act else 1.0,
            corruptions=sum(sc["corruptions"] for _, sc in pairs),
            seam_ledger_rate=sum(1 for _, sc in pairs if sc["ledger_match"]) / len(pairs),
        ))
    return rows
