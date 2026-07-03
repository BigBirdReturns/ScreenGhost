"""Live Commerce order intake — chat text into structured order events.

Deterministic, no model. Live Commerce buyers confirm an order by commenting a
keyword ("CF", from "confirm") usually with an item code and a quantity. This
turns those comments into typed :class:`OrderEvent` records and — the part that
actually matters at volume — emits each comment **exactly once** across repeated
view-tree polls, correctly keeping two different buyers who type the same thing
apart.

Identity is ``(sender, ts, text)``. In LINE/Facebook Live the commenter name and
a timestamp render as their own nodes per comment, so this tuple is a strong
natural key: the same comment re-read on the next poll collapses to one event,
while "CF A12" from two buyers stays two orders. The one honest limitation: if
the in-app timestamp granularity is coarse (minute-level) and one buyer sends
the identical line twice inside that minute, the two merge. That is a bounded,
documented miss, not a silent one — see ``Identity`` notes in the tests.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Buyers confirm with a keyword. "cf" is near-universal in Thai streams; local
# words appear too. Configurable rather than pretending one list is complete.
DEFAULT_CONFIRM_KEYWORDS: Tuple[str, ...] = ("cf", "รับ", "จอง")

# Item code: 1-3 letters + 1-3 digits (A12, B7, L03). Quantity: "x2" or a
# number followed by a Thai/English counter (2 ตัว, 2 ชิ้น, 2 pcs).
_ITEM_RE = re.compile(r"\b([A-Za-z]{1,3}\d{1,3})\b")
_QTY_RE = re.compile(
    r"x\s*(\d{1,3})|(\d{1,3})\s*(?:ตัว|ชิ้น|อัน|pcs?|pieces?)",
    re.IGNORECASE,
)


def _hash(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ChatMessage:
    """One comment as read from the view tree — exact text, never OCR'd."""

    sender: str
    ts: str  # in-app timestamp string, e.g. "14:32"
    text: str

    @property
    def id(self) -> str:
        return _hash(self.sender, self.ts, self.text)


def parse_confirm(
    text: str, keywords: Sequence[str] = DEFAULT_CONFIRM_KEYWORDS
) -> Tuple[bool, Optional[str], Optional[int]]:
    """Return (is_confirm, item_code, qty) from a comment. Pure and boring."""
    low = text.lower()
    is_confirm = any(re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", low)
                     for k in keywords)
    item_m = _ITEM_RE.search(text)
    qty_m = _QTY_RE.search(text)
    qty: Optional[int] = None
    if qty_m:
        qty = int(next(g for g in qty_m.groups() if g is not None))
    return is_confirm, (item_m.group(1) if item_m else None), qty


@dataclass
class OrderEvent:
    msg_id: str
    sender: str
    ts: str
    text: str
    is_confirm: bool
    item_code: Optional[str]
    qty: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "sender": self.sender,
            "ts": self.ts,
            "text": self.text,
            "is_confirm": self.is_confirm,
            "item_code": self.item_code,
            "qty": self.qty,
        }

    @classmethod
    def from_message(cls, m: ChatMessage,
                     keywords: Sequence[str] = DEFAULT_CONFIRM_KEYWORDS) -> "OrderEvent":
        is_confirm, item, qty = parse_confirm(m.text, keywords)
        return cls(m.id, m.sender, m.ts, m.text, is_confirm, item, qty)


class EventType:
    """What a comment does to the order ledger."""

    ORDER = "order"    # a new confirmation
    MODIFY = "modify"  # change quantity on an existing order
    CANCEL = "cancel"  # withdraw an order
    CHATTER = "chatter"  # not order-bearing


CANCEL_KEYWORDS: Tuple[str, ...] = ("ยกเลิก", "ไม่เอาแล้ว", "cancel")
MODIFY_KEYWORDS: Tuple[str, ...] = ("เปลี่ยนเป็น", "แก้เป็น", "เพิ่มเป็น", "ลดเป็น", "change to")


def classify_event(
    text: str, confirm_keywords: Sequence[str] = DEFAULT_CONFIRM_KEYWORDS
) -> str:
    """Cancel > modify > order > chatter. Deterministic, no model."""
    low = text.lower()
    if any(k.lower() in low for k in CANCEL_KEYWORDS):
        return EventType.CANCEL
    if any(k.lower() in low for k in MODIFY_KEYWORDS):
        return EventType.MODIFY
    if parse_confirm(text, confirm_keywords)[0]:
        return EventType.ORDER
    return EventType.CHATTER


def reduce_ledger(
    entries: Iterable[Tuple[str, str, Optional[str], Optional[int]]]
) -> Dict[Tuple[str, str], int]:
    """Fold (event_type, buyer, sku, qty) rows into a final order ledger.

    ORDER sets a quantity, MODIFY replaces it, CANCEL removes the line. The same
    reducer runs over ground-truth labels and over emitted events, so a ledger
    mismatch is a real accounting failure, not a definitional one.
    """
    ledger: Dict[Tuple[str, str], int] = {}
    for etype, buyer, sku, qty in entries:
        if sku is None:
            continue
        key = (buyer, sku)
        if etype == EventType.CANCEL:
            ledger.pop(key, None)
        elif etype in (EventType.ORDER, EventType.MODIFY) and qty is not None:
            ledger[key] = qty
    return {k: v for k, v in ledger.items() if v > 0}


class OrderBook:
    """Exactly-once emission of order events across repeated polls.

    A live screen is re-read many times; the same comment reappears every poll.
    :meth:`ingest` returns only comments not seen before, so each becomes one
    event no matter how many polls it survives on screen.
    """

    def __init__(self, keywords: Sequence[str] = DEFAULT_CONFIRM_KEYWORDS):
        self._keywords = tuple(keywords)
        self._seen: Set[str] = set()
        self.duplicates_suppressed = 0

    def ingest(self, messages: Iterable[ChatMessage]) -> List[OrderEvent]:
        fresh: List[OrderEvent] = []
        for m in messages:
            if m.id in self._seen:
                self.duplicates_suppressed += 1
                continue
            self._seen.add(m.id)
            fresh.append(OrderEvent.from_message(m, self._keywords))
        return fresh

    @property
    def seen_count(self) -> int:
        return len(self._seen)
