"""Adapter conformance — turn 'what if the app layout differs?' into a test.

An adapter reads a surface snapshot and emits **candidates** (never order
events). OrderBook emits events. A surface is conformant if its adapter produces
the expected candidate set: exact Unicode, stable row grouping, scroll dedupe,
correct payload classification, and finite-window handling — or fails with
exactly ONE named cause.

All hashing is stable sha256 (never Python's salted builtin hash), so fixtures,
expected ledgers, and receipts reproduce from a cold run.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from core.texttree import TextNode, parse_ui_dump

# ---- payload taxonomy ---------------------------------------------------- #
PAYLOAD_TYPES = ("text", "emoji_text", "sticker", "image", "payment_screenshot",
                 "location", "attachment", "unknown")
_PAYLOAD_KEYWORDS = [
    ("payment_screenshot", ("slip", "payment", "โอน", "สลิป", "transfer", "receipt")),
    ("location", ("location", "map", "pin", "ตลาด", "สถานที่")),
    ("sticker", ("sticker",)),
    ("image", ("photo", "image", "picture", "gif", "รูป")),
    ("attachment", ("attachment", "file", "document", "voice", "audio", "video")),
]

# ---- single-cause failure taxonomy --------------------------------------- #
FAILURE_CAUSES = ("no_text_exposed", "unicode_corruption", "row_grouping_failure",
                  "dedupe_failure", "payload_classification_failure",
                  "visible_window_loss", "unsupported_surface", "malformed_fixture")

_TS_RE = re.compile(r"^(t\d+|\d{1,2}:\d{2})$")
_META_RE = re.compile(r"<!--META\s*(\{.*?\})\s*-->", re.DOTALL)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def is_emoji_text(text: str) -> bool:
    t = text.strip()
    return bool(t) and not any(c.isalnum() for c in t)


def classify_payload(text: str, content_desc: str) -> str:
    if text.strip():
        return "emoji_text" if is_emoji_text(text) else "text"
    hay = content_desc.lower()
    for ptype, needles in _PAYLOAD_KEYWORDS:
        if any(n in hay for n in needles):
            return ptype
    return "unknown"


@dataclass
class Candidate:
    capture_id: str
    source_app: str
    source_surface: str
    thread_or_screen_id: Optional[str]
    first_seen_at: str
    last_seen_at: str
    raw_text: str
    unicode_ok: bool
    sender: Optional[str]
    node_bounds: Optional[Tuple[int, int, int, int]]
    row_bounds: Optional[Tuple[int, int, int, int]]
    candidate_key: str
    dedupe_key: str
    payload_type: str
    visibility: str
    parser_eligible: bool
    snapshot_hash: str

    def to_chat_fields(self):
        return (self.sender or "", self.first_seen_at, self.raw_text)


@dataclass
class AdapterResult:
    meta: Dict
    candidates: List[Candidate]
    node_count: int
    text_node_count: int
    thai_text_node_count: int
    unicode_corruptions: int
    grouping_ok: bool
    dedupe_ok: bool
    payloads_ok: bool
    window_loss: int
    fixture_hash: str
    expected_ledger_hash: str
    computed_ledger_hash: str
    causes: List[str] = field(default_factory=list)


def _has_thai(s: str) -> bool:
    return any("฀" <= c <= "๿" for c in s)


def _bounds_union(nodes: List[TextNode]):
    bs = [n.bounds for n in nodes if n.bounds]
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def _group_rows(nodes: List[TextNode]) -> Tuple[List[List[TextNode]], bool]:
    """Cluster nodes into rows by y-band. Returns (rows, grouping_ok).

    grouping_ok is False when rows physically overlap (median height >= row
    spacing), i.e. the surface cannot be separated into per-message rows.
    """
    items = [n for n in nodes if n.bounds]
    if not items:
        return [], True
    yc = lambda n: (n.bounds[1] + n.bounds[3]) / 2
    heights = sorted(n.bounds[3] - n.bounds[1] for n in items)
    med_h = heights[len(heights) // 2]
    items.sort(key=yc)
    rows: List[List[TextNode]] = [[items[0]]]
    for n in items[1:]:
        if abs(yc(n) - yc(rows[-1][0])) <= 0.6 * med_h:
            rows[-1].append(n)
        else:
            rows.append([n])
    # A correctly separated row carries at most one timestamp. When rows
    # physically overlap (row spacing < text height), many messages collapse
    # into one band and it holds several timestamps -> grouping failed.
    def ts_count(r):
        return sum(1 for n in r if _TS_RE.match(n.text.strip()))
    grouping_ok = all(ts_count(r) <= 1 for r in rows)
    return rows, grouping_ok


def _row_to_candidates(row: List[TextNode], meta: Dict, snap_id: str,
                       snap_hash: str) -> List[Candidate]:
    row.sort(key=lambda n: (n.bounds[0] + n.bounds[2]) / 2 if n.bounds else 0)
    ts = sender = None
    text_nodes, payload_nodes = [], []
    for n in row:
        if not n.text.strip():
            payload_nodes.append(n)
        elif ts is None and _TS_RE.match(n.text.strip()):
            ts = n.text.strip()
        else:
            text_nodes.append(n)
    # Assign sender vs body. Two+ text nodes: first is the sender label. One
    # text node: it's the SENDER if the row carries a payload (sticker/payment),
    # otherwise it's the BODY (flat surface / hidden sender).
    if len(text_nodes) >= 2:
        sender, body_nodes = text_nodes[0].text, text_nodes[1:]
    elif len(text_nodes) == 1 and payload_nodes:
        sender, body_nodes = text_nodes[0].text, []
    else:
        sender, body_nodes = None, text_nodes

    out: List[Candidate] = []
    row_bounds = _bounds_union(row)
    ts_val = ts or snap_id

    def mk(text, ndbounds, ptype, eligible):
        raw = text
        norm = unicodedata.normalize("NFC", raw)
        return Candidate(
            capture_id=sha(f"{meta['fixture_id']}|{sender}|{ts_val}|{raw}|{ptype}"),
            source_app=meta.get("source_app", meta["surface_type"]),
            source_surface=meta["surface_type"],
            thread_or_screen_id=meta.get("thread_or_screen_id"),
            first_seen_at=ts_val, last_seen_at=ts_val, raw_text=raw,
            unicode_ok=(raw == norm), sender=sender, node_bounds=ndbounds,
            row_bounds=row_bounds,
            candidate_key=sha(f"{sender}|{ts_val}|{raw}|{ptype}"),
            dedupe_key=sha(f"{sender}|{ts_val}|{raw}|{ptype}"),
            payload_type=ptype, visibility="visible",
            parser_eligible=eligible, snapshot_hash=snap_hash)

    for n in body_nodes:
        ptype = classify_payload(n.text, "")
        out.append(mk(n.text, n.bounds, ptype, ptype in ("text", "emoji_text")))
    for n in payload_nodes:
        ptype = classify_payload("", n.content_desc)
        out.append(mk("", n.bounds, ptype, False))
    return out


def extract(fixture_xml: str) -> Tuple[Dict, List[Candidate], Dict]:
    """Parse a fixture into (meta, deduped candidates, stats)."""
    m = _META_RE.search(fixture_xml)
    if not m:
        raise ValueError("malformed_fixture: no <!--META ...--> block")
    try:
        meta = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed_fixture: bad META json ({e})")

    body = fixture_xml[m.end():]
    hierarchies = re.findall(r"<hierarchy.*?</hierarchy>", body, re.DOTALL)
    if not hierarchies:
        raise ValueError("malformed_fixture: no <hierarchy> snapshots")

    node_count = text_node_count = thai = 0
    grouping_ok = True
    seen: Dict[str, Candidate] = {}
    order: List[str] = []
    for i, hx in enumerate(hierarchies):
        snap_hash = sha(hx)
        nodes = parse_ui_dump(hx)
        node_count += len(list(ET.fromstring(hx).iter("node")))
        for n in nodes:
            if n.text.strip():
                text_node_count += 1
                if _has_thai(n.text):
                    thai += 1
        rows, ok = _group_rows(nodes)
        grouping_ok = grouping_ok and ok
        for row in rows:
            for c in _row_to_candidates(row, meta, f"s{i}", snap_hash):
                if c.dedupe_key in seen:
                    seen[c.dedupe_key].last_seen_at = c.first_seen_at  # re-seen
                else:
                    seen[c.dedupe_key] = c
                    order.append(c.dedupe_key)
    candidates = [seen[k] for k in order]
    stats = {"node_count": node_count, "text_node_count": text_node_count,
             "thai_text_node_count": thai, "grouping_ok": grouping_ok}
    return meta, candidates, stats


def canonical_ledger(candidates: List[Candidate]) -> str:
    rows = sorted(({"sender": c.sender, "text": c.raw_text,
                    "payload_type": c.payload_type} for c in candidates),
                  key=lambda r: (r["sender"] or "", r["text"], r["payload_type"]))
    return json.dumps(rows, ensure_ascii=False, sort_keys=True)


def fixture_body_hash(fixture_xml: str) -> str:
    m = _META_RE.search(fixture_xml)
    body = fixture_xml[m.end():] if m else fixture_xml
    return sha(re.sub(r"\s+", " ", body).strip())


def conformance(fixture_xml: str) -> AdapterResult:
    """Run the full contract check on a fixture, single-cause on failure."""
    causes: List[str] = []
    try:
        meta, candidates, stats = extract(fixture_xml)
    except ValueError as e:
        cause = str(e).split(":")[0]
        return AdapterResult(
            meta={}, candidates=[], node_count=0, text_node_count=0,
            thai_text_node_count=0, unicode_corruptions=0, grouping_ok=False,
            dedupe_ok=False, payloads_ok=False, window_loss=0, fixture_hash="",
            expected_ledger_hash="", computed_ledger_hash="",
            causes=[cause if cause in FAILURE_CAUSES else "malformed_fixture"])

    expected = meta.get("expected_candidates", [])
    exp_hash = sha(json.dumps(sorted(expected,
                   key=lambda r: (r.get("sender") or "", r["text"], r["payload_type"])),
                   ensure_ascii=False, sort_keys=True))
    got_hash = sha(canonical_ledger(candidates))

    unicode_corruptions = sum(0 if c.unicode_ok else 1 for c in candidates)

    # no text exposed at all
    if stats["text_node_count"] == 0 and not any(
            c.payload_type in ("text", "emoji_text") for c in candidates):
        if not candidates:
            causes.append("no_text_exposed")

    if unicode_corruptions:
        causes.append("unicode_corruption")
    if not stats["grouping_ok"]:
        causes.append("row_grouping_failure")

    # dedupe: a well-formed adapter never emits the same candidate_key twice;
    # extract() dedups, so a mismatch vs a no-dedupe count reveals a failure.
    got_texts = {(c.sender, c.raw_text, c.payload_type) for c in candidates}
    dedupe_ok = len(got_texts) == len(candidates)
    if not dedupe_ok:
        causes.append("dedupe_failure")

    # payload classification: compare produced payload set to expected
    _k = lambda t: (t[0] or "", t[1], t[2])
    exp_payloads = sorted(((r.get("sender"), r["text"], r["payload_type"]) for r in expected), key=_k)
    got_payloads = sorted(((c.sender, c.raw_text, c.payload_type) for c in candidates), key=_k)
    payloads_ok = exp_payloads == got_payloads
    if not payloads_ok and "row_grouping_failure" not in causes and "no_text_exposed" not in causes:
        causes.append("payload_classification_failure")

    window_loss = int(meta.get("declared_window_loss", 0))
    if window_loss:
        causes.append("visible_window_loss")

    return AdapterResult(
        meta=meta, candidates=candidates, node_count=stats["node_count"],
        text_node_count=stats["text_node_count"],
        thai_text_node_count=stats["thai_text_node_count"],
        unicode_corruptions=unicode_corruptions, grouping_ok=stats["grouping_ok"],
        dedupe_ok=dedupe_ok, payloads_ok=payloads_ok, window_loss=window_loss,
        fixture_hash=fixture_body_hash(fixture_xml), expected_ledger_hash=exp_hash,
        computed_ledger_hash=got_hash, causes=causes)


def verdict(result: AdapterResult) -> Tuple[str, Optional[str]]:
    """(verdict, primary_cause). Enforces single-cause + declared expectations.

    Returns one of PASS / EXPECTED_FAIL / FAIL / MULTI_CAUSE / UNDECLARED_PASS.
    """
    meta = result.meta
    expected = meta.get("expected_verdict", "PASS")
    if len(result.causes) > 1:
        return "MULTI_CAUSE", ",".join(result.causes)
    if result.causes:
        cause = result.causes[0]
        if expected == "EXPECTED_FAIL" and meta.get("expected_failure_cause") == cause:
            return "EXPECTED_FAIL", cause
        return "FAIL", cause
    # no failure
    if expected == "EXPECTED_FAIL":
        return "UNDECLARED_PASS", None
    # a clean pass must still match the expected candidate ledger
    if result.computed_ledger_hash != result.expected_ledger_hash:
        return "FAIL", "payload_classification_failure"
    return "PASS", None
