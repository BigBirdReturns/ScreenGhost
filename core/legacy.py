"""Legacy surface capture — the ghost inherits the user's *terminal* session.

The capture ladder does not stop at a phone. Push "the ghost inherits the
user's system" to its floor and you reach the machines with no API and, at the
very bottom, no digital output at all:

    api -> view_tree -> vision -> physical(robot+webcam) -> unsupported

This module implements the **green-screen (3270/5250) rung**, which is a
`view_tree`-class source, not a vision one. A mainframe terminal exposes its
screen as a *structured field buffer* — text with exact grid positions and a
protected/unprotected attribute — over the TN3270 datastream the user's own
terminal session already receives. Reading it is decades-old screen scraping
(HLLAPI), not OCR. So a green screen decodes to the SAME Candidate contract the
phone adapters emit, and the grid gives us something a chat UI never does:
row grouping is *exact*, because the row coordinate is literal, not inferred.

Honesty boundary (mirrors the `api` rows in core/surfaces.py):
  * What is exercised here: the decoded-buffer -> Candidate mapping, against a
    fixture that has the shape a TN3270 client (py3270/s3270) yields.
  * What is FROZEN, not built: a live TN3270 network connection and the
    EBCDIC->Unicode decode against a real mainframe ([2b]-class). We do not
    claim a live session, exactly as `api` rows do not claim a live webhook.
  * The physical(robot+webcam) rung below this one is FROZEN and unclaimed —
    it reintroduces genuine OCR error and needs a read-back verification loop.
    See docs/LEGACY_SURFACE_LADDER.md.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.adapter import Candidate, classify_payload, sha

# Pseudo-pixel geometry so a grid cell maps to real Candidate bounds. A 3270
# character cell is ~ this many device units; only the ratios matter.
_CHAR_W, _LINE_H = 10, 20


@dataclass
class ScreenField:
    """One decoded field from a terminal screen buffer.

    protected=True is chrome the mainframe painted (labels, titles, the panel
    id); protected=False is data (an operator entry or an output value). This
    is exactly what a TN3270 client reports per field, already decoded to text.
    """
    row: int
    col: int
    protected: bool
    text: str

    def bounds(self):
        x, y = self.col * _CHAR_W, self.row * _LINE_H
        return (x, y, x + len(self.text) * _CHAR_W, y + _LINE_H)


def load_screen_fixture(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _label_for(data: ScreenField, row_fields: List[ScreenField]) -> Optional[str]:
    """The nearest protected field to the left on the same row = the label."""
    labels = [f for f in row_fields
              if f.protected and f.text.strip() and f.col < data.col]
    if not labels:
        return None
    return max(labels, key=lambda f: f.col).text.strip().rstrip(":").strip()


def candidates_from_screen_buffer(fixture: Dict) -> List[Candidate]:
    """Decoded terminal screen buffer -> candidates (same contract as chat).

    Every unprotected field carrying text becomes one candidate; its label
    (the protected field to its left) becomes the `sender`, so a panel's
    "CUST: NOK RATANA" reads like a chat row exactly the way the ledger expects.
    Row grouping is exact — the grid row is literal, so there is no y-band
    heuristic and no `row_grouping_failure` mode on this surface.
    """
    surface = fixture.get("surface_type", "green_screen_3270")
    source_app = fixture.get("source_app", surface)
    screen_id = fixture.get("screen_id", "SCRN")
    fields = [ScreenField(**f) for f in fixture.get("fields", [])]

    by_row: Dict[int, List[ScreenField]] = {}
    for f in fields:
        by_row.setdefault(f.row, []).append(f)

    out: List[Candidate] = []
    seen = set()
    for row in sorted(by_row):
        row_fields = sorted(by_row[row], key=lambda f: f.col)
        row_bounds = (min(f.bounds()[0] for f in row_fields),
                      row * _LINE_H,
                      max(f.bounds()[2] for f in row_fields),
                      row * _LINE_H + _LINE_H)
        for f in row_fields:
            if f.protected or not f.text.strip():
                continue  # chrome/labels are context, not candidates
            sender = _label_for(f, row_fields)
            loc = f"{screen_id}:{f.row:02d}{f.col:02d}"
            raw = f.text
            norm = unicodedata.normalize("NFC", raw)
            ptype = classify_payload(raw, "")
            key = sha(f"{surface}|{sender}|{loc}|{raw}|{ptype}")
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(
                capture_id=key, source_app=source_app, source_surface=surface,
                thread_or_screen_id=screen_id, first_seen_at=loc, last_seen_at=loc,
                raw_text=raw, unicode_ok=(raw == norm), sender=sender,
                node_bounds=f.bounds(), row_bounds=row_bounds,
                candidate_key=key, dedupe_key=key, payload_type=ptype,
                visibility="visible",
                parser_eligible=ptype in ("text", "emoji_text"),
                snapshot_hash=sha(json.dumps(fixture, ensure_ascii=False,
                                             sort_keys=True))))
    return out


def field_value(fixture: Dict, label: str) -> Optional[str]:
    """Read one labeled data value off the panel (label match, case-insensitive)."""
    fields = [ScreenField(**f) for f in fixture.get("fields", [])]
    by_row: Dict[int, List[ScreenField]] = {}
    for f in fields:
        by_row.setdefault(f.row, []).append(f)
    want = label.strip().rstrip(":").strip().lower()
    for row_fields in by_row.values():
        for f in sorted(row_fields, key=lambda f: f.col):
            if f.protected or not f.text.strip():
                continue
            lbl = _label_for(f, row_fields)
            if lbl and lbl.lower() == want:
                return f.text.strip()
    return None


def screen_to_order_line(fixture: Dict) -> Optional[Dict]:
    """Project an order-entry panel to the SAME order signal the chat path uses.

    Reads the panel's labeled CUST / ITEM / QTY data fields and returns a
    normalized order line, proving a mainframe-sourced order and a LINE-sourced
    order reach the identical ledger. Returns None if the panel is not an order
    panel (no ITEM field), rather than inventing one.
    """
    item = field_value(fixture, "ITEM")
    if not item:
        return None
    cust = field_value(fixture, "CUST") or "unknown"
    qty_raw = field_value(fixture, "QTY") or "1"
    qty = int(qty_raw.lstrip("0") or "0") or 1
    return {"buyer": cust, "text": f"CF {item} x{qty}",
            "screen_id": fixture.get("screen_id", "SCRN")}
