"""Source-agnostic capture — the ghost inherits the user's system.

Once ScreenGhost runs on the user's own device and account, its capture options
are exactly the user's own access, tried in order:

  1. api        — official platform events the user's authenticated session
                  already receives (LINE OA / Facebook Page / Instagram). The
                  app's UI obfuscation is irrelevant on this path.
  2. view_tree  — exact on-screen text the user can read (UiAutomator), for
                  surfaces with no API.
  3. vision     — last resort, genuine pixels only (not on the text path).
  4. none       — no API and no readable text -> unsupported_surface, named.

All paths emit the SAME Candidate contract, so the OrderBook/ledger pipeline
never knows which source produced a candidate. That is why every routed
objection is a reusable strategy, not a per-platform rewrite.

Honesty boundary: the routing and the Candidate contract are tested here. Live
platform API integration is NOT built — that is a frozen, [2b]-class piece of
work. The `api` path is exercised against representative event payloads (the
shape a webhook/Graph response yields), never a live LINE/Meta connection.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.adapter import Candidate, classify_payload, extract, sha

STRATEGIES = ("api", "view_tree", "vision", "none")


@dataclass
class CaptureResult:
    surface: str
    strategy: str                       # source that produced candidates, or "none"
    candidates: List[Candidate]
    unsupported_reason: Optional[str] = None   # set only when strategy == "none"


def _candidate_from_event(ev: Dict, surface: str) -> Candidate:
    """One authenticated-session event -> one Candidate (same contract as XML)."""
    sender = ev.get("sender")
    ts = str(ev.get("ts", ""))
    text = ev.get("text", "") or ""
    ptype = ev.get("payload_type") or classify_payload(text, ev.get("content_desc", ""))
    norm = unicodedata.normalize("NFC", text)
    key = sha(f"{surface}|{sender}|{ts}|{text}|{ptype}")
    return Candidate(
        capture_id=key, source_app=surface, source_surface=surface,
        thread_or_screen_id=ev.get("thread_id"),
        first_seen_at=ts, last_seen_at=ts, raw_text=text,
        unicode_ok=(text == norm), sender=sender,
        node_bounds=None, row_bounds=None, candidate_key=key, dedupe_key=key,
        payload_type=ptype, visibility="visible",
        parser_eligible=ptype in ("text", "emoji_text"),
        snapshot_hash=sha(json.dumps(ev, ensure_ascii=False, sort_keys=True)))


def candidates_from_api_events(events: List[Dict], surface: str) -> List[Candidate]:
    """Events the user's own authenticated session receives -> candidates.

    Deduped by candidate_key so a re-delivered webhook event is captured once.
    """
    seen, out = set(), []
    for ev in events:
        c = _candidate_from_event(ev, surface)
        if c.dedupe_key not in seen:
            seen.add(c.dedupe_key)
            out.append(c)
    return out


def candidates_from_view_tree(fixture_or_xml: str) -> List[Candidate]:
    """Exact on-screen text via the real view-tree extractor (reused)."""
    _meta, cands, _stats = extract(fixture_or_xml)
    return cands


def capture(surface: str, *, api_events: Optional[List[Dict]] = None,
            view_tree_xml: Optional[str] = None) -> CaptureResult:
    """Route capture: api-first -> view_tree -> (vision) -> unsupported_surface.

    The ghost prefers the path its inherited access affords. A view tree that
    yields no parser-eligible text (accessibility stripped / obfuscated) is not
    a usable source and falls through — to unsupported_surface if nothing else
    is available, named honestly.
    """
    if api_events:
        return CaptureResult(surface, "api",
                             candidates_from_api_events(api_events, surface))
    if view_tree_xml is not None:
        cands = candidates_from_view_tree(view_tree_xml)
        if any(c.parser_eligible for c in cands):
            return CaptureResult(surface, "view_tree", cands)
    # vision path intentionally not built (fallback for genuine pixels only)
    return CaptureResult(surface, "none", [], unsupported_reason="unsupported_surface")
