"""LINE Messaging API webhook — the `line_oa` `api` rung, made live-ready.

A seller's LINE Official Account delivers every inbound message to a webhook as
a signed JSON event. The ghost, running as the seller, receives exactly that —
no app, no scrape, no OCR. This module turns a real LINE webhook request into
the SAME Candidate contract every other surface emits, so the ledger pipeline
downstream never learns the message came from LINE.

Read-only by construction: verifying and parsing needs only the channel
*secret*. The channel access token (which sends/pushes) is never imported here —
the ghost extracts, it does not distribute. That boundary is enforced by the
absence of the token, not by a promise.

Honesty boundary: this is the real webhook schema + real HMAC-SHA256 signature
verification, exercised offline against captured payloads. A LIVE send from a
real OA through a public endpoint is the operator's step (docs/
LINE_LIVE_INTEGRATION.md); it needs a channel and a reachable URL this sandbox
cannot provide. Until then `line_oa` proof stays `event-schema`, honestly.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Dict, List, Optional

from core.capture import candidates_from_api_events
from core.orders import ChatMessage, OrderBook, classify_event

SURFACE = "line_oa"


def verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    """True iff X-Line-Signature matches HMAC-SHA256(secret, raw body), base64.

    Uses a constant-time compare. `body` MUST be the exact raw request bytes —
    re-serialized JSON will not match, so callers hash what arrived on the wire.
    """
    if not channel_secret or not signature:
        return False
    digest = hmac.new(channel_secret.encode("utf-8"), body,
                      hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


def _source_id(source: Dict) -> Optional[str]:
    """Stable thread id: the user/group/room the event came from."""
    return source.get("userId") or source.get("groupId") or source.get("roomId")


def normalize_events(payload: Dict) -> List[Dict]:
    """LINE webhook JSON -> internal event dicts (candidates_from_api_events shape).

    Only message events carry content; non-message events (follow, join, postback)
    are not order-bearing and are dropped here rather than faked into candidates.
    A redelivered event keeps its message id, so downstream dedupe collapses it.
    """
    out: List[Dict] = []
    for ev in payload.get("events", []):
        if ev.get("type") != "message":
            continue
        msg = ev.get("message", {})
        src = ev.get("source", {})
        thread = _source_id(src)
        sender = src.get("userId")   # display name needs the profile API (frozen)
        ts = str(msg.get("id") or ev.get("webhookEventId") or ev.get("timestamp", ""))
        mtype = msg.get("type")
        base = {"sender": sender, "ts": ts, "thread_id": thread}
        if mtype == "text":
            out.append({**base, "text": msg.get("text", ""), "payload_type": None})
        elif mtype == "sticker":
            kw = " ".join(msg.get("keywords", []) or [])
            out.append({**base, "text": "", "payload_type": "sticker",
                        "content_desc": f"sticker {kw}".strip()})
        elif mtype == "image":
            out.append({**base, "text": "", "payload_type": "image",
                        "content_desc": "image photo"})
        elif mtype == "location":
            title = msg.get("title") or msg.get("address") or "location"
            out.append({**base, "text": "", "payload_type": "location",
                        "content_desc": f"location {title}"})
        else:  # audio/video/file -> attachment payload, still captured (named)
            out.append({**base, "text": "", "payload_type": "attachment",
                        "content_desc": f"attachment {mtype or ''}".strip()})
    return out


def webhook_to_candidates(body: str, surface: str = SURFACE):
    """Raw webhook body (JSON text) -> deduped candidates on the LINE surface."""
    payload = json.loads(body)
    return candidates_from_api_events(normalize_events(payload), surface)


def process(body: str, signature: Optional[str] = None,
            channel_secret: Optional[str] = None,
            surface: str = SURFACE, *, require_signature: bool = True) -> Dict:
    """End-to-end: verify -> candidates -> order events. Pure, socket-free.

    When require_signature is True (the live default), an absent/invalid
    signature yields verified=False and NO candidates — an unverifiable request
    is not trusted. Tests may pass require_signature=False to exercise parsing.
    """
    verified = None
    if require_signature:
        verified = verify_signature(channel_secret or "",
                                    body.encode("utf-8"), signature or "")
        if not verified:
            return {"verified": False, "candidates": [], "orders": [],
                    "reason": "signature_unverified"}

    cands = webhook_to_candidates(body, surface)
    book = OrderBook()
    orders = []
    for c in cands:
        s, ts, txt = c.to_chat_fields()
        if not txt.strip():
            continue
        for ev in book.ingest([ChatMessage(s, f"{surface}:{ts}", txt)]):
            if classify_event(ev.text) == "order":
                orders.append({"buyer": s, "text": ev.text})
    return {"verified": verified, "candidates": cands, "orders": orders,
            "reason": None}
