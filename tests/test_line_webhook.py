"""LINE Messaging API webhook: real signature + real schema -> same contract."""
import base64
import hashlib
import hmac
import json
import os

from core.line_webhook import (
    normalize_events, process, verify_signature, webhook_to_candidates,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "examples", "line_fixtures", "webhook_sample.json")
SECRET = "test_channel_secret_do_not_use_in_prod"


def _body():
    return open(FIX, encoding="utf-8").read()


def _sign(secret, body):
    return base64.b64encode(
        hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()).decode()


def test_signature_roundtrip():
    body = _body()
    good = _sign(SECRET, body)
    assert verify_signature(SECRET, body.encode(), good) is True
    assert verify_signature("wrong_secret", body.encode(), good) is False
    assert verify_signature(SECRET, body.encode(), "AAAA") is False
    assert verify_signature(SECRET, body.encode(), "") is False


def test_signature_is_body_exact():
    # a single byte change to the body invalidates the signature (no re-serialize)
    body = _body()
    sig = _sign(SECRET, body)
    assert verify_signature(SECRET, (body + " ").encode(), sig) is False


def test_text_event_exact_thai():
    cands = webhook_to_candidates(_body())
    order = next(c for c in cands if c.raw_text.startswith("CF C01"))
    assert order.raw_text == "CF C01 x2 ตัว ค่ะ"   # byte-exact, no OCR
    assert order.unicode_ok and order.parser_eligible
    assert order.source_surface == "line_oa"
    assert order.thread_or_screen_id == "Ubuyer0000000000000000000000nok"


def test_payload_types_mapped():
    by = {c.payload_type for c in webhook_to_candidates(_body())}
    assert {"text", "sticker", "image"} <= by
    # non-message (follow) event produced no candidate
    stickers = [c for c in webhook_to_candidates(_body()) if c.payload_type == "sticker"]
    assert stickers and not stickers[0].parser_eligible


def test_non_message_events_dropped():
    payload = json.loads(_body())
    evs = normalize_events(payload)
    # 4 message events in, 1 follow dropped
    assert len(evs) == 4


def test_redelivery_dedupes():
    payload = json.loads(_body())
    # duplicate the first event verbatim (a LINE redelivery keeps the message id)
    payload["events"].append(payload["events"][0])
    body = json.dumps(payload, ensure_ascii=False)
    texts = [c.raw_text for c in webhook_to_candidates(body)]
    assert texts.count("CF C01 x2 ตัว ค่ะ") == 1


def test_process_requires_valid_signature():
    body = _body()
    bad = process(body, "AAAA", SECRET)
    assert bad["verified"] is False and bad["candidates"] == []
    good = process(body, _sign(SECRET, body), SECRET)
    assert good["verified"] is True and good["candidates"]


def test_process_produces_order_events():
    body = _body()
    res = process(body, _sign(SECRET, body), SECRET)
    assert any("C01" in o["text"] for o in res["orders"])


def test_process_parse_only_mode():
    # tests may bypass signature to exercise parsing; live default does not
    res = process(_body(), require_signature=False)
    assert res["verified"] is None and res["candidates"]


def test_no_send_token_imported():
    # read-only by construction: the module must not reference an access token
    src = open(os.path.join(ROOT, "core", "line_webhook.py"), encoding="utf-8").read()
    assert "access_token" not in src.lower() and "channel_access" not in src.lower()
