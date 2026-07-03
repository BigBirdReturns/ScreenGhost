"""Generate the adapter conformance fixtures (deterministic, stable-hashed).

    python -m tools.gen_adapter_fixtures

Writes examples/adapter_fixtures/*.xml. Each file is:
  <!--META {json}-->
  <hierarchy>...</hierarchy>   (one or more snapshots)
META carries fixture_id, surface_type, expected_verdict, expected_failure_cause,
positive/negative conditions, the golden expected_candidates, and stable hashes.
"""
import os
import sys
from xml.sax.saxutils import escape

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import json

from core.adapter import canonical_ledger, extract, fixture_body_hash, sha

OUT = os.path.join(_ROOT, "examples", "adapter_fixtures")


def node(text="", cls="android.widget.TextView", cd="", b=(0, 0, 1, 1)):
    return (f'<node text="{escape(text)}" class="{cls}" content-desc="{escape(cd)}" '
            f'clickable="false" bounds="[{b[0]},{b[1]}][{b[2]},{b[3]}]"/>')


def render(rows, row_h=130, w=1080, line_h=60):
    """rows: list of dicts {sender?, ts?, text?, payload?:(kind,desc)}."""
    parts = ['<hierarchy rotation="0">',
             f'<node class="android.widget.FrameLayout" bounds="[0,0][{w},2400]">']
    for i, r in enumerate(rows):
        y = 120 + i * row_h
        if r.get("sender"):
            parts.append(node(r["sender"], b=(40, y, 240, y + line_h)))
        if r.get("text") is not None:
            parts.append(node(r["text"], b=(280, y, 900, y + line_h)))
        if r.get("ts"):
            parts.append(node(r["ts"], b=(w - 140, y, w - 20, y + line_h)))
        if r.get("payload"):
            kind, desc = r["payload"]
            parts.append(node("", "android.widget.ImageView", desc,
                              (280, y + line_h, 460, y + line_h + 150)))
        if r.get("icon_only"):   # empty node, dropped by the parser
            parts.append(node("", "android.widget.ImageView", "",
                              (40, y, 120, y + line_h)))
    parts.append("</node></hierarchy>")
    return "\n".join(parts)


def snap(rows, **kw):
    return render(rows, **kw)


# ---- fixture definitions ------------------------------------------------- #
def line_basic():
    rows = [{"sender": "Nok", "ts": "t00001", "text": "CF C01 x2 ค่ะ"},
            {"sender": "Ann", "ts": "t00002", "text": "cf cookie x1 คะ"},
            {"sender": "Nok", "ts": "t00003", "payload": ("sticker", "Sticker")}]
    return "line_like", "PASS", None, ["order_text_candidate"], ["sticker_payload"], [snap(rows)]


def line_emoji():
    rows = [{"sender": "Ploy", "ts": "t00001", "text": "CF A01 x1 ค่ะ"},
            {"sender": "Bee", "ts": "t00002", "text": "❤️❤️"}]
    return "line_like", "PASS", None, ["order_text_candidate"], ["emoji_text_candidate"], [snap(rows)]


def line_sticker():
    rows = [{"sender": "Mai", "ts": "t00001", "text": "CF B02 x1 ค่ะ"},
            {"sender": "Mai", "ts": "t00002", "payload": ("sticker", "Sticker: cat")}]
    return "line_like", "PASS", None, ["order_text_candidate"], ["sticker_payload"], [snap(rows)]


def line_payment():
    rows = [{"sender": "Aum", "ts": "t00001", "text": "โอนแล้วค่ะ"},
            {"sender": "Aum", "ts": "t00002", "payload": ("payment_screenshot", "payment slip image")}]
    return "line_like", "PASS", None, ["text_candidate"], ["payment_screenshot_payload"], [snap(rows)]


def fb_basic():
    rows = [{"sender": "Fon", "ts": "t00001", "text": "CF A03 x2 ค่ะ"},
            {"sender": "Nan", "ts": "t00002", "text": "cf serum x1 คะ"},
            {"sender": "Beam", "ts": "t00003", "payload": ("image", "photo")}]
    return "fb_live", "PASS", None, ["order_text_candidate"], ["image_payload"], [snap(rows)]


def fb_repeat_cf():
    rows = [{"sender": "Aum", "ts": "t00001", "text": "CF A01 x1 ค่ะ"},
            {"sender": "Fon", "ts": "t00002", "text": "CF A01 x1 ค่ะ"},
            {"sender": "Nan", "ts": "t00003", "payload": ("sticker", "Sticker")}]
    return "fb_live", "PASS", None, ["distinct_buyer_candidates"], ["sticker_payload"], [snap(rows)]


def dup_scroll():
    r1 = [{"sender": "Nok", "ts": "t00001", "text": "CF C01 x2 ค่ะ"},
          {"sender": "Ann", "ts": "t00002", "text": "cf cookie x1 คะ"},
          {"sender": "Nok", "ts": "t00003", "payload": ("sticker", "Sticker")}]
    r2 = r1 + [{"sender": "Bee", "ts": "t00004", "text": "CF C02 x1 ค่ะ"}]  # scrolled, old rows re-shown
    return "line_like", "PASS", None, ["order_text_candidate"], ["scroll_dedupe"], [snap(r1), snap(r2)]


def pathological():
    rows = [{"sender": "Nok", "ts": "t00001", "text": "CF C01 x1 ค่ะ"},
            {"sender": "Ann", "ts": "t00002", "text": "CF C02 x1 ค่ะ"},
            {"sender": "Bee", "ts": "t00003", "text": "CF C01 x2 ค่ะ"}]
    return ("line_like", "EXPECTED_FAIL", "row_grouping_failure",
            ["order_text_candidate"], ["overlapping_rows"], [snap(rows, row_h=15)])


def missing_text():
    rows = [{"icon_only": True}, {"icon_only": True}, {"icon_only": True}]
    return ("unknown_surface", "EXPECTED_FAIL", "no_text_exposed",
            ["no_positive"], ["no_text_nodes"], [snap(rows)])


def webview_flat():
    rows = [{"text": "CF C01 x2"}, {"text": "cf cookie x1"}, {"text": "❤️"}]
    return "webview", "PASS", None, ["text_candidate"], ["emoji_text_candidate"], [snap(rows)]


def address_fragments():
    rows = [{"sender": "Nok", "ts": "t00001", "text": "CF C01 x1 ค่ะ"},
            {"sender": "Nok", "ts": "t00002", "text": "123 ถนนสุขุมวิท"},
            {"sender": "Nok", "ts": "t00003", "text": "กรุงเทพ 10110"}]
    return "line_like", "PASS", None, ["order_text_candidate"], ["address_fragment_text"], [snap(rows)]


def variant_ambiguity():
    rows = [{"sender": "Aum", "ts": "t00001", "text": "CF กระเป๋า x1 ค่ะ"},
            {"sender": "Fon", "ts": "t00002", "payload": ("sticker", "Sticker")}]
    return "line_like", "PASS", None, ["order_text_candidate"], ["sticker_payload"], [snap(rows)]


def stockout():
    rows = [{"sender": "Nan", "ts": "t00001", "text": "CF coat x1 ค่ะ"},
            {"sender": "Nan", "ts": "t00002", "payload": ("image", "photo")}]
    return "line_like", "PASS", None, ["order_text_candidate"], ["image_payload"], [snap(rows)]


FIXTURES = {
    "line_like_basic": line_basic, "line_like_emoji": line_emoji,
    "line_like_sticker": line_sticker, "line_like_payment_screenshot": line_payment,
    "facebook_live_comments_basic": fb_basic,
    "facebook_live_comments_repeat_cf": fb_repeat_cf,
    "duplicate_scroll_rows": dup_scroll, "pathological_overlap": pathological,
    "missing_text_nodes": missing_text, "webview_flat_text": webview_flat,
    "address_fragments": address_fragments, "variant_ambiguity": variant_ambiguity,
    "stockout_substitution": stockout,
}


def build_one(fid, fn):
    surface, verdict, cause, pos, neg, snaps = fn()
    body = "\n".join(snaps)
    # golden expected candidates: run the adapter on a stub META, pin the result
    stub = json.dumps({"fixture_id": fid, "surface_type": surface})
    _meta, cands, _stats = extract(f"<!--META {stub}-->\n{body}")
    expected = json.loads(canonical_ledger(cands))
    meta = {
        "fixture_id": fid, "surface_type": surface, "expected_verdict": verdict,
        "expected_failure_cause": cause,
        "positive_conditions_exercised": pos, "negative_conditions_exercised": neg,
        "expected_candidates": expected,
        "expected_ledger_hash": sha(json.dumps(sorted(expected,
            key=lambda r: (r.get("sender") or "", r["text"], r["payload_type"])),
            ensure_ascii=False, sort_keys=True)),
        "fixture_hash": fixture_body_hash(body),
    }
    return f"<!--META {json.dumps(meta, ensure_ascii=False)}-->\n{body}\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    for fid, fn in FIXTURES.items():
        with open(os.path.join(OUT, f"{fid}.xml"), "w", encoding="utf-8") as f:
            f.write(build_one(fid, fn))
        print(f"  wrote {fid}.xml")
    print(f"{len(FIXTURES)} fixtures written to {OUT}")


if __name__ == "__main__":
    main()
