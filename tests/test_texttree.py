"""The proof, run offline: the OS view tree yields exact Thai — no OCR.

These tests feed a real UiAutomator-shaped hierarchy dump (including a Thai
Live-Commerce order line, a sticker, and a toggle) through the pure parser and
assert exact extraction. No device, no model, no GPU — which is exactly the
argument: the robust read path is boring, deterministic, and script-agnostic.
"""
from core.texttree import PAYLOAD_HINTS, parse_ui_dump, to_elements

# A slice of a chat/Live-Commerce screen as UiAutomator actually emits it.
# The order comment is Thai: "CF 2 ตัว ค่ะ" ("confirm, 2 pieces, please").
# The quantity and script must survive verbatim — a misread here is a
# mis-shipment, which is precisely where a small VLM fails and this does not.
THAI_CHAT_DUMP = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.FrameLayout" bounds="[0,0][1080,2340]">
    <node index="0" text="ร้านค้าไลฟ์สด" class="android.widget.TextView"
          content-desc="" clickable="false" bounds="[48,120][520,180]"/>
    <node index="1" text="CF 2 ตัว ค่ะ" class="android.widget.TextView"
          content-desc="" clickable="false" bounds="[64,400][700,470]"/>
    <node index="2" text="" class="android.widget.ImageView"
          content-desc="Sticker" clickable="false" bounds="[64,500][240,676]"/>
    <node index="3" text="" class="android.widget.ImageView"
          content-desc="Shared location: ตลาดนัด" clickable="true"
          bounds="[64,700][520,900]"/>
    <node index="4" text="ส่งข้อความ" class="android.widget.EditText"
          content-desc="" clickable="true" bounds="[48,2200][900,2280]"/>
    <node index="5" text="Notifications" class="android.widget.Switch"
          content-desc="" checkable="true" checked="true"
          clickable="true" bounds="[920,2200][1040,2280]"/>
    <node index="6" text="" class="android.widget.ImageButton"
          content-desc="Back" clickable="true" bounds="[0,120][80,200]"/>
  </node>
</hierarchy>"""


def _by_label(nodes):
    return {n.label: n for n in nodes}


def test_thai_text_is_exact_no_ocr():
    nodes = parse_ui_dump(THAI_CHAT_DUMP)
    labels = {n.text for n in nodes}
    # Exact, byte-for-byte — the whole point. No "confidence", no near-miss.
    assert "CF 2 ตัว ค่ะ" in labels
    assert "ร้านค้าไลฟ์สด" in labels


def test_quantity_survives_verbatim():
    # The failure a small VLM makes on Thai is a wrong quantity -> wrong order.
    order = next(n for n in parse_ui_dump(THAI_CHAT_DUMP) if n.text.startswith("CF"))
    assert order.text == "CF 2 ตัว ค่ะ"
    assert "2" in order.text


def test_non_text_payloads_are_structured_not_dropped():
    by = {n.type: n for n in parse_ui_dump(THAI_CHAT_DUMP)}
    # Sticker and location arrive as typed payloads, the way a protocol API
    # would hand them over — not as pixels to re-recognize.
    assert "sticker" in by
    assert "location" in by
    assert "ตลาดนัด" in by["location"].content_desc


def test_toggle_state_and_input_recognized():
    by = _by_label(parse_ui_dump(THAI_CHAT_DUMP))
    assert by["Notifications"].type == "toggle"
    assert by["Notifications"].to_element()["value"] == "on"
    assert by["ส่งข้อความ"].type == "input"


def test_center_feeds_the_hands():
    order = next(n for n in parse_ui_dump(THAI_CHAT_DUMP) if n.text.startswith("CF"))
    # bounds [64,400][700,470] -> center (382, 435): read label + tap it,
    # no cached coordinate to go stale on a redesign.
    assert order.center() == (382, 435)


def test_pure_layout_chrome_is_dropped():
    nodes = parse_ui_dump(THAI_CHAT_DUMP)
    # The root FrameLayout has neither text nor content-desc; "Back" is a
    # decorative nav content-desc with no payload class -> both dropped.
    assert all(n.label != "" for n in nodes)
    assert "Back" not in {n.label for n in nodes}


def test_to_elements_is_screenstate_shaped():
    els = to_elements(parse_ui_dump(THAI_CHAT_DUMP))
    for el in els:
        assert "type" in el and "label" in el
    order = next(e for e in els if e["label"].startswith("CF"))
    assert order["type"] == "text"


def test_payload_hint_table_is_reachable():
    # Guard against typo'd keys silently disabling payload classification.
    assert set(PAYLOAD_HINTS) == {"sticker", "image", "location", "attachment", "reaction"}
