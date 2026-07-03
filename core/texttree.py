"""Exact-text UI extraction from the OS view tree — no OCR, no VLM.

The screenshot-into-a-vision-model loop was never the whole repertoire; it was
its weakest member. Android already exposes the live screen as a hierarchy of
text nodes via UiAutomator (``adb shell uiautomator dump``). Each visible node's
``text`` attribute is the **real Unicode string the OS is rendering** — so
complex scripts (Thai, and every script with no word breaks or stacked
diacritics) come back exact, because the text is never turned into an image to
be recognized in the first place.

This is deliberately boring, pre-2023 technology. That is the point: the robust
answer to "vision models can't read Thai reliably" is to not put a vision model
on the text path at all. The VLM is kept only as a last-resort fallback for
genuine pixels (a sticker's artwork, a map thumbnail) — see ``PAYLOAD_HINTS``.

Nothing here needs a model or a GPU. The parser is pure and testable offline;
only :func:`read_tree` touches a device, and only through the existing driver.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

# UiAutomator encodes bounds as "[x1,y1][x2,y2]".
_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

# content-desc / class keywords that mark a node as a non-text payload. When a
# node has no text but one of these, we surface it as a structured payload
# element instead of dropping it — this is what protocol APIs hand you for free
# and what a screenshot would force you to re-recognize from pixels.
PAYLOAD_HINTS: Dict[str, Tuple[str, ...]] = {
    "sticker": ("sticker",),
    "image": ("photo", "image", "picture", "gif"),
    "location": ("location", "map", "pin"),
    "attachment": ("attachment", "file", "document", "voice", "audio", "video"),
    "reaction": ("reaction", "reacted", "like", "heart"),
}


def _parse_bounds(raw: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not raw:
        return None
    m = _BOUNDS_RE.search(raw)
    if not m:
        return None
    x1, y1, x2, y2 = (int(v) for v in m.groups())
    return (x1, y1, x2, y2)


def _classify_payload(text: str, content_desc: str, cls: str) -> Optional[str]:
    """Return a payload type for a non-text node, or None.

    Matches on ``content-desc`` only — deliberately not the class name, since
    "ImageView"/"ImageButton" contain the literal substring "image" and would
    misclassify every icon as an image payload.
    """
    if text.strip():
        return None
    haystack = content_desc.lower()
    for payload_type, needles in PAYLOAD_HINTS.items():
        if any(n in haystack for n in needles):
            return payload_type
    return None


def _node_type(text: str, content_desc: str, cls: str, clickable: bool,
               checkable: bool) -> str:
    cls_l = cls.lower()
    payload = _classify_payload(text, content_desc, cls)
    if payload:
        return payload
    if checkable or "switch" in cls_l or "checkbox" in cls_l or "toggle" in cls_l:
        return "toggle"
    if "edittext" in cls_l:
        return "input"
    if "button" in cls_l or (clickable and 0 < len(text) <= 24):
        return "button"
    return "text"


@dataclass
class TextNode:
    """One visible UI node, read as exact text — not recognized from pixels."""

    text: str
    content_desc: str
    cls: str
    clickable: bool
    checkable: bool
    checked: bool
    bounds: Optional[Tuple[int, int, int, int]]

    @property
    def label(self) -> str:
        """What a human would read off this node."""
        return self.text or self.content_desc

    @property
    def type(self) -> str:
        return _node_type(self.text, self.content_desc, self.cls,
                          self.clickable, self.checkable)

    def center(self) -> Optional[Tuple[int, int]]:
        """Tap target — reading feeds the hands directly, no coordinate cache."""
        if not self.bounds:
            return None
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def to_element(self) -> Dict[str, Any]:
        """Shape-compatible with ScreenState's ``elements`` entries."""
        d: Dict[str, Any] = {"type": self.type, "label": self.label}
        if self.type == "toggle":
            d["value"] = "on" if self.checked else "off"
        if self.bounds is not None:
            d["bounds"] = self.bounds
        return d


def parse_ui_dump(xml: str) -> List[TextNode]:
    """Parse a UiAutomator window-hierarchy dump into text nodes.

    Pure and offline: feed it the XML from ``read_tree`` (or a fixture) and it
    returns every node that carries readable content — exact ``text`` first,
    then ``content-desc`` for non-text payloads. Chrome with neither is dropped.
    """
    root = ET.fromstring(xml)
    nodes: List[TextNode] = []
    for el in root.iter("node"):
        text = el.get("text", "") or ""
        content_desc = el.get("content-desc", "") or ""
        cls = el.get("class", "") or ""
        if not text.strip() and not content_desc.strip():
            continue  # pure layout/chrome — nothing to read
        node = TextNode(
            text=text,
            content_desc=content_desc,
            cls=cls,
            clickable=el.get("clickable") == "true",
            checkable=el.get("checkable") == "true",
            checked=el.get("checked") == "true",
            bounds=_parse_bounds(el.get("bounds")),
        )
        # Keep non-text nodes only when they classify as a real payload
        # (sticker/location/…), so decorative content-descs — "Back", nav
        # chrome, icon buttons — don't flood the reading output.
        if not text.strip() and node.type not in PAYLOAD_HINTS:
            continue
        nodes.append(node)
    return nodes


def to_elements(nodes: List[TextNode]) -> List[Dict[str, Any]]:
    return [n.to_element() for n in nodes]


def read_tree(device: Optional[str] = None, driver: Any = None) -> List[TextNode]:
    """Read the live view tree from a device via the driver (exact text).

    The device round-trip lives on the driver so it inherits the same
    local-only guard as the hands. Import is local to keep this module free of
    the imaging/driver stack for pure parsing use.
    """
    if driver is None:
        from drivers import AndroidAdbDriver

        driver = AndroidAdbDriver()
    xml = driver.dump_ui_xml(device)
    return parse_ui_dump(xml)
