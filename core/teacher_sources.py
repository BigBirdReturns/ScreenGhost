"""Privileged structure adapters for ScreenGhost Surface Teacher v0.

Adapters are read-only. They turn UI Automator XML or a browser DOM snapshot into
``TeacherNode`` records and then pair those records with a screenshot. They do not
expose or invoke any input method.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from PIL import Image

from core.surface_teacher import (
    LessonArtifact,
    LessonPolicy,
    LessonRefused,
    Rect,
    SourceKind,
    TeacherNode,
    compile_lesson,
)

_BOUNDS_RE = re.compile(r"^\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]$")


class SourceParseError(LessonRefused):
    """The privileged source could not be interpreted without guessing."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_pixel_sha256(png_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
        image = image.convert("RGB")
    except Exception as exc:
        raise SourceParseError(f"capture is not a decodable PNG: {exc}") from exc
    header = json.dumps({"mode": image.mode, "size": image.size}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def _image_png(image: Any) -> Tuple[bytes, Tuple[int, int]]:
    if not hasattr(image, "save") or not hasattr(image, "size"):
        raise SourceParseError("screencap did not return an image-like object")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), tuple(image.size)


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact or None


def _android_role(class_name: str, *, clickable: bool, scrollable: bool) -> str:
    tail = (class_name or "").rsplit(".", 1)[-1].lower()
    if tail in {"button", "materialbutton"}:
        return "button"
    if tail in {"imagebutton", "floatingactionbutton"}:
        return "icon_button"
    if tail in {"edittext", "textfield"}:
        return "text_field"
    if tail in {"switch", "switchcompat"}:
        return "switch"
    if tail in {"checkbox", "checkedtextview"}:
        return "checkbox"
    if tail in {"radiobutton"}:
        return "radio"
    if tail in {"recyclerview", "listview"} or scrollable:
        return "list"
    if tail in {"imageview"}:
        return "icon_button" if clickable else "image"
    if tail in {"textview"}:
        return "button" if clickable else "text"
    if tail in {"webview"}:
        return "web_view"
    if tail in {"viewgroup", "linearlayout", "framelayout", "relativelayout", "constraintlayout"}:
        return "group"
    return "control" if clickable else "unknown"


def _parse_android_bounds(raw: str) -> Optional[Rect]:
    match = _BOUNDS_RE.match(raw or "")
    if not match:
        return None
    x1, y1, x2, y2 = map(float, match.groups())
    if x2 <= x1 or y2 <= y1:
        return None
    return Rect(x1, y1, x2, y2)


def parse_uiautomator_xml(xml_text: str, *, viewport: Tuple[int, int]) -> Tuple[TeacherNode, ...]:
    """Parse a UI Automator hierarchy into visible teacher nodes.

    Visible interactive nodes with no usable bounds are refused. Noninteractive
    implementation nodes with no geometry are ignored because they cannot be
    correlated with pixels.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SourceParseError(f"invalid UI Automator XML: {exc}") from exc
    width, height = viewport
    if width <= 0 or height <= 0:
        raise SourceParseError(f"invalid viewport: {viewport}")

    nodes: List[TeacherNode] = []

    def visit(element: ET.Element, path: str, parent_ref: Optional[str]) -> None:
        attrs = element.attrib
        visible = attrs.get("visible-to-user", "true").lower() != "false"
        clickable = _bool(attrs.get("clickable"))
        focusable = _bool(attrs.get("focusable"))
        scrollable = _bool(attrs.get("scrollable"))
        interactive = clickable or focusable or scrollable
        resource_id = attrs.get("resource-id") or ""
        # Android resource IDs repeat legitimately inside lists. The structural
        # path makes the teacher locator unique without exposing it downstream.
        source_ref = f"{resource_id}@{path}" if resource_id else path
        bounds = _parse_android_bounds(attrs.get("bounds", ""))
        if visible and bounds is None and interactive:
            raise SourceParseError(f"interactive UI Automator node has no usable bounds: {source_ref}")

        current_parent = parent_ref
        if visible and bounds is not None:
            class_name = attrs.get("class", "")
            role = _android_role(class_name, clickable=clickable, scrollable=scrollable)
            text = _text(attrs.get("text"))
            description = _text(attrs.get("content-desc"))
            sensitive = _bool(attrs.get("password")) or "password" in class_name.lower()

            if role == "text_field":
                label = description
                label_source = "content-desc" if description else None
                value = text
            else:
                label = text or description
                label_source = "text" if text else "content-desc" if description else None
                value = None

            states = tuple(
                sorted(
                    {
                        "checked": attrs.get("checked", "false"),
                        "selected": attrs.get("selected", "false"),
                        "focused": attrs.get("focused", "false"),
                        "scrollable": attrs.get("scrollable", "false"),
                    }.items()
                )
            )
            nodes.append(
                TeacherNode(
                    source_ref=source_ref,
                    role=role,
                    bounds=bounds,
                    label=label,
                    value=value,
                    interactive=interactive,
                    enabled=attrs.get("enabled", "true").lower() != "false",
                    visible=True,
                    parent_ref=parent_ref,
                    states=states,
                    label_source=label_source,
                    raw_type=class_name or None,
                    sensitive=sensitive,
                )
            )
            current_parent = source_ref

        for index, child in enumerate(list(element)):
            visit(child, f"{path}/{child.tag}[{index}]", current_parent)

    visit(root, f"/{root.tag}[0]", None)
    if not nodes:
        raise SourceParseError("UI Automator hierarchy exposed no visible bounded nodes")
    return tuple(nodes)


def _dom_role(record: Mapping[str, Any]) -> str:
    explicit = _text(record.get("role"))
    if explicit:
        role = explicit.lower().replace("-", "_")
        aliases = {
            "textbox": "text_field",
            "searchbox": "text_field",
            "menuitem": "menu_item",
            "listitem": "list_item",
        }
        return aliases.get(role, role)

    tag = str(record.get("tag") or "").lower()
    input_type = str(record.get("input_type") or "").lower()
    if tag == "button":
        return "button"
    if tag == "a":
        return "link"
    if tag in {"input", "textarea", "select"}:
        return {
            "checkbox": "checkbox",
            "radio": "radio",
            "button": "button",
            "submit": "button",
            "reset": "button",
        }.get(input_type, "text_field")
    if tag in {"ul", "ol"}:
        return "list"
    if tag == "li":
        return "list_item"
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return "heading"
    if tag == "img":
        return "image"
    return "text" if _text(record.get("text")) else "group"


def _dom_bounds(record: Mapping[str, Any]) -> Optional[Rect]:
    raw = record.get("bounds")
    if isinstance(raw, Mapping):
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw["width"])
            height = float(raw["height"])
        except (KeyError, TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return Rect(x, y, x + width, y + height)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 4:
        try:
            x, y, width, height = map(float, raw)
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return Rect(x, y, x + width, y + height)
    return None


def parse_dom_snapshot(records: Iterable[Mapping[str, Any]], *, viewport: Tuple[int, int]) -> Tuple[TeacherNode, ...]:
    """Parse page-produced DOM records into source-neutral teacher nodes."""
    width, height = viewport
    if width <= 0 or height <= 0:
        raise SourceParseError(f"invalid viewport: {viewport}")
    nodes: List[TeacherNode] = []
    for index, record in enumerate(records):
        if not _bool(record.get("visible"), True):
            continue
        interactive = _bool(record.get("interactive"))
        bounds = _dom_bounds(record)
        source_ref = str(record.get("source_ref") or record.get("selector") or record.get("node_id") or f"dom:{index}")
        if bounds is None:
            if interactive:
                raise SourceParseError(f"interactive DOM node has no usable bounds: {source_ref}")
            continue
        role = _dom_role(record)
        sensitive = str(record.get("input_type") or "").lower() == "password"
        accessible_name = _text(record.get("accessible_name"))
        aria_label = _text(record.get("aria_label"))
        text = _text(record.get("text"))
        if role == "text_field":
            label = accessible_name or aria_label
            value = _text(record.get("value"))
        else:
            label = accessible_name or aria_label or text
            value = _text(record.get("value")) if role in {"checkbox", "radio", "switch"} else None
        label_source = (
            "accessible_name" if accessible_name else "aria_label" if aria_label else "text" if text else None
        )
        raw_states = record.get("states") or {}
        if not isinstance(raw_states, Mapping):
            raise SourceParseError(f"DOM states must be an object for {source_ref}")
        states = tuple(sorted((str(k), str(v)) for k, v in raw_states.items()))
        nodes.append(
            TeacherNode(
                source_ref=source_ref,
                role=role,
                bounds=bounds,
                label=label,
                value=value,
                interactive=interactive,
                enabled=_bool(record.get("enabled"), True),
                visible=True,
                parent_ref=str(record["parent_ref"]) if record.get("parent_ref") is not None else None,
                states=states,
                label_source=label_source,
                raw_type=str(record.get("tag") or record.get("raw_type") or "") or None,
                sensitive=sensitive,
            )
        )
    if not nodes:
        raise SourceParseError("DOM snapshot exposed no visible bounded nodes")
    return tuple(nodes)


@runtime_checkable
class AndroidTeacherDriver(Protocol):
    """Read-only subset of the existing Android driver."""

    def screencap(self, device: Optional[str] = None) -> Any: ...

    def dump_ui_xml(self, device: Optional[str] = None) -> str: ...


def capture_android_lesson(
    driver: AndroidTeacherDriver,
    *,
    surface_id: str,
    device: Optional[str] = None,
    policy: LessonPolicy = LessonPolicy(),
    app_version: Optional[str] = None,
    locale: Optional[str] = None,
    alignment_attempts: int = 3,
) -> LessonArtifact:
    """Read one aligned Android frame/hierarchy pair and compile a lesson.

    A hierarchy is accepted only when equal canonical pixel frames bracket the
    XML dump. This prevents a transient animation or navigation from teaching a
    structural tree against the wrong image. The retry count is bounded and no
    input method is available to this function.
    """
    if alignment_attempts < 1:
        raise SourceParseError("alignment_attempts must be positive")
    for _attempt in range(alignment_attempts):
        before_png, before_size = _image_png(driver.screencap(device))
        xml_text = driver.dump_ui_xml(device)
        after_png, after_size = _image_png(driver.screencap(device))
        if before_size != after_size:
            continue
        if _canonical_pixel_sha256(before_png) != _canonical_pixel_sha256(after_png):
            continue
        nodes = parse_uiautomator_xml(xml_text, viewport=after_size)
        return compile_lesson(
            after_png,
            surface_id=surface_id,
            source_kind=SourceKind.ANDROID_UIAUTOMATOR,
            source_payload_sha256=_sha256_text(xml_text),
            nodes=nodes,
            policy=policy,
            app_version=app_version,
            locale=locale,
        )
    raise SourceParseError(
        f"surface changed while pairing pixels with UI Automator XML across "
        f"{alignment_attempts} bounded attempt(s)"
    )


DOM_SNAPSHOT_SCRIPT = r"""() => {
  const visible = (el, r, style) =>
    style.display !== 'none' && style.visibility !== 'hidden' &&
    Number(style.opacity || 1) !== 0 && r.width > 0 && r.height > 0 &&
    r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
  const selector = (el) => {
    if (el.id) return `#${CSS.escape(el.id)}`;
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === 1 && parts.length < 8) {
      let part = cur.tagName.toLowerCase();
      if (cur.parentElement) {
        const peers = [...cur.parentElement.children].filter(x => x.tagName === cur.tagName);
        if (peers.length > 1) part += `:nth-of-type(${peers.indexOf(cur) + 1})`;
      }
      parts.unshift(part);
      cur = cur.parentElement;
    }
    return parts.join(' > ');
  };
  const nameOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim();
    const labelledBy = (el.getAttribute('aria-labelledby') || '').trim();
    if (labelledBy) {
      const text = labelledBy.split(/\s+/).map(id => document.getElementById(id)?.innerText || '').join(' ').trim();
      if (text) return text;
    }
    if (el.labels && el.labels.length) {
      const text = [...el.labels].map(label => label.innerText || '').join(' ').trim();
      if (text) return text;
    }
    return (el.getAttribute('alt') || el.getAttribute('title') ||
            el.getAttribute('placeholder') || '').trim() || null;
  };
  return [...document.querySelectorAll('body *')].map((el) => {
    const r = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role');
    const inputType = tag === 'input' ? (el.getAttribute('type') || 'text') : null;
    const interactiveRoles = new Set([
      'button','link','checkbox','radio','switch','textbox','searchbox','combobox',
      'menuitem','option','tab','slider','spinbutton','treeitem'
    ]);
    const interactive = ['button','a','input','textarea','select','summary'].includes(tag) ||
      interactiveRoles.has(role) || el.tabIndex >= 0 || typeof el.onclick === 'function';
    const accessibleName = nameOf(el);
    return {
      source_ref: selector(el),
      parent_ref: el.parentElement ? selector(el.parentElement) : null,
      tag,
      role,
      input_type: inputType,
      accessible_name: accessibleName,
      aria_label: el.getAttribute('aria-label'),
      text: interactive ? (el.innerText || '').trim().slice(0, 500) :
                          (tag.match(/^h[1-6]$/) ? (el.innerText || '').trim().slice(0, 500) : null),
      value: ['input','textarea','select'].includes(tag) && inputType !== 'password' ? el.value : null,
      interactive,
      enabled: !el.disabled,
      visible: visible(el, r, style),
      bounds: {
        x: r.x * devicePixelRatio,
        y: r.y * devicePixelRatio,
        width: r.width * devicePixelRatio,
        height: r.height * devicePixelRatio
      },
      states: {
        checked: 'checked' in el ? String(!!el.checked) : 'false',
        selected: el.getAttribute('aria-selected') || 'false',
        expanded: el.getAttribute('aria-expanded') || 'false'
      }
    };
  }).filter(x => x.visible);
}"""


@runtime_checkable
class BrowserTeacherPage(Protocol):
    """Read-only page subset used by ``capture_browser_lesson``."""

    def screenshot(self, **kwargs: Any) -> bytes: ...

    def evaluate(self, expression: str) -> Any: ...


def capture_browser_lesson(
    page: BrowserTeacherPage,
    *,
    surface_id: str,
    policy: LessonPolicy = LessonPolicy(),
    app_version: Optional[str] = None,
    locale: Optional[str] = None,
    alignment_attempts: int = 3,
) -> LessonArtifact:
    """Capture one aligned page/DOM pair from a caller-owned page.

    The DOM snapshot is accepted only when equal canonical pixel frames bracket
    the evaluation. No browser is launched and no input method is invoked here.
    """
    if alignment_attempts < 1:
        raise SourceParseError("alignment_attempts must be positive")
    from core.pixel_evidence import png_dimensions

    for _attempt in range(alignment_attempts):
        before = page.screenshot(type="png")
        if not isinstance(before, (bytes, bytearray)):
            raise SourceParseError("page.screenshot(type='png') did not return bytes")
        records = page.evaluate(DOM_SNAPSHOT_SCRIPT)
        if not isinstance(records, list):
            raise SourceParseError("DOM snapshot script did not return a list")
        after = page.screenshot(type="png")
        if not isinstance(after, (bytes, bytearray)):
            raise SourceParseError("page.screenshot(type='png') did not return bytes")
        before_bytes, after_bytes = bytes(before), bytes(after)
        before_viewport = png_dimensions(before_bytes)
        after_viewport = png_dimensions(after_bytes)
        if before_viewport is None or after_viewport is None:
            raise SourceParseError("browser screenshot is not a readable PNG")
        if before_viewport != after_viewport:
            continue
        if _canonical_pixel_sha256(before_bytes) != _canonical_pixel_sha256(after_bytes):
            continue
        payload = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        nodes = parse_dom_snapshot(records, viewport=after_viewport)
        return compile_lesson(
            after_bytes,
            surface_id=surface_id,
            source_kind=SourceKind.WEB_DOM,
            source_payload_sha256=_sha256_text(payload),
            nodes=nodes,
            policy=policy,
            app_version=app_version,
            locale=locale,
        )
    raise SourceParseError(
        f"surface changed while pairing pixels with DOM structure across "
        f"{alignment_attempts} bounded attempt(s)"
    )
