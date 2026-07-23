"""Read-only DOM adapter for Surface Teacher v0."""
from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from core.pixel_evidence import png_dimensions
from core.surface_teacher import (
    LessonArtifact,
    LessonPolicy,
    Rect,
    SourceKind,
    TeacherNode,
    compile_lesson,
)
from core.teacher_source_contract import (
    SourceParseError,
    bool_value,
    canonical_pixel_sha256,
    clean_text,
    sha256_text,
)


def _dom_role(record: Mapping[str, Any]) -> str:
    explicit = clean_text(record.get("role"))
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
    return "text" if clean_text(record.get("text")) else "group"


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
    if (
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes))
        and len(raw) == 4
    ):
        try:
            x, y, width, height = map(float, raw)
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return Rect(x, y, x + width, y + height)
    return None


def parse_dom_snapshot(
    records: Iterable[Mapping[str, Any]], *, viewport: Tuple[int, int]
) -> Tuple[TeacherNode, ...]:
    """Parse page-produced DOM records into source-neutral teacher nodes."""
    width, height = viewport
    if width <= 0 or height <= 0:
        raise SourceParseError(f"invalid viewport: {viewport}")
    nodes: List[TeacherNode] = []
    for index, record in enumerate(records):
        if not bool_value(record.get("visible"), True):
            continue
        interactive = bool_value(record.get("interactive"))
        bounds = _dom_bounds(record)
        source_ref = str(
            record.get("source_ref")
            or record.get("selector")
            or record.get("node_id")
            or f"dom:{index}"
        )
        if bounds is None:
            if interactive:
                raise SourceParseError(
                    f"interactive DOM node has no usable bounds: {source_ref}"
                )
            continue
        role = _dom_role(record)
        sensitive = str(record.get("input_type") or "").lower() == "password"
        accessible_name = clean_text(record.get("accessible_name"))
        aria_label = clean_text(record.get("aria_label"))
        text = clean_text(record.get("text"))
        if role == "text_field":
            label = accessible_name or aria_label
            value = clean_text(record.get("value"))
        else:
            label = accessible_name or aria_label or text
            value = (
                clean_text(record.get("value"))
                if role in {"checkbox", "radio", "switch"}
                else None
            )
        label_source = (
            "accessible_name"
            if accessible_name
            else "aria_label"
            if aria_label
            else "text"
            if text
            else None
        )
        raw_states = record.get("states") or {}
        if not isinstance(raw_states, Mapping):
            raise SourceParseError(f"DOM states must be an object for {source_ref}")
        states = tuple(sorted((str(key), str(value)) for key, value in raw_states.items()))
        nodes.append(
            TeacherNode(
                source_ref=source_ref,
                role=role,
                bounds=bounds,
                label=label,
                value=value,
                interactive=interactive,
                enabled=bool_value(record.get("enabled"), True),
                visible=True,
                parent_ref=(
                    str(record["parent_ref"])
                    if record.get("parent_ref") is not None
                    else None
                ),
                states=states,
                label_source=label_source,
                raw_type=str(record.get("tag") or record.get("raw_type") or "") or None,
                sensitive=sensitive,
            )
        )
    if not nodes:
        raise SourceParseError("DOM snapshot exposed no visible bounded nodes")
    return tuple(nodes)


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
    """Capture one aligned page and DOM pair from a caller-owned page."""
    if alignment_attempts < 1:
        raise SourceParseError("alignment_attempts must be positive")
    for _attempt in range(alignment_attempts):
        before = page.screenshot(type="png")
        if not isinstance(before, (bytes, bytearray)):
            raise SourceParseError(
                "page.screenshot(type='png') did not return bytes"
            )
        records = page.evaluate(DOM_SNAPSHOT_SCRIPT)
        if not isinstance(records, list):
            raise SourceParseError("DOM snapshot script did not return a list")
        after = page.screenshot(type="png")
        if not isinstance(after, (bytes, bytearray)):
            raise SourceParseError(
                "page.screenshot(type='png') did not return bytes"
            )
        before_bytes, after_bytes = bytes(before), bytes(after)
        before_viewport = png_dimensions(before_bytes)
        after_viewport = png_dimensions(after_bytes)
        if before_viewport is None or after_viewport is None:
            raise SourceParseError("browser screenshot is not a readable PNG")
        if before_viewport != after_viewport:
            continue
        if canonical_pixel_sha256(before_bytes) != canonical_pixel_sha256(after_bytes):
            continue
        payload = json.dumps(
            records, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        parsed = parse_dom_snapshot(records, viewport=after_viewport)
        return compile_lesson(
            after_bytes,
            surface_id=surface_id,
            source_kind=SourceKind.WEB_DOM,
            source_payload_sha256=sha256_text(payload),
            nodes=parsed,
            policy=policy,
            app_version=app_version,
            locale=locale,
        )
    raise SourceParseError(
        f"surface changed while pairing pixels with DOM structure across "
        f"{alignment_attempts} bounded attempt(s)"
    )
