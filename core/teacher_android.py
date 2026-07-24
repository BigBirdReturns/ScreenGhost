"""Read-only Android UI Automator adapter for Surface Teacher v0."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

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
    image_png,
    sha256_text,
)

_BOUNDS_RE = re.compile(r"^\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]$")


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
    if tail == "radiobutton":
        return "radio"
    if tail in {"recyclerview", "listview"} or scrollable:
        return "list"
    if tail == "imageview":
        return "icon_button" if clickable else "image"
    if tail == "textview":
        return "button" if clickable else "text"
    if tail == "webview":
        return "web_view"
    if tail in {
        "viewgroup",
        "linearlayout",
        "framelayout",
        "relativelayout",
        "constraintlayout",
    }:
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


def parse_uiautomator_xml(
    xml_text: str, *, viewport: Tuple[int, int]
) -> Tuple[TeacherNode, ...]:
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
        clickable = bool_value(attrs.get("clickable"))
        focusable = bool_value(attrs.get("focusable"))
        scrollable = bool_value(attrs.get("scrollable"))
        interactive = clickable or focusable or scrollable
        resource_id = attrs.get("resource-id") or ""
        source_ref = f"{resource_id}@{path}" if resource_id else path
        bounds = _parse_android_bounds(attrs.get("bounds", ""))
        if visible and bounds is None and interactive:
            raise SourceParseError(
                f"interactive UI Automator node has no usable bounds: {source_ref}"
            )

        current_parent = parent_ref
        if visible and bounds is not None:
            class_name = attrs.get("class", "")
            role = _android_role(
                class_name, clickable=clickable, scrollable=scrollable
            )
            text = clean_text(attrs.get("text"))
            description = clean_text(attrs.get("content-desc"))
            sensitive = bool_value(attrs.get("password")) or "password" in class_name.lower()

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
        raise SourceParseError(
            "UI Automator hierarchy exposed no visible bounded nodes"
        )

    # UI Automator commonly exposes a clickable Android preference row as an
    # unlabeled layout whose visible title lives on a non-clickable TextView
    # descendant.  That title is the row's accessible name for semantic replay,
    # even though the raw dump does not copy it onto the clickable ancestor.
    # Prefer Android's conventional ``id/title`` descendant and otherwise use
    # the first visible labeled descendant in tree order.
    by_ref = {node.source_ref: node for node in nodes}

    def is_descendant(candidate: TeacherNode, ancestor_ref: str) -> bool:
        parent_ref = candidate.parent_ref
        seen = set()
        while parent_ref and parent_ref not in seen:
            if parent_ref == ancestor_ref:
                return True
            seen.add(parent_ref)
            parent = by_ref.get(parent_ref)
            parent_ref = parent.parent_ref if parent is not None else None
        return False

    enriched: List[TeacherNode] = []
    for node in nodes:
        structural_click_target = (
            node.interactive
            and not node.label
            and node.role in {"group", "control"}
        )
        if not structural_click_target:
            enriched.append(node)
            continue
        labeled_descendants = [
            candidate
            for candidate in nodes
            if candidate.label
            and not candidate.sensitive
            and is_descendant(candidate, node.source_ref)
        ]
        if not labeled_descendants:
            enriched.append(node)
            continue
        title_descendants = [
            candidate
            for candidate in labeled_descendants
            if candidate.source_ref.split("@", 1)[0].endswith(("id/title", ":title"))
        ]
        label_node = (title_descendants or labeled_descendants)[0]
        enriched.append(
            replace(
                node,
                role="menu_item",
                label=label_node.label,
                label_source=(
                    "descendant:title"
                    if title_descendants
                    else "descendant:visible-label"
                ),
            )
        )
    return tuple(enriched)


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
    """Read one aligned Android frame and hierarchy pair, then compile it."""
    if alignment_attempts < 1:
        raise SourceParseError("alignment_attempts must be positive")
    for _attempt in range(alignment_attempts):
        before_png, before_size = image_png(driver.screencap(device))
        xml_text = driver.dump_ui_xml(device)
        after_png, after_size = image_png(driver.screencap(device))
        if before_size != after_size:
            continue
        if canonical_pixel_sha256(before_png) != canonical_pixel_sha256(after_png):
            continue
        parsed = parse_uiautomator_xml(xml_text, viewport=after_size)
        return compile_lesson(
            after_png,
            surface_id=surface_id,
            source_kind=SourceKind.ANDROID_UIAUTOMATOR,
            source_payload_sha256=sha256_text(xml_text),
            nodes=parsed,
            policy=policy,
            app_version=app_version,
            locale=locale,
        )
    raise SourceParseError(
        f"surface changed while pairing pixels with UI Automator XML across "
        f"{alignment_attempts} bounded attempt(s)"
    )
