from __future__ import annotations

import inspect
from io import BytesIO

import pytest
from PIL import Image

from core.teacher_sources import (
    SourceParseError,
    capture_android_lesson,
    capture_browser_lesson,
    parse_dom_snapshot,
    parse_uiautomator_xml,
)
from .support import ANDROID_XML, DOM_RECORDS, android_node, png_bytes


def test_uiautomator_parser_preserves_roles_unicode_and_parentage():
    parsed = parse_uiautomator_xml(
        ANDROID_XML.replace("Continue", "ดำเนินการต่อ"), viewport=(400, 800)
    )
    root = android_node(parsed, "root")
    button = android_node(parsed, "continue")
    name = android_node(parsed, "name")
    password = android_node(parsed, "password")
    assert button.role == "button"
    assert button.label == "ดำเนินการต่อ"
    assert button.parent_ref == root.source_ref
    assert name.role == "text_field"
    assert name.label == "Name" and name.value == "Jonathan"
    assert password.sensitive is True
    assert not any(node.source_ref.startswith("hidden@") for node in parsed)


def test_uiautomator_repeated_resource_ids_receive_unique_teacher_locators():
    xml = """<hierarchy><node resource-id="root" class="android.widget.FrameLayout" bounds="[0,0][100,200]">
      <node resource-id="item" class="android.widget.Button" text="One" clickable="true" bounds="[0,0][100,50]" />
      <node resource-id="item" class="android.widget.Button" text="Two" clickable="true" bounds="[0,60][100,110]" />
    </node></hierarchy>"""
    parsed = parse_uiautomator_xml(xml, viewport=(100, 200))
    refs = [node.source_ref for node in parsed if node.source_ref.startswith("item@")]
    assert len(refs) == 2 and len(set(refs)) == 2


def test_uiautomator_refuses_unbounded_interactive_node():
    bad = ANDROID_XML.replace('bounds="[40,180][360,250]"', 'bounds=""', 1)
    with pytest.raises(SourceParseError, match="interactive UI Automator node"):
        parse_uiautomator_xml(bad, viewport=(400, 800))


def test_dom_parser_uses_accessible_semantics_and_filters_hidden_nodes():
    parsed = parse_dom_snapshot(DOM_RECORDS, viewport=(400, 800))
    by_ref = {node.source_ref: node for node in parsed}
    assert by_ref["#continue"].role == "button"
    assert by_ref["#continue"].label == "Continue"
    assert by_ref["#name"].role == "text_field"
    assert by_ref["#name"].value == "Jonathan"
    assert "#display-none" not in by_ref


def test_dom_parser_does_not_invent_an_unlabeled_control_name():
    record = {
        "source_ref": "#mystery",
        "tag": "button",
        "visible": True,
        "interactive": True,
        "bounds": [10, 10, 30, 30],
    }
    node = parse_dom_snapshot([record], viewport=(100, 100))[0]
    assert node.label is None and node.label_source is None


class FakeAndroidDriver:
    def __init__(self):
        self.calls = []

    def screencap(self, device=None):
        self.calls.append(("screencap", device))
        return Image.open(BytesIO(png_bytes())).convert("RGB")

    def dump_ui_xml(self, device=None):
        self.calls.append(("dump_ui_xml", device))
        return ANDROID_XML

    def tap(self, *args, **kwargs):
        self.calls.append(("tap", args, kwargs))
        raise AssertionError("teacher attempted input")

    def swipe(self, *args, **kwargs):
        self.calls.append(("swipe", args, kwargs))
        raise AssertionError("teacher attempted input")

    def type_text(self, *args, **kwargs):
        self.calls.append(("type_text", args, kwargs))
        raise AssertionError("teacher attempted input")

    def keyevent(self, *args, **kwargs):
        self.calls.append(("keyevent", args, kwargs))
        raise AssertionError("teacher attempted input")


def test_android_capture_reads_frame_and_tree_only():
    driver = FakeAndroidDriver()
    lesson = capture_android_lesson(
        driver, surface_id="com.example/.Login", device="emulator-5554"
    )
    assert lesson.lesson.source_kind == "android_uiautomator"
    assert driver.calls == [
        ("screencap", "emulator-5554"),
        ("dump_ui_xml", "emulator-5554"),
        ("screencap", "emulator-5554"),
    ]


class FakePage:
    def __init__(self):
        self.calls = []

    def screenshot(self, **kwargs):
        self.calls.append(("screenshot", kwargs))
        return png_bytes()

    def evaluate(self, expression):
        self.calls.append(("evaluate", expression))
        return DOM_RECORDS

    def click(self, *args, **kwargs):
        raise AssertionError("teacher attempted input")


def test_browser_capture_uses_caller_owned_page_and_never_clicks():
    page = FakePage()
    lesson = capture_browser_lesson(page, surface_id="https://example.invalid/login")
    assert lesson.lesson.source_kind == "web_dom"
    assert [call[0] for call in page.calls] == ["screenshot", "evaluate", "screenshot"]


class RacingAndroidDriver(FakeAndroidDriver):
    def __init__(self):
        super().__init__()
        self._frame = 0

    def screencap(self, device=None):
        self.calls.append(("screencap", device))
        self._frame += 1
        fill = (30, 80 + self._frame, 180)
        return Image.open(BytesIO(png_bytes(button_fill=fill))).convert("RGB")


def test_android_capture_refuses_a_pixel_structure_race_without_input():
    driver = RacingAndroidDriver()
    with pytest.raises(SourceParseError, match="surface changed while pairing pixels"):
        capture_android_lesson(
            driver, surface_id="com.example/.Login", alignment_attempts=2
        )
    assert all(call[0] in {"screencap", "dump_ui_xml"} for call in driver.calls)
    assert len(driver.calls) == 6


class RacingPage(FakePage):
    def __init__(self):
        super().__init__()
        self._frame = 0

    def screenshot(self, **kwargs):
        self.calls.append(("screenshot", kwargs))
        self._frame += 1
        return png_bytes(button_fill=(30, 80 + self._frame, 180))


def test_browser_capture_refuses_a_pixel_dom_race_without_clicking():
    page = RacingPage()
    with pytest.raises(SourceParseError, match="surface changed while pairing pixels"):
        capture_browser_lesson(
            page, surface_id="https://example.invalid/login", alignment_attempts=2
        )
    assert [call[0] for call in page.calls] == [
        "screenshot",
        "evaluate",
        "screenshot",
        "screenshot",
        "evaluate",
        "screenshot",
    ]


def test_teacher_modules_have_no_process_or_input_transport_imports():
    import core.surface_teacher as teacher
    import core.teacher_android as android
    import core.teacher_source_contract as contract
    import core.teacher_sources as sources
    import core.teacher_web as web

    source = "".join(
        inspect.getsource(module)
        for module in (teacher, contract, android, web, sources)
    )
    assert "import subprocess" not in source
    assert "from drivers" not in source
    assert "FastAPI" not in source
    assert "uvicorn" not in source
