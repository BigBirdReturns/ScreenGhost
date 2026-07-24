"""Read-only privileged source facade for ScreenGhost Surface Teacher v0."""
from core.teacher_android import (
    AndroidTeacherDriver,
    capture_android_lesson,
    parse_uiautomator_xml,
)
from core.teacher_source_contract import SourceParseError
from core.teacher_web import (
    BrowserTeacherPage,
    DOM_SNAPSHOT_SCRIPT,
    capture_browser_lesson,
    parse_dom_snapshot,
)

__all__ = [
    "AndroidTeacherDriver",
    "BrowserTeacherPage",
    "DOM_SNAPSHOT_SCRIPT",
    "SourceParseError",
    "capture_android_lesson",
    "capture_browser_lesson",
    "parse_dom_snapshot",
    "parse_uiautomator_xml",
]
