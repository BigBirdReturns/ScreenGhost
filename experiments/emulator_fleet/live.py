"""Live emulator instance adapter for Surface Teacher and semantic replay.

The adapter is deliberately instance-scoped.  It never moves the host pointer or
sends host keyboard events.  Pixels and UI structure are read through the vendor's
per-instance ADB wrapper; motor calls are likewise sent to that Android instance.

PR #13 is imported lazily by :func:`compile_android_teacher_projection`, so the
emulated campaign and provider parser tests remain runnable in an isolated overlay.
"""
from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from PIL import Image

from experiments.emulator_fleet.providers.base import FleetProviderError
from experiments.emulator_fleet.schema import InstanceRef


@dataclass(frozen=True)
class LiveActionResult:
    accepted: bool
    injected: bool
    action_type: str
    target_key: Optional[str]
    target_label: Optional[str]
    transition_due_ms: Optional[float]
    reason: str


@runtime_checkable
class AndroidFleetProvider(Protocol):
    def capture_png(self, instance: InstanceRef) -> bytes: ...
    def dump_ui_xml(self, instance: InstanceRef) -> str: ...
    def tap(self, instance: InstanceRef, x: int, y: int) -> Any: ...
    def swipe(
        self,
        instance: InstanceRef,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> Any: ...
    def input_text(self, instance: InstanceRef, text: str) -> Any: ...
    def keyevent(self, instance: InstanceRef, keycode: int | str) -> Any: ...


TeacherCompiler = Callable[[bytes, str, str, str], Mapping[str, Any]]


def compile_android_teacher_projection(
    png_bytes: bytes,
    xml_text: str,
    surface_id: str,
    app_family: str,
) -> Mapping[str, Any]:
    """Compile an Android lesson through Surface Teacher PR #13.

    The structural screen name is derived from the teacher-blind control hash.  A
    caller may later map that family to a human name, but the live compiler never
    invents one from a package-specific selector.
    """

    try:
        from core.surface_teacher import LessonPolicy, SourceKind, compile_lesson
        from core.teacher_android import parse_uiautomator_xml
    except ImportError as exc:  # pragma: no cover - exercised in the real checkout
        raise FleetProviderError(
            "live teaching requires ScreenGhost Surface Teacher PR #13 on PYTHONPATH"
        ) from exc

    try:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
        viewport = tuple(image.size)
    except Exception as exc:
        raise FleetProviderError(f"provider returned an unreadable PNG: {exc}") from exc
    nodes = parse_uiautomator_xml(xml_text, viewport=viewport)
    artifact = compile_lesson(
        png_bytes,
        surface_id=surface_id,
        source_kind=SourceKind.ANDROID_UIAUTOMATOR,
        source_payload_sha256=hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
        nodes=nodes,
        policy=LessonPolicy(retain_values=False, write_element_crops=True),
    )
    projection = dict(artifact.lesson.runtime_projection())
    projection["app_family"] = app_family
    projection["screen_name"] = "screen:" + str(projection.get("control_hash") or "unknown")[:16]
    return projection


class LiveFleetInstanceAdapter:
    """One running vendor instance exposed through the campaign backend contract."""

    def __init__(
        self,
        provider: AndroidFleetProvider,
        instance: InstanceRef,
        *,
        app_family: str,
        surface_id: Optional[str] = None,
        teacher_compiler: TeacherCompiler = compile_android_teacher_projection,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.ref = instance
        self.instance_id = instance.instance_id
        self.app_family = str(app_family)
        self.surface_id = surface_id or f"{instance.vendor.value}.{instance.name}"
        self.teacher_compiler = teacher_compiler
        self._sleep = sleep
        self._monotonic = monotonic
        self._teacher_reads = 0
        self._pixel_reads = 0
        self._actions_injected = 0
        self._last_projection: Optional[Mapping[str, Any]] = None
        self._last_teacher_png: Optional[bytes] = None
        self._last_size: Optional[tuple[int, int]] = None

    @property
    def pending(self) -> bool:
        # Android changes asynchronously, but transaction settlement is observed
        # from pixels rather than delegated to a vendor-specific pending flag.
        return False

    @property
    def teacher_reads(self) -> int:
        return self._teacher_reads

    @property
    def pixel_reads(self) -> int:
        return self._pixel_reads

    @property
    def actions_injected(self) -> int:
        return self._actions_injected

    def now_ms(self) -> float:
        return self._monotonic() * 1000.0

    def capture_png(self) -> bytes:
        value = self.provider.capture_png(self.ref)
        self._pixel_reads += 1
        try:
            image = Image.open(io.BytesIO(value))
            image.load()
            self._last_size = tuple(image.size)
        except Exception as exc:
            raise FleetProviderError(f"provider returned an unreadable PNG: {exc}") from exc
        return bytes(value)

    @staticmethod
    def _canonical_pixel_sha256(png_bytes: bytes) -> str:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
        image = image.convert("RGB")
        header = f"{image.mode}:{image.size[0]}x{image.size[1]}".encode("ascii")
        return hashlib.sha256(header + image.tobytes()).hexdigest()

    @property
    def last_teacher_png(self) -> Optional[bytes]:
        return self._last_teacher_png

    def capture_teacher(self, *, alignment_attempts: int = 3) -> Mapping[str, Any]:
        if alignment_attempts < 1:
            raise ValueError("alignment_attempts must be positive")
        for _ in range(alignment_attempts):
            before = self.capture_png()
            xml = self.provider.dump_ui_xml(self.ref)
            after = self.capture_png()
            if self._canonical_pixel_sha256(before) != self._canonical_pixel_sha256(after):
                continue
            projection = dict(
                self.teacher_compiler(after, xml, self.surface_id, self.app_family)
            )
            self._teacher_reads += 1
            self._last_projection = projection
            self._last_teacher_png = after
            return projection
        raise FleetProviderError(
            f"surface changed while pairing pixels and UI structure across {alignment_attempts} attempt(s)"
        )

    def advance(self, milliseconds: float) -> None:
        if milliseconds < 0:
            raise ValueError("cannot advance time backwards")
        self._sleep(float(milliseconds) / 1000.0)

    def _size(self) -> tuple[int, int]:
        if self._last_size is None:
            self.capture_png()
        assert self._last_size is not None
        return self._last_size

    @staticmethod
    def _result_ok(result: Any) -> bool:
        if hasattr(result, "ok"):
            return bool(result.ok)
        if hasattr(result, "returncode"):
            return int(result.returncode) == 0
        return bool(result)

    def tap_normalized(self, x: float, y: float) -> LiveActionResult:
        width, height = self._size()
        px = min(width - 1, max(0, int(round(float(x) * width))))
        py = min(height - 1, max(0, int(round(float(y) * height))))
        return self.tap_source_pixels(px, py)

    def tap_source_pixels(self, x: int, y: int) -> LiveActionResult:
        result = self.provider.tap(self.ref, int(x), int(y))
        ok = self._result_ok(result)
        if ok:
            self._actions_injected += 1
        return LiveActionResult(ok, ok, "tap", None, None, None, "ADB tap injected" if ok else "ADB tap failed")

    def type_text(self, text: str) -> LiveActionResult:
        result = self.provider.input_text(self.ref, str(text))
        ok = self._result_ok(result)
        if ok:
            self._actions_injected += 1
        return LiveActionResult(ok, ok, "text", None, None, None, "ADB text injected" if ok else "ADB text failed")

    def back(self) -> LiveActionResult:
        result = self.provider.keyevent(self.ref, 4)
        ok = self._result_ok(result)
        if ok:
            self._actions_injected += 1
        return LiveActionResult(ok, ok, "back", None, None, None, "ADB Back injected" if ok else "ADB Back failed")

    def swipe_normalized(self, path, duration_ms: float = 300.0) -> LiveActionResult:
        points = tuple(path)
        if len(points) < 2:
            return LiveActionResult(False, False, "swipe", None, None, None, "swipe needs two points")
        width, height = self._size()
        x1, y1 = points[0]
        x2, y2 = points[-1]
        result = self.provider.swipe(
            self.ref,
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height)),
            int(round(duration_ms)),
        )
        ok = self._result_ok(result)
        if ok:
            self._actions_injected += 1
        return LiveActionResult(ok, ok, "swipe", None, None, None, "ADB swipe injected" if ok else "ADB swipe failed")

    def long_press_normalized(self, x: float, y: float, duration_ms: float) -> LiveActionResult:
        # Android's input swipe with identical endpoints is the portable long-press
        # primitive available through both MEMUC and LDPlayer's ADB wrappers.
        return self.swipe_normalized(((x, y), (x, y)), duration_ms)

    def current_screen(self) -> str:
        if self._last_projection is None:
            return "unknown"
        return str(
            self._last_projection.get("screen_name")
            or self._last_projection.get("screen_key")
            or "unknown"
        )
