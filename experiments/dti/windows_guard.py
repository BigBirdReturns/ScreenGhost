"""Official Roblox window discovery and target custody guards."""
from __future__ import annotations

import ctypes
import platform
import re
import sys
from typing import Any, Callable, Optional

from experiments.dti.windows_base import (
    DriverDoctor,
    EmergencyStop,
    TargetWindowError,
    WindowTarget,
    WindowsDriverUnavailable,
)


class WindowsTargetGuard:
    """Guard the visible official-client target before every capture or input."""

    KEY_MAP = {
        "w": 0x57,
        "a": 0x41,
        "s": 0x53,
        "d": 0x44,
        "q": 0x51,
        "e": 0x45,
        "space": 0x20,
        "shift": 0x10,
        "ctrl": 0x11,
        "tab": 0x09,
        "escape": 0x1B,
        "enter": 0x0D,
        "up": 0x26,
        "down": 0x28,
        "left": 0x25,
        "right": 0x27,
        "0": 0x30,
        "1": 0x31,
        "2": 0x32,
        "3": 0x33,
        "4": 0x34,
        "5": 0x35,
        "6": 0x36,
        "7": 0x37,
        "8": 0x38,
        "9": 0x39,
    }
    EXTENDED_KEYS = frozenset({"up", "down", "left", "right"})

    def __init__(
        self,
        target: WindowTarget = WindowTarget(),
        *,
        emergency_vk: int = 0x7B,
        camera_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.target = target
        self.emergency_vk = int(emergency_vk)
        self.camera_factory = camera_factory
        self._camera: Any = None
        self._dpi_configured = False

    @staticmethod
    def is_windows() -> bool:
        return sys.platform == "win32" or platform.system().casefold() == "windows"

    def available(self) -> bool:
        if not self.is_windows():
            return False
        try:
            if self.camera_factory is None:
                __import__("dxcam")
            return True
        except Exception:
            return False

    def _require_windows(self) -> None:
        if not self.is_windos():
            raise WindowsDriverUnavailable("WindowsGameDriver requires Windows 10 or 11")
        self._configure_dpi()

    def _configure_dpi(self) -> None:
        if self._dpi_configured or not self.is_windos():
            return
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
        self._dpi_configured = True

    def _emergency_active(self) -> bool:
        if not self.is_windos():
            return False
        user32 = ctypes.windll.user32
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = ctypes.c_short
        return bool(user32.GetAsyncKeyState(self.emergency_vk) & 0x8000)

    def _find_target_window(self) -> int:
        self._require_windows()
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        pattern = re.compile(self.target.title_pattern, re.IGNORECASE)
        matches: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def callback(hwnd: int, _lparam: int) -> bool:
            window = wintypes.HWND(hwnd)
            if not user32.IsWindowVisible(window):
                return True
            length = user32.GetWindowTextLengthW(window)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(window, buffer, len(buffer))
            if pattern.search(buffer.value):
                matches.append(int(hwnd))
            return True

        callback_ref = callback_type(callback)
        user32.EnumWindows(callback_ref, 0)
        if not matches:
            raise TargetWindowError(
                f"no visible window matched title pattern {self.target.title_pattern!r}"
            )
        foreground = int(user32.GetForegroundWindow() or 0)
        if foreground in matches:
            return foreground
        if len(matches) == 1:
            return matches[0]
        raise TargetWindowError(
            f"multiple Roblox-like windows matched and none was foreground: {matches}"
        )

    @staticmethod
    def _client_rect_screen(hwnd: int) -> tuple[int, int, int, int]:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetClientRect.restype = wintypes.BOOL
        user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        user32.ClientToScreen.restype = wintypes.BOOL
        window = wintypes.HWND(hwnd)
        rect = wintypes.RECT()
        if not user32.GetClientRect(window, ctypes.byref(rect)):
            raise TargetWindowError("GetClientRect failed")
        point = wintypes.POINT(rect.left, rect.top)
        if not user32.ClientToScreen(window, ctypes.byref(point)):
            raise TargetWindowError("ClientToScreen failed")
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        return (int(point.x), int(point.y), int(point.x + width), int(point.y + height))

    @staticmethod
    def _process_name(hwnd: int) -> Optional[str]:
        from ctypes import wintypes

        pid = wintypes.DWORD()
        user32 = ctypes.windll.user32
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
        if not pid.value:
            return None
        try:
            import psutil

            return psutil.Process(pid.value).name()
        except Exception:
            return None

    def _geometry_allowed(self, rect: tuple[int, int, int, int]) -> bool:
        if self.target.expected_client_size is None:
            return True
        left, top, right, bottom = rect
        actual = (right - left, bottom - top)
        expected = self.target.expected_client_size
        tolerance = self.target.size_tolerance_px
        return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))

    @staticmethod
    def _primary_monitor_rect() -> tuple[int, int, int, int]:
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32
        user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        user32.MonitorFromPoint.restype = ctypes.c_void_p
        point = wintypes.POINT(0, 0)
        monitor = user32.MonitorFromPoint(point, 1)
        if not monitor:
            raise TargetWindowError("could not resolve the primary display output")
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFO)]
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            raise TargetWindowError("GetMonitorInfoW failed for the primary display")
        rect = info.rcMonitor
        return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))

    @staticmethod
    def _rect_inside(
        inner: tuple[int, int, int, int],
        outer: tuple[int, int, int, int],
    ) -> bool:
        left, top, right, bottom = inner
        o_left, o_top, o_right, o_bottom = outer
        return (
            o_left <= left
            and o_top <= top
            and right <= o_right
            and bottom <= o_bottom
        )

    def _capture_output_allowed(self, rect: tuple[int, int, int, int]) -> bool:
        if not self.target.require_primary_output:
            return True
        return self._rect_inside(rect, self._primary_monitor_rect())

    def _guard(self) -> tuple[int, tuple[int, int, int, int]]:
        self._require_windows()
        if self._emergency_active():
            raise EmergencyStop("F12 emergency stop is active")
        hwnd = self._find_target_window()
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        if self.target.require_foreground and int(user32.GetForegroundWindow() or 0) != hwnd:
            raise TargetWindowError("refusing input because the Roblox window is not foreground")
        process_name = self._process_name(hwnd)
        if process_name is None:
            raise TargetWindowError("could not verify the target process name")
        allowed = {value.casefold() for value in self.target.allowed_process_names}
        if process_name.casefold() not in allowed:
            raise TargetWindowError(
                f"target process {process_name!r} is not in the official-client allowlist"
            )
        rect = self._client_rect_screen(hwnd)
        if not self._geometry_allowed(rect):
            actual = (rect[2] - rect[0], rect[3] - rect[1])
            raise TargetWindowError(
                f"client geometry {actual!r} does not match {self.target.expected_client_size!r}"
            )
        if not self._capture_output_allowed(rect):
            raise TargetWindowError(
                "Roblox client must lie completely on the primary display for DXcam capture"
            )
        return hwnd, rect

    def doctor(self) -> DriverDoctor:
        reasons: list[str] = []
        windows = self.is_windows()
        if not windows:
            return DriverDoctor(
                windows=False,
                dxcam_importable=False,
                target_found=False,
                target_foreground=False,
                process_name=None,
                process_allowed=False,
                client_size=None,
                geometry_allowed=False,
                capture_output_allowed=False,
                emergency_stop_active=False,
                reasons=("Windows host required",),
            )

        try:
            if self.camera_factory is None:
                __import__("dxcam")
            dxcam_importable = True
        except Exception as exc:
            dxcam_importable = False
            reasons.append(f"dxcam unavailable: {exc}")

        emergency = self._emergency_active()
        if emergency:
            reasons.append("F12 emergency stop is active")

        try:
            hwnd = self._find_target_window()
            target_found = True
        except Exception as exc:
            hwnd = 0
            target_found = False
            reasons.append(str(exc))

        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        foreground = bool(hwnd and int(user32.GetForegroundWindow() or 0) == int(hwnd))
        if target_found and self.target.require_foreground and not foreground:
            reasons.append("Roblox window is not foreground")

        process_name = self._process_name(hwnd) if hwnd else None
        allowed_names = {value.casefold() for value in self.target.allowed_process_names}
        process_allowed = bool(process_name and process_name.casefold() in allowed_names)
        if target_found and not process_allowed:
            reasons.append(f"unverified target process: {process_name!r}")

        rect = self._client_rect_screen(hwnd) if hwnd else None
        client_size = (
            (rect[2] - rect[0], rect[3] - rect[1]) if rect is not None else None
        )
        geometry_allowed = bool(rect and self._geometry_allowed(rect))
        if target_found and not geometry_allowed:
            reasons.append(
                f"client geometry {client_size!r} does not match {self.target.expected_client_size!r}"
            )

        try:
            capture_output_allowed = bool(rect and self._capture_output_allowed(rect))
        except Exception as exc:
            capture_output_allowed = False
            reasons.append(str(exc))
        if target_found and rect and not capture_output_allowed:
            reasons.append("Roblox client is outside the admitted primary DXcam output")

        return DriverDoctor(
            windows=windows,
            dxcam_importable=dxcam_importable,
            target_found=target_found,
            target_foreground=foreground,
            process_name=process_name,
            process_allowed=process_allowed,
            client_size=client_size,
            geometry_allowed=geometry_allowed,
            capture_output_allowed=capture_output_allowed,
            emergency_stop_active=emergency,
            reasons=tuple(dict.fromkeys(reasons)),
        )
