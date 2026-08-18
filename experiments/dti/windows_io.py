"""DXcam capture and bounded ordinary Win32 input."""
from __future__ import annotations

import ctypes
import time
from typing import Any, Sequence

from experiments.dti.windows_base import EmergencyStop, WindowsDriverError
from experiments.dti.windows_guard import WindowsTargetGuard


class WindowsGameDriver(WindowsTargetGuard):
    def _camera_instance(self) -> Any:
        if self._camera is not None:
            return self._camera
        factory = self.camera_factory
        if factory is None:
            import dxcam

            factory = dxcam.create
        self._camera = factory(output_color="RGB", processor_backend="numpy")
        return self._camera

    def capture(self) -> Any:
        """Capture the current visible client area as a Pillow RGB image."""
        _hwnd, rect = self._guard()
        camera = self._camera_instance()
        frame = camera.grab(region=rect, new_frame_only=False)
        if frame is None:
            raise WindowsDriverError("DXcam returned no frame")
        from PIL import Image

        return Image.fromarray(frame).convert("RGB")

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.release()
            finally:
                self._camera = None

    @staticmethod
    def _input_types() -> tuple[Any, Any, Any, Any, Any]:
        from ctypes import wintypes

        ULONG_PTR = wintypes.WPARAM

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class INPUTUNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _anonymous_ = ("union",)
            _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]

        return MOUSEINPUT, KEYBDINPUT, HARDWAREINPUT, INPUTUNION, INPUT

    @classmethod
    def _key_input(cls, key: str, *, up: bool = False) -> Any:
        _mouse, keyboard_type, _hardware, input_union, input_type = cls._input_types()
        normalized = key.casefold()
        if normalized not in cls.KEY_MAP:
            raise WindowsDriverError(f"unsupported key: {key!r}")
        vk = cls.KEY_MAP[normalized]
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        user32.MapVirtualKeyW.restype = wintypes.UINT
        scan = user32.MapVirtualKeyW(vk, 0)
        flags = 0x0008
        if normalized in cls.EXTENDED_KEYS:
            flags |= 0x0001
        if up:
            flags |= 0x0002
        keyboard = keyboard_type(0, scan, flags, 0, 0)
        return input_type(type=1, union=input_union(ki=keyboard))

    @classmethod
    def _mouse_input(
        cls,
        *,
        dx: int = 0,
        dy: int = 0,
        data: int = 0,
        flags: int,
    ) -> Any:
        mouse_type, _keyboard, _hardware, input_union, input_type = cls._input_types()
        mouse = mouse_type(dx, dy, data, flags, 0, 0)
        return input_type(type=0, union=input_union(mi=mouse))

    @staticmethod
    def _send(inputs: Sequence[Any]) -> None:
        if not inputs:
            return
        input_type = type(inputs[0])
        array_type = input_type * len(inputs)
        array = array_type(*inputs)
        from ctypes import wintypes

        send_input = ctypes.windll.user32.SendInput
        send_input.argtypes = [wintypes.UINT, ctypes.POINTER(input_type), ctypes.c_int]
        send_input.restype = wintypes.UINT
        sent = send_input(len(inputs), array, ctypes.sizeof(input_type))
        if sent != len(inputs):
            raise WindowsDriverError(f"SendInput sent {sent} of {len(inputs)} events")

    def press(self, keys: Sequence[str]) -> None:
        self._guard()
        normalized = tuple(str(key).casefold() for key in keys)
        self._send([self._key_input(key) for key in normalized])
        time.sleep(0.04)
        self._send([self._key_input(key, up=True) for key in reversed(normalized)])

    def hold(self, keys: Sequence[str], duration_ms: int) -> None:
        self._guard()
        if not 1 <= int(duration_ms) <= 2500:
            raise WindowsDriverError("hold duration must remain in [1,2500] ms")
        normalized = tuple(str(key).casefold() for key in keys)
        self._send([self._key_input(key) for key in normalized])
        try:
            time.sleep(duration_ms / 1000.0)
        finally:
            self._send([self._key_input(key, up=True) for key in reversed(normalized)])

    def click_normalized(self, x: float, y: float) -> None:
        _hwnd, rect = self._guard()
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise WindowsDriverError("click point must lie in normalized client space")
        left, top, right, bottom = rect
        px = int(round(left + float(x) * max(1, right - left - 1)))
        py = int(round(top + float(y) * max(1, bottom - top - 1)))
        from ctypes import wintypes

        set_cursor_pos = ctypes.windll.user32.SetCursorPos
        set_cursor_pos.argtypes = [ctypes.c_int, ctypes.c_int]
        set_cursor_pos.restype = wintypes.BOOL
        if not set_cursor_pos(px, py):
            raise WindowsDriverError("SetCursorPos failed")
        self._send(
            [
                self._mouse_input(flags=0x0002),
                self._mouse_input(flags=0x0004),
            ]
        )

    def move_mouse(self, dx: int, dy: int) -> None:
        self._guard()
        if abs(int(dx)) > 1500 or abs(int(dy)) > 1500:
            raise WindowsDriverError("mouse delta exceeds the bounded movement")
        self._send([self._mouse_input(dx=int(dx), dy=int(dy), flags=0x0001)])

    def scroll(self, delta: int) -> None:
        self._guard()
        if int(delta) == 0 or abs(int(delta)) > 1200:
            raise WindowsDriverError("scroll delta is zero or exceeds the bound")
        self._send([self._mouse_input(data=int(delta), flags=0x0800)])

    def wait(self, duration_ms: int) -> None:
        if duration_ms < 0:
            raise WindowsDriverError("wait duration cannot be negative")
        if self._emergency_active():
            raise EmergencyStop("F12 emergency stop is active")
        time.sleep(duration_ms / 1000.0)
