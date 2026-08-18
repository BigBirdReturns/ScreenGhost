"""Official-client Windows eyes and hands for the DTI cartridge.

Capture uses DXcam against the visible Roblox window.  Input uses ordinary Win32
SendInput events.  The driver refuses to act unless the declared Roblox window is
foreground, its process name is allowlisted, its client geometry is within the
declared tolerance, and the emergency-stop key is not held.  It contains no
process-memory, injection, client-modification, or anti-cheat bypass path.
"""
from __future__ import annotations

import ctypes
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple


class WindowsDriverError(RuntimeError):
    pass


class WindowsDriverUnavailable(WindowsDriverError):
    pass


class TargetWindowError(WindowsDriverError):
    pass


class EmergencyStop(WindowsDriverError):
    pass


@dataclass(frozen=True)
class WindowTarget:
    title_pattern: str = r"Roblox"
    allowed_process_names: Tuple[str, ...] = (
        "RobloxPlayerBeta.exe",
        "Windows10Universal.exe",
    )
    expected_client_size: Optional[Tuple[int, int]] = (1600, 900)
    size_tolerance_px: int = 8
    require_foreground: bool = True

    def __post_init__(self) -> None:
        re.compile(self.title_pattern)
        if self.expected_client_size is not None:
            width, height = self.expected_client_size
            if width <= 0 or height <= 0:
                raise ValueError("expected_client_size must be positive")
        if self.size_tolerance_px < 0:
            raise ValueError("size_tolerance_px cannot be negative")


@dataclass(frozen=True)
class DriverDoctor:
    windows: bool
    dxcam_importable: bool
    target_found: bool
    target_foreground: bool
    process_name: Optional[str]
    process_allowed: bool
    client_size: Optional[Tuple[int, int]]
    geometry_allowed: bool
    emergency_stop_active: bool
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def ready(self) -> bool:
        return (
            self.windows
            and self.dxcam_importable
            and self.target_found
            and self.target_foreground
            and self.process_allowed
            and self.geometry_allowed
            and not self.emergency_stop_active
        )


