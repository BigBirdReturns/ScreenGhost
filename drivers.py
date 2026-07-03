from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional
import re
import subprocess

if TYPE_CHECKING:  # imaging stack is only needed at screencap time
    from PIL import Image


# A network ADB target looks like "host:port" (e.g. "192.168.1.5:5555").
# USB serials and emulator ids ("emulator-5554") never match this shape.
_NETWORK_TARGET = re.compile(r"^.+:\d+$")


def is_network_target(device: Optional[str]) -> bool:
    """True if ``device`` is a TCP/IP ADB endpoint rather than a local one.

    The whole security thesis of fast hands is that execution is physically
    local: there is no remote surface to attack or to switch off. ADB over
    TCP/IP (``adb connect host:port``) quietly reintroduces exactly that
    surface, so we treat such targets as non-local.
    """
    return bool(device) and bool(_NETWORK_TARGET.match(device.strip()))


class RemoteHandsError(RuntimeError):
    """Raised when someone tries to move the hands over a network transport."""


class DeviceDriver(ABC):
    """OS/device abstraction for Screen Ghost fast-hands execution."""

    name: str = "unknown"

    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def list_devices(self) -> List[str]:
        pass

    @abstractmethod
    def screencap(self, device: Optional[str] = None) -> Image.Image:
        pass

    @abstractmethod
    def tap(self, x: int, y: int, device: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
        device: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def type_text(self, text: str, device: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def keyevent(self, keycode: int, device: Optional[str] = None) -> None:
        pass


class AndroidAdbDriver(DeviceDriver):
    name = "android_adb"

    def __init__(self, allow_network: bool = False):
        # Local-only by default: the hands stay on a physically attached
        # device so no remote system can reach or disable them. Opt in
        # explicitly (eyes open) if you really want network ADB.
        self.allow_network = allow_network

    def _guard_local(self, device: Optional[str]) -> None:
        if not self.allow_network and is_network_target(device):
            raise RemoteHandsError(
                f"refusing network ADB target {device!r}: fast hands are "
                "local-only. Pass allow_network=True to override."
            )

    def available(self) -> bool:
        try:
            result = subprocess.run(["adb", "version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def list_devices(self) -> List[str]:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")[1:]
        return [line.split("\t")[0] for line in lines if "\tdevice" in line]

    def screencap(self, device: Optional[str] = None) -> "Image.Image":
        from io import BytesIO

        from PIL import Image

        self._guard_local(device)
        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["exec-out", "screencap", "-p"])

        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        if proc.returncode != 0:
            raise RuntimeError(f"adb screencap failed: {proc.stderr.decode('utf-8', errors='ignore')}")

        return Image.open(BytesIO(proc.stdout)).convert("RGB")

    def dump_ui_xml(self, device: Optional[str] = None) -> str:
        """Exact on-screen text via the OS view tree — not pixels, no OCR.

        UiAutomator dumps the live window hierarchy as XML in which every
        visible node's ``text`` attribute is the real Unicode string the OS is
        rendering. Complex scripts (Thai, etc.) come back exact because nothing
        is ever recognized from an image. This is the boring, pre-vision-model
        read path; the VLM is only a fallback for genuine pixels.

        Dump-to-file then ``cat`` is used rather than ``dump /dev/tty`` because
        the file path is the portable behavior across Android versions.
        """
        self._guard_local(device)
        base = ["adb"]
        if device:
            base.extend(["-s", device])

        remote = "/sdcard/window_dump.xml"
        dump = subprocess.run(
            base + ["shell", "uiautomator", "dump", remote],
            capture_output=True, timeout=20,
        )
        if dump.returncode != 0:
            raise RuntimeError(
                f"adb uiautomator dump failed: {dump.stderr.decode('utf-8', errors='ignore')}"
            )

        cat = subprocess.run(
            base + ["exec-out", "cat", remote], capture_output=True, timeout=10
        )
        if cat.returncode != 0:
            raise RuntimeError(
                f"adb cat {remote} failed: {cat.stderr.decode('utf-8', errors='ignore')}"
            )
        # UiAutomator writes UTF-8; decode leniently so a stray byte never
        # takes down an otherwise-exact hierarchy read.
        return cat.stdout.decode("utf-8", errors="replace")

    def tap(self, x: int, y: int, device: Optional[str] = None) -> None:
        self._guard_local(device)
        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["shell", "input", "tap", str(int(x)), str(int(y))])
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"adb tap failed: {proc.stderr.decode('utf-8', errors='ignore')}")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
        device: Optional[str] = None,
    ) -> None:
        self._guard_local(device)
        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"adb swipe failed: {proc.stderr.decode('utf-8', errors='ignore')}")

    def type_text(self, text: str, device: Optional[str] = None) -> None:
        self._guard_local(device)
        escaped = text.replace(" ", "%s").replace("'", "\\'").replace('"', '\\"')
        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["shell", "input", "text", escaped])
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"adb text failed: {proc.stderr.decode('utf-8', errors='ignore')}")

    def keyevent(self, keycode: int, device: Optional[str] = None) -> None:
        self._guard_local(device)
        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["shell", "input", "keyevent", str(keycode)])
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"adb keyevent failed: {proc.stderr.decode('utf-8', errors='ignore')}")
