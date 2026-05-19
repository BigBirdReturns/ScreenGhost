from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
from PIL import Image
import subprocess


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

    def screencap(self, device: Optional[str] = None) -> Image.Image:
        from io import BytesIO

        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["exec-out", "screencap", "-p"])

        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        if proc.returncode != 0:
            raise RuntimeError(f"adb screencap failed: {proc.stderr.decode('utf-8', errors='ignore')}")

        return Image.open(BytesIO(proc.stdout)).convert("RGB")

    def tap(self, x: int, y: int, device: Optional[str] = None) -> None:
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
        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"adb swipe failed: {proc.stderr.decode('utf-8', errors='ignore')}")

    def type_text(self, text: str, device: Optional[str] = None) -> None:
        escaped = text.replace(" ", "%s").replace("'", "\\'").replace('"', '\\"')
        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["shell", "input", "text", escaped])
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"adb text failed: {proc.stderr.decode('utf-8', errors='ignore')}")

    def keyevent(self, keycode: int, device: Optional[str] = None) -> None:
        cmd = ["adb"]
        if device:
            cmd.extend(["-s", device])
        cmd.extend(["shell", "input", "keyevent", str(keycode)])
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"adb keyevent failed: {proc.stderr.decode('utf-8', errors='ignore')}")
