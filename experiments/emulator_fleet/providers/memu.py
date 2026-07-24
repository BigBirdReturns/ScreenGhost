"""MEmu fleet provider backed by the documented ``memuc`` command.

Only read-only discovery runs by default.  Start, stop, clone, configuration, and
Android input require ``apply=True``.  Removal and import/export replacement also
require the explicit destructive token inherited from :class:`VendorProviderBase`.
"""
from __future__ import annotations

import csv
import io
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from experiments.emulator_fleet.command import CommandResult, CommandRunner
from experiments.emulator_fleet.providers.base import (
    FleetProviderError,
    ProviderCapability,
    VendorProviderBase,
)
from experiments.emulator_fleet.schema import (
    EmulatorVendor,
    InstanceProfile,
    InstanceRef,
    InstanceStatus,
)


_RUNNING_WORDS = {"1", "true", "running", "start", "started", "on"}
_STOPPED_WORDS = {"0", "false", "stopped", "stop", "off", "shutdown"}


def _status_from_fields(fields: Sequence[str]) -> InstanceStatus:
    lowered = {str(v).strip().lower() for v in fields}
    if lowered.intersection(_RUNNING_WORDS):
        return InstanceStatus.RUNNING
    if lowered.intersection(_STOPPED_WORDS):
        return InstanceStatus.STOPPED
    # A positive PID is strong evidence that the instance is running.
    for value in reversed(fields):
        try:
            if int(str(value).strip(), 0) > 0:
                return InstanceStatus.RUNNING
        except ValueError:
            continue
    return InstanceStatus.UNKNOWN


def parse_memuc_listvms(output: str) -> tuple[InstanceRef, ...]:
    """Parse CSV and table-like ``memuc listvms`` output defensively.

    MEmu versions have emitted both comma-separated and padded table output.  The
    parser never fabricates a row: a line must begin with an integer index and
    contain a non-empty title.
    """

    instances: list[InstanceRef] = []
    for raw in output.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.lower().startswith(("index", "vm index", "total")):
            continue
        fields: list[str]
        if "," in line:
            fields = [part.strip() for part in next(csv.reader([line]))]
        else:
            fields = [part.strip() for part in re.split(r"\s{2,}|\t+", line) if part.strip()]
            if len(fields) < 2:
                fields = line.split()
        if len(fields) < 2:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        name = fields[1].strip().strip('"')
        if not name:
            continue
        window_handle = None
        pid = None
        numeric: list[int] = []
        for value in fields[2:]:
            try:
                numeric.append(int(value, 0))
            except ValueError:
                continue
        if numeric:
            window_handle = numeric[0] if numeric[0] > 0 else None
            pid = next((value for value in reversed(numeric) if value > 0), None)
        status = _status_from_fields(fields[2:])
        instances.append(
            InstanceRef(
                vendor=EmulatorVendor.MEMU,
                instance_id=f"memu:{index}",
                name=name,
                index=index,
                status=status,
                window_handle=window_handle,
                pid=pid,
                metadata={"raw_fields": fields[2:]},
            )
        )
    return tuple(sorted(instances, key=lambda row: (row.index if row.index is not None else 10**9, row.name)))


class MEmuFleetProvider(VendorProviderBase):
    vendor = EmulatorVendor.MEMU

    DEFAULT_PATHS = (
        Path(r"C:\Program Files\Microvirt\MEmu\memuc.exe"),
        Path(r"C:\Program Files\Microvirt\MEmuHyperv\memuc.exe"),
        Path(r"D:\Program Files\Microvirt\MEmu\memuc.exe"),
    )

    def __init__(
        self,
        executable: str | Path,
        *,
        runner: CommandRunner,
        apply: bool = False,
        destructive_token: Optional[str] = None,
    ) -> None:
        super().__init__(
            executable,
            runner=runner,
            apply=apply,
            destructive_token=destructive_token,
        )

    @classmethod
    def discover(cls) -> Optional[Path]:
        return next((path for path in cls.DEFAULT_PATHS if path.exists()), None)

    def capability(self) -> ProviderCapability:
        installed = self.executable.exists()
        return ProviderCapability(
            vendor=self.vendor,
            executable=str(self.executable),
            installed=installed,
            lifecycle_cli=True,
            clone_supported=True,
            configuration_supported=True,
            adb_wrapper_supported=True,
            macro_import_supported=True,
            # Operation Recorder is exposed through MEmu's UI/file catalog, not
            # by a documented MEMUC play command. ScreenGhost therefore imports
            # or observes macros but does not claim headless vendor playback.
            macro_execution_supported=False,
            missing=(() if installed else ("memuc.exe",)),
            notes=(
                "operation-recorder artifacts can be cataloged and distilled",
                "vendor macro playback remains a UI-owned baseline",
                "all Android control uses instance-scoped MEMUC/ADB commands",
            ),
        )

    @staticmethod
    def _selector(instance: InstanceRef) -> tuple[str, str]:
        if instance.vendor is not EmulatorVendor.MEMU:
            raise FleetProviderError("instance does not belong to MEmu provider")
        if instance.index is not None:
            return ("-i", str(instance.index))
        return ("-n", instance.name)

    def list_instances(self, *, running_only: bool = False) -> tuple[InstanceRef, ...]:
        args = ["listvms"]
        if running_only:
            args.append("--running")
        result = self._run(args, timeout_s=20, metadata={"operation": "list_instances"})
        if not result.ok:
            if result.missing_executable:
                return ()
            raise FleetProviderError(
                f"memuc listvms failed: {result.stderr_text() or result.stdout_text()}"
            )
        return parse_memuc_listvms(result.stdout_text())

    def is_running(self, instance: InstanceRef) -> bool:
        result = self._run(
            ["isvmrunning", *self._selector(instance)],
            timeout_s=15,
            metadata={"operation": "is_running", "instance": instance.instance_id},
        )
        if not result.ok:
            return False
        text = result.stdout_text().strip().lower()
        return any(word in text for word in ("running", "true", "1")) and "not" not in text

    def start(self, instance: InstanceRef) -> CommandResult:
        return self._run(
            ["start", *self._selector(instance)],
            mutation=True,
            timeout_s=90,
            metadata={"operation": "start", "instance": instance.instance_id},
        )

    def stop(self, instance: InstanceRef) -> CommandResult:
        return self._run(
            ["stop", *self._selector(instance)],
            mutation=True,
            timeout_s=90,
            metadata={"operation": "stop", "instance": instance.instance_id},
        )

    def stop_all(self) -> CommandResult:
        return self._run(
            ["stopall"], mutation=True, timeout_s=90, metadata={"operation": "stop_all"}
        )

    def reboot(self, instance: InstanceRef) -> CommandResult:
        return self._run(
            ["reboot", *self._selector(instance)],
            mutation=True,
            timeout_s=120,
            metadata={"operation": "reboot", "instance": instance.instance_id},
        )

    def create(self, *, android_image: Optional[str] = None) -> CommandResult:
        args = ["create"]
        if android_image:
            args.append(str(android_image))
        return self._run(
            args,
            mutation=True,
            timeout_s=300,
            metadata={"operation": "create", "android_image": android_image},
        )

    def clone(self, source: InstanceRef, *, new_name: str) -> tuple[CommandResult, ...]:
        name = self._require_name(new_name)
        clone = self._run(
            ["clone", *self._selector(source), "-r", name],
            mutation=True,
            timeout_s=600,
            metadata={"operation": "clone", "source": source.instance_id, "new_name": name},
        )
        return (clone,)

    def rename(self, instance: InstanceRef, *, new_name: str) -> CommandResult:
        name = self._require_name(new_name)
        return self._run(
            ["rename", *self._selector(instance), name],
            mutation=True,
            timeout_s=60,
            metadata={"operation": "rename", "instance": instance.instance_id, "new_name": name},
        )

    def remove(self, instance: InstanceRef) -> CommandResult:
        return self._run(
            ["remove", *self._selector(instance)],
            mutation=True,
            destructive=True,
            timeout_s=300,
            metadata={"operation": "remove", "instance": instance.instance_id},
        )

    def export_instance(self, instance: InstanceRef, path: str | Path) -> CommandResult:
        target = Path(path)
        return self._run(
            ["export", *self._selector(instance), str(target)],
            mutation=True,
            timeout_s=1800,
            metadata={"operation": "export", "instance": instance.instance_id, "path": str(target)},
        )

    def import_instance(self, path: str | Path) -> CommandResult:
        source = Path(path)
        return self._run(
            ["import", str(source)],
            mutation=True,
            destructive=True,
            timeout_s=1800,
            metadata={"operation": "import", "path": str(source)},
        )

    def get_config(self, instance: InstanceRef, key: str) -> CommandResult:
        return self._run(
            ["getconfigex", *self._selector(instance), str(key)],
            timeout_s=20,
            metadata={"operation": "get_config", "instance": instance.instance_id, "key": key},
        )

    def set_config(self, instance: InstanceRef, key: str, value: Any) -> CommandResult:
        return self._run(
            ["setconfigex", *self._selector(instance), str(key), str(value)],
            mutation=True,
            timeout_s=60,
            metadata={"operation": "set_config", "instance": instance.instance_id, "key": key},
        )

    def configure(self, instance: InstanceRef, profile: InstanceProfile) -> tuple[CommandResult, ...]:
        return (
            self.set_config(instance, "cpus", profile.cpu_cores),
            self.set_config(instance, "memory", profile.memory_mb),
            # Current MEMUC builds may return success for ``custom_resolution``
            # while leaving the stored dimensions unchanged.  The individual
            # keys have stable readback through ``getconfigex``.
            self.set_config(instance, "resolution_width", profile.width),
            self.set_config(instance, "resolution_height", profile.height),
            self.set_config(instance, "vbox_dpi", profile.dpi),
            self.set_config(instance, "cache_mode", 0),
            self.set_config(instance, "disable_resize", 1),
        )

    def sort_windows(self) -> CommandResult:
        return self._run(
            ["sortwin"], mutation=True, timeout_s=30, metadata={"operation": "sort_windows"}
        )

    def install_app(self, instance: InstanceRef, apk_path: str | Path) -> CommandResult:
        return self._run(
            ["installapp", *self._selector(instance), str(Path(apk_path))],
            mutation=True,
            timeout_s=300,
            metadata={"operation": "install_app", "instance": instance.instance_id},
        )

    def start_app(self, instance: InstanceRef, package_activity: str) -> CommandResult:
        return self._run(
            ["startapp", *self._selector(instance), package_activity],
            mutation=True,
            timeout_s=60,
            metadata={"operation": "start_app", "instance": instance.instance_id},
        )

    def stop_app(self, instance: InstanceRef, package_name: str) -> CommandResult:
        return self._run(
            ["stopapp", *self._selector(instance), package_name],
            mutation=True,
            timeout_s=60,
            metadata={"operation": "stop_app", "instance": instance.instance_id},
        )

    def send_key(self, instance: InstanceRef, key: str) -> CommandResult:
        return self._run(
            ["sendkey", *self._selector(instance), key],
            mutation=True,
            timeout_s=20,
            metadata={"operation": "send_key", "instance": instance.instance_id, "key": key},
        )

    def input_text(self, instance: InstanceRef, text: str) -> CommandResult:
        return self._run(
            ["input", *self._selector(instance), str(text)],
            mutation=True,
            timeout_s=30,
            metadata={"operation": "input_text", "instance": instance.instance_id, "text_length": len(text)},
        )

    def adb(self, instance: InstanceRef, command: str, *, timeout_s: float = 30.0) -> CommandResult:
        return self._run(
            [*self._selector(instance), "adb", str(command)],
            mutation=self._adb_is_mutating(command),
            timeout_s=timeout_s,
            metadata={"operation": "adb", "instance": instance.instance_id, "command": command},
        )

    @staticmethod
    def _adb_is_mutating(command: str) -> bool:
        text = command.strip().lower()
        read_prefixes = (
            "shell uiautomator dump",
            "shell cat ",
            "shell getprop",
            "shell dumpsys",
            "shell wm size",
            "shell wm density",
            "shell pm list",
            "pull ",
        )
        return not text.startswith(read_prefixes)

    def tap(self, instance: InstanceRef, x: int, y: int) -> CommandResult:
        return self.adb(instance, f"shell input tap {int(x)} {int(y)}", timeout_s=20)

    def swipe(
        self,
        instance: InstanceRef,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> CommandResult:
        return self.adb(
            instance,
            f"shell input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration_ms)}",
            timeout_s=30,
        )

    def keyevent(self, instance: InstanceRef, keycode: int | str) -> CommandResult:
        return self.adb(instance, f"shell input keyevent {keycode}", timeout_s=20)

    def pull_file(self, instance: InstanceRef, remote: str, local: str | Path) -> CommandResult:
        local_path = Path(local)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return self.adb(instance, f'pull "{remote}" "{local_path}"', timeout_s=60)

    def capture_png(self, instance: InstanceRef) -> bytes:
        if not self.apply:
            raise FleetProviderError("capture_png requires apply=True because it writes a temporary guest file")
        remote = f"/sdcard/screenghost_{instance.index if instance.index is not None else 'vm'}.png"
        first = self.adb(instance, f"shell screencap -p {remote}", timeout_s=30)
        if not first.ok:
            raise FleetProviderError(f"MEmu screencap failed: {first.stderr_text()}")
        with tempfile.TemporaryDirectory(prefix="sg-memu-") as tmp:
            local = Path(tmp) / "screen.png"
            pull = self.pull_file(instance, remote, local)
            if not pull.ok or not local.exists():
                raise FleetProviderError(f"MEmu screenshot pull failed: {pull.stderr_text()}")
            return local.read_bytes()

    def dump_ui_xml(self, instance: InstanceRef) -> str:
        if not self.apply:
            raise FleetProviderError("dump_ui_xml requires apply=True because it writes a temporary guest file")
        remote = f"/sdcard/screenghost_{instance.index if instance.index is not None else 'vm'}.xml"
        first = self.adb(instance, f"shell uiautomator dump {remote}", timeout_s=45)
        if not first.ok:
            raise FleetProviderError(f"MEmu UI dump failed: {first.stderr_text()}")
        with tempfile.TemporaryDirectory(prefix="sg-memu-") as tmp:
            local = Path(tmp) / "window.xml"
            pull = self.pull_file(instance, remote, local)
            if not pull.ok or not local.exists():
                raise FleetProviderError(f"MEmu UI dump pull failed: {pull.stderr_text()}")
            return local.read_text(encoding="utf-8", errors="replace")
