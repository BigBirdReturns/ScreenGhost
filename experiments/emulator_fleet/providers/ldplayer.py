"""LDPlayer fleet provider backed by the documented ``dnconsole`` interface."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any, Optional

from experiments.emulator_fleet.command import CommandResult, CommandRunner
from experiments.emulator_fleet.providers.base import FleetProviderError, ProviderCapability, VendorProviderBase
from experiments.emulator_fleet.schema import EmulatorVendor, InstanceProfile, InstanceRef, InstanceStatus


def parse_ldplayer_list2(output: str) -> tuple[InstanceRef, ...]:
    """Parse ``dnconsole list2``.

    Official field order: index, title, top window handle, bind window handle,
    Android started, PID, VBox PID.
    """

    rows: list[InstanceRef] = []
    for raw in output.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or "," not in line:
            continue
        fields = [part.strip() for part in next(csv.reader([line]))]
        if len(fields) < 7:
            continue
        try:
            index = int(fields[0])
            top_handle = int(fields[2] or 0)
            started = int(fields[4] or 0)
            pid = int(fields[5] or 0)
            vbox_pid = int(fields[6] or 0)
        except ValueError:
            continue
        name = fields[1]
        if not name:
            continue
        rows.append(
            InstanceRef(
                vendor=EmulatorVendor.LDPLAYER,
                instance_id=f"ldplayer:{index}",
                name=name,
                index=index,
                status=InstanceStatus.RUNNING if started else InstanceStatus.STOPPED,
                window_handle=top_handle or None,
                pid=pid or None,
                metadata={"bind_window_handle": fields[3], "vbox_pid": vbox_pid or None},
            )
        )
    return tuple(sorted(rows, key=lambda row: row.index if row.index is not None else 10**9))


class LDPlayerFleetProvider(VendorProviderBase):
    vendor = EmulatorVendor.LDPLAYER

    DEFAULT_PATHS = (
        Path(r"C:\LDPlayer\LDPlayer9\dnconsole.exe"),
        Path(r"C:\Program Files\LDPlayer\LDPlayer9\dnconsole.exe"),
        Path(r"D:\LDPlayer\LDPlayer9\dnconsole.exe"),
        Path(r"C:\ChangZhi\LDPlayer\dnconsole.exe"),
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
            macro_execution_supported=True,
            missing=(() if installed else ("dnconsole.exe or ldconsole.exe",)),
            notes=(
                "coordinate macros can be parsed from the documented size/touch/wait/key/text syntax",
                "synchronizer behavior is treated as a coordinate baseline",
                "ScreenGhost semantic replay remains per-instance and independently verified",
            ),
        )

    @staticmethod
    def _selector(instance: InstanceRef) -> tuple[str, str]:
        if instance.vendor is not EmulatorVendor.LDPLAYER:
            raise FleetProviderError("instance does not belong to LDPlayer provider")
        if instance.index is not None:
            return ("--index", str(instance.index))
        return ("--name", instance.name)

    def list_instances(self, *, running_only: bool = False) -> tuple[InstanceRef, ...]:
        result = self._run(["list2"], timeout_s=20, metadata={"operation": "list_instances"})
        if not result.ok:
            if result.missing_executable:
                return ()
            raise FleetProviderError(
                f"LDPlayer list2 failed: {result.stderr_text() or result.stdout_text()}"
            )
        rows = parse_ldplayer_list2(result.stdout_text())
        return tuple(row for row in rows if not running_only or row.status is InstanceStatus.RUNNING)

    def start(self, instance: InstanceRef) -> CommandResult:
        return self._run(
            ["launch", *self._selector(instance)],
            mutation=True,
            timeout_s=120,
            metadata={"operation": "start", "instance": instance.instance_id},
        )

    def stop(self, instance: InstanceRef) -> CommandResult:
        return self._run(
            ["quit", *self._selector(instance)],
            mutation=True,
            timeout_s=90,
            metadata={"operation": "stop", "instance": instance.instance_id},
        )

    def stop_all(self) -> CommandResult:
        return self._run(["quitall"], mutation=True, timeout_s=90, metadata={"operation": "stop_all"})

    def create(self, *, name: str) -> CommandResult:
        value = self._require_name(name)
        return self._run(
            ["add", "--name", value],
            mutation=True,
            timeout_s=300,
            metadata={"operation": "create", "name": value},
        )

    def clone(self, source: InstanceRef, *, new_name: str) -> tuple[CommandResult, ...]:
        name = self._require_name(new_name)
        result = self._run(
            ["copy", "--name", name, "--from", str(source.index if source.index is not None else source.name)],
            mutation=True,
            timeout_s=600,
            metadata={"operation": "clone", "source": source.instance_id, "new_name": name},
        )
        return (result,)

    def remove(self, instance: InstanceRef) -> CommandResult:
        return self._run(
            ["remove", *self._selector(instance)],
            mutation=True,
            destructive=True,
            timeout_s=300,
            metadata={"operation": "remove", "instance": instance.instance_id},
        )

    def configure(self, instance: InstanceRef, profile: InstanceProfile) -> tuple[CommandResult, ...]:
        result = self._run(
            [
                "modify",
                *self._selector(instance),
                "--resolution",
                profile.resolution,
                "--cpu",
                str(profile.cpu_cores),
                "--memory",
                str(profile.memory_mb),
                "--autorotate",
                "0",
                "--lockwindow",
                "1",
            ],
            mutation=True,
            timeout_s=90,
            metadata={"operation": "configure", "instance": instance.instance_id, "profile": profile.to_dict()},
        )
        return (result,)

    def install_app(self, instance: InstanceRef, apk_path: str | Path) -> CommandResult:
        return self._run(
            ["installapp", *self._selector(instance), "--filename", str(Path(apk_path))],
            mutation=True,
            timeout_s=300,
            metadata={"operation": "install_app", "instance": instance.instance_id},
        )

    def start_app(self, instance: InstanceRef, package_name: str) -> CommandResult:
        return self._run(
            ["runapp", *self._selector(instance), "--packagename", package_name],
            mutation=True,
            timeout_s=60,
            metadata={"operation": "start_app", "instance": instance.instance_id},
        )

    def stop_app(self, instance: InstanceRef, package_name: str) -> CommandResult:
        return self._run(
            ["killapp", *self._selector(instance), "--packagename", package_name],
            mutation=True,
            timeout_s=60,
            metadata={"operation": "stop_app", "instance": instance.instance_id},
        )

    def adb(self, instance: InstanceRef, command: str, *, timeout_s: float = 30.0) -> CommandResult:
        return self._run(
            ["adb", *self._selector(instance), "--command", str(command)],
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

    def input_text(self, instance: InstanceRef, text: str) -> CommandResult:
        escaped = str(text).replace(" ", "%s")
        return self.adb(instance, f"shell input text {escaped}", timeout_s=30)

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
            raise FleetProviderError(f"LDPlayer screencap failed: {first.stderr_text()}")
        with tempfile.TemporaryDirectory(prefix="sg-ldplayer-") as tmp:
            local = Path(tmp) / "screen.png"
            pull = self.pull_file(instance, remote, local)
            if not pull.ok or not local.exists():
                raise FleetProviderError(f"LDPlayer screenshot pull failed: {pull.stderr_text()}")
            return local.read_bytes()

    def dump_ui_xml(self, instance: InstanceRef) -> str:
        if not self.apply:
            raise FleetProviderError("dump_ui_xml requires apply=True because it writes a temporary guest file")
        remote = f"/sdcard/screenghost_{instance.index if instance.index is not None else 'vm'}.xml"
        first = self.adb(instance, f"shell uiautomator dump {remote}", timeout_s=45)
        if not first.ok:
            raise FleetProviderError(f"LDPlayer UI dump failed: {first.stderr_text()}")
        with tempfile.TemporaryDirectory(prefix="sg-ldplayer-") as tmp:
            local = Path(tmp) / "window.xml"
            pull = self.pull_file(instance, remote, local)
            if not pull.ok or not local.exists():
                raise FleetProviderError(f"LDPlayer UI dump pull failed: {pull.stderr_text()}")
            return local.read_text(encoding="utf-8", errors="replace")
