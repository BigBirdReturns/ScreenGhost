"""Provider contracts for local Android emulator fleets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from experiments.emulator_fleet.command import CommandResult, CommandRunner, planned_result
from experiments.emulator_fleet.schema import EmulatorVendor, InstanceProfile, InstanceRef


class FleetProviderError(RuntimeError):
    pass


class MutationRefused(FleetProviderError):
    pass


@dataclass(frozen=True)
class ProviderCapability:
    vendor: EmulatorVendor
    executable: str
    installed: bool
    lifecycle_cli: bool
    clone_supported: bool
    configuration_supported: bool
    adb_wrapper_supported: bool
    macro_import_supported: bool
    macro_execution_supported: bool
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor.value,
            "executable": self.executable,
            "installed": self.installed,
            "lifecycle_cli": self.lifecycle_cli,
            "clone_supported": self.clone_supported,
            "configuration_supported": self.configuration_supported,
            "adb_wrapper_supported": self.adb_wrapper_supported,
            "macro_import_supported": self.macro_import_supported,
            "macro_execution_supported": self.macro_execution_supported,
            "missing": list(self.missing),
            "notes": list(self.notes),
        }


@runtime_checkable
class FleetProvider(Protocol):
    vendor: EmulatorVendor

    def capability(self) -> ProviderCapability: ...
    def list_instances(self, *, running_only: bool = False) -> tuple[InstanceRef, ...]: ...
    def start(self, instance: InstanceRef) -> CommandResult: ...
    def stop(self, instance: InstanceRef) -> CommandResult: ...
    def clone(self, source: InstanceRef, *, new_name: str) -> tuple[CommandResult, ...]: ...
    def configure(self, instance: InstanceRef, profile: InstanceProfile) -> tuple[CommandResult, ...]: ...
    def adb(self, instance: InstanceRef, command: str, *, timeout_s: float = 30.0) -> CommandResult: ...


class VendorProviderBase:
    vendor: EmulatorVendor

    def __init__(
        self,
        executable: str | Path,
        *,
        runner: CommandRunner,
        apply: bool = False,
        destructive_token: Optional[str] = None,
    ) -> None:
        self.executable = Path(executable)
        self.runner = runner
        self.apply = bool(apply)
        self.destructive_token = destructive_token

    def _run(
        self,
        args: Sequence[str],
        *,
        mutation: bool = False,
        destructive: bool = False,
        timeout_s: float = 30.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CommandResult:
        argv = (str(self.executable), *tuple(str(v) for v in args))
        if mutation and not self.apply:
            return planned_result(argv, metadata={"mutation": True, **dict(metadata or {})})
        if destructive and self.destructive_token != "SCREEN_GHOST_FLEET_MUTATION":
            raise MutationRefused(
                "destructive fleet mutation requires destructive_token='SCREEN_GHOST_FLEET_MUTATION'"
            )
        return self.runner.run(argv, timeout_s=timeout_s, metadata=metadata)

    @staticmethod
    def _require_name(name: str) -> str:
        value = " ".join(str(name).split())
        if not value:
            raise ValueError("instance name is required")
        if any(ch in value for ch in "\r\n\0"):
            raise ValueError("instance name contains control characters")
        return value
