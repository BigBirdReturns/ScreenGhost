"""BlueStacks capability reporting.

BlueStacks exposes mature multi-instance, sync, macro import/export, and Eco Mode
through its supported UI.  This module deliberately does not automate undocumented
lifecycle executables.  Exported JSON macros are parsed by ``macro.py`` and can be
used as coordinate demonstrations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from experiments.emulator_fleet.providers.base import ProviderCapability
from experiments.emulator_fleet.schema import EmulatorVendor


class BlueStacksCapabilityProvider:
    vendor = EmulatorVendor.BLUESTACKS
    DEFAULT_PATHS = (
        Path(r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"),
        Path(r"C:\Program Files\BlueStacks_nxt\HD-MultiInstanceManager.exe"),
    )

    def __init__(self, executable: Optional[str | Path] = None) -> None:
        self.executable = Path(executable) if executable else next(
            (path for path in self.DEFAULT_PATHS if path.exists()),
            self.DEFAULT_PATHS[0],
        )

    def capability(self) -> ProviderCapability:
        installed = any(path.exists() for path in self.DEFAULT_PATHS) or self.executable.exists()
        return ProviderCapability(
            vendor=self.vendor,
            executable=str(self.executable),
            installed=installed,
            lifecycle_cli=False,
            clone_supported=True,
            configuration_supported=True,
            adb_wrapper_supported=False,
            macro_import_supported=True,
            macro_execution_supported=True,
            missing=(() if installed else ("BlueStacks 5 installation",)),
            notes=(
                "multi-instance lifecycle and Eco Mode remain operator-controlled through supported UI",
                "exported macro JSON can be distilled without automating the manager window",
                "no undocumented BlueStacks executable is invoked",
            ),
        )
