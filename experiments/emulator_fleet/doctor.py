"""Machine capability report for semantic multibox experiments."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from experiments.emulator_fleet.command import SubprocessCommandRunner
from experiments.emulator_fleet.providers.bluestacks import BlueStacksCapabilityProvider
from experiments.emulator_fleet.providers.ldplayer import LDPlayerFleetProvider
from experiments.emulator_fleet.providers.memu import MEmuFleetProvider


def _find_path(explicit: Optional[str], discovered: Optional[Path], command: str) -> Optional[Path]:
    if explicit:
        return Path(explicit)
    if discovered is not None:
        return discovered
    found = shutil.which(command)
    return Path(found) if found else None


def doctor_report(
    *,
    memuc: Optional[str] = None,
    ldconsole: Optional[str] = None,
    bluestacks: Optional[str] = None,
) -> dict[str, Any]:
    runner = SubprocessCommandRunner()
    memu_path = _find_path(memuc, MEmuFleetProvider.discover(), "memuc")
    ld_path = _find_path(ldconsole, LDPlayerFleetProvider.discover(), "ldconsole")
    blue = BlueStacksCapabilityProvider(bluestacks)

    memu_cap = MEmuFleetProvider(
        memu_path or Path("memuc.exe"), runner=runner
    ).capability()
    ld_cap = LDPlayerFleetProvider(
        ld_path or Path("ldconsole.exe"), runner=runner
    ).capability()
    blue_cap = blue.capability()

    pr13_modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("core.surface_teacher", "core.teacher_android")
    }
    report = {
        "schema": "semantic_multibox_doctor_v1",
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "providers": {
            "memu": memu_cap.to_dict(),
            "ldplayer": ld_cap.to_dict(),
            "bluestacks": blue_cap.to_dict(),
        },
        "surface_teacher_pr13": {
            "ready": all(pr13_modules.values()),
            "modules": pr13_modules,
        },
        "generic_campaign_ready": importlib.util.find_spec(
            "experiments.generic_utility"
        )
        is not None,
        "adb_on_path": shutil.which("adb"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "pytest": importlib.util.find_spec("pytest") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "mutation_default": "dry_run",
        "destructive_token_required": "SCREEN_GHOST_FLEET_MUTATION",
        "environment": {
            "SG_MEMUC": os.getenv("SG_MEMUC"),
            "SG_LDCONSOLE": os.getenv("SG_LDCONSOLE"),
            "SG_BLUESTACKS": os.getenv("SG_BLUESTACKS"),
        },
    }
    report["ready_for_emulated_campaign"] = bool(
        report["generic_campaign_ready"] and report["pytest"] and report["pillow"] and report["numpy"]
    )
    report["installed_vendor_count"] = sum(
        bool(row["installed"]) for row in report["providers"].values()
    )
    return report
