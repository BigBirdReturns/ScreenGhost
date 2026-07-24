"""Side-effect-free capability probes for the experiment campaign."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {"available": path is not None, "path": path}


def capability_report() -> dict[str, Any]:
    chromium_candidates = [
        os.environ.get("SG_CHROMIUM"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        str(Path.home() / ".cache" / "ms-playwright"),
    ]
    chromium = next((value for value in chromium_candidates if value and Path(value).exists()), None)
    nvidia = {"available": False, "name": None, "memory_total_mb": None, "error": None}
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            row = proc.stdout.strip().splitlines()[0].split(",") if proc.stdout.strip() else []
            if len(row) >= 3:
                nvidia.update(
                    available=True,
                    name=row[0].strip(),
                    memory_total_mb=float(row[1].strip()),
                    driver_version=row[2].strip(),
                )
            else:
                nvidia["error"] = proc.stderr.strip() or "nvidia-smi returned no GPU row"
        except Exception as exc:
            nvidia["error"] = str(exc)
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "PIL",
            "numpy",
            "pytest",
            "psutil",
            "playwright",
            "torch",
            "transformers",
            "qwen_vl_utils",
            "android_world",
        )
    }
    report = {
        "schema": "screenghost_generic_utility_doctor_v1",
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "supported": sys.version_info >= (3, 11),
        },
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "commands": {
            "git": _command("git"),
            "adb": _command("adb"),
            "emulator": _command("emulator"),
            "ffmpeg": _command("ffmpeg"),
            "nvidia_smi": _command("nvidia-smi"),
        },
        "modules": modules,
        "versions": {
            "pillow": _version("Pillow"),
            "numpy": _version("numpy"),
            "pytest": _version("pytest"),
            "psutil": _version("psutil"),
            "playwright": _version("playwright"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
        "chromium": {"available": chromium is not None, "path": chromium},
        "gpu": nvidia,
        "surface_teacher_pr13_available": all(
            importlib.util.find_spec(name) is not None
            for name in ("core.surface_teacher", "core.teacher_android", "core.teacher_web")
        ),
    }
    report["emulated_campaign_ready"] = all(
        modules[name] for name in ("PIL", "numpy", "pytest", "psutil")
    ) and report["python"]["supported"]
    report["browser_smoke_ready"] = modules["playwright"] and chromium is not None
    report["local_grounder_ready"] = modules["torch"] and modules["transformers"] and nvidia["available"]
    report["androidworld_ready"] = modules["android_world"] and report["commands"]["adb"]["available"]
    report["physical_adb_ready"] = report["commands"]["adb"]["available"] and report["surface_teacher_pr13_available"]
    return report
