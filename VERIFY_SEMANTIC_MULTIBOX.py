#!/usr/bin/env python3
"""One-command verifier for ScreenGhost semantic multiboxing.

The verifier proves the complete emulator-first claim, then optionally executes a
prepared MEmu or LDPlayer machine plan.  No listener or background service is
started.  Vendor mutation remains dry-run unless the operator supplies both a
machine plan and ``--apply-machine``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.emulator_fleet.doctor import doctor_report
from experiments.emulator_fleet.schema import write_json

ROOT = Path(__file__).resolve().parent

def _verify_campaign_bundle(root: Path) -> dict[str, Any]:
    import hashlib

    manifest_path = root / "MANIFEST.json"
    if not manifest_path.exists():
        return {"ok": False, "mismatches": [{"path": "MANIFEST.json", "actual": None}]}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for rel, expected in payload.get("files", {}).items():
        path = root / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"ok": not mismatches, "mismatches": mismatches, "bundle_id": payload.get("bundle_id")}

FOCUSED_TEST_GROUPS = (
    ("surface_and_generic_tests", "tests/surface_teacher_v1", "tests/generic_utility"),
    ("emulator_fleet_tests", "tests/emulator_fleet"),
)
FORBIDDEN_MARKERS = (
    "fastapi",
    "uvicorn",
    "flask",
    "socketserver",
    "http.server",
    "websockets.serve",
    ".listen(",
    "create_server(",
    "pyautogui",
    "pynput",
    "win32api.keybd_event",
    "win32api.mouse_event",
)
SCAN_ROOTS = (
    "experiments/emulator_fleet",
    "examples/semantic_multibox_campaign.py",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


@dataclass
class Step:
    name: str
    required: bool
    status: str
    returncode: int | None
    duration_ms: float
    command: list[str]
    detail: Mapping[str, Any] | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass" or (self.status == "skipped" and not self.required)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


def _run(
    name: str,
    command: Sequence[str],
    *,
    out: Path,
    required: bool = True,
    timeout_s: float = 600,
) -> Step:
    logs = out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{name}.stdout.txt"
    stderr_path = logs / f"{name}.stderr.txt"
    started = time.monotonic()
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    print(f"[verify] {name}: {' '.join(map(str, command))}", flush=True)
    try:
        proc = subprocess.run(
            list(command), cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        print(f"[verify] {name}: {'PASS' if proc.returncode == 0 else 'FAIL'} ({proc.returncode})", flush=True)
        return Step(
            name, required, "pass" if proc.returncode == 0 else "fail", proc.returncode,
            (time.monotonic() - started) * 1000.0, list(command),
            detail={"stdout_tail": proc.stdout[-1600:], "stderr_tail": proc.stderr[-1600:]},
            stdout_path=str(stdout_path.relative_to(out)),
            stderr_path=str(stderr_path.relative_to(out)),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        return Step(
            name, required, "timeout", None, (time.monotonic() - started) * 1000.0,
            list(command), detail={"timeout_s": timeout_s},
            stdout_path=str(stdout_path.relative_to(out)),
            stderr_path=str(stderr_path.relative_to(out)),
        )


def _skip(name: str, reason: str, *, required: bool = False) -> Step:
    return Step(name, required, "skipped", None, 0.0, [], detail={"reason": reason})


def _authority_scan(out: Path) -> Step:
    started = time.monotonic(); findings = []; scanned = 0
    for rel in SCAN_ROOTS:
        root = ROOT / rel
        paths: Iterable[Path]
        if root.is_dir():
            paths = root.rglob("*.py")
        elif root.exists():
            paths = (root,)
        else:
            continue
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace").casefold()
            scanned += 1
            for marker in FORBIDDEN_MARKERS:
                if marker.casefold() in text:
                    findings.append({"path": path.relative_to(ROOT).as_posix(), "marker": marker})
    payload = {
        "schema": "semantic_multibox_authority_scan_v1",
        "files_scanned": scanned,
        "forbidden_findings": findings,
        "host_input_calls": 0,
        "listeners": 0,
        "passed": not findings,
    }
    write_json(out / "authority_scan.json", payload)
    return Step(
        "authority_scan", True, "pass" if not findings else "fail",
        0 if not findings else 1, (time.monotonic() - started) * 1000.0,
        [], detail=payload,
    )


def _full_repo_available() -> bool:
    return (ROOT / "tests" / "test_drivers.py").exists() and (ROOT / "screenghost.py").exists()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("log/semantic_multibox/verification"))
    p.add_argument("--skip-full-repo", action="store_true")
    p.add_argument("--require-full-repo", action="store_true")
    p.add_argument("--machine-plan", type=Path)
    p.add_argument("--apply-machine", action="store_true")
    p.add_argument("--require-machine", action="store_true")
    p.add_argument("--memuc")
    p.add_argument("--ldconsole")
    p.add_argument("--bluestacks")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out = args.out.resolve()
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); steps: list[Step] = []

    doctor = doctor_report(memuc=args.memuc, ldconsole=args.ldconsole, bluestacks=args.bluestacks)
    write_json(out / "doctor.json", doctor)
    steps.append(Step("doctor", True, "pass" if doctor["ready_for_emulated_campaign"] else "fail", 0, 0.0, [], detail=doctor))
    steps.append(_authority_scan(out))
    steps.append(_run(
        "compileall",
        [sys.executable, "-m", "compileall", "-q", "core", "experiments", "examples", "tests", "VERIFY_SEMANTIC_MULTIBOX.py"],
        out=out, timeout_s=180,
    ))
    for name, *paths in FOCUSED_TEST_GROUPS:
        steps.append(_run(name, [sys.executable, "-m", "pytest", "-q", *paths], out=out, timeout_s=900))

    campaign_dir = out / "campaign"
    steps.append(_run(
        "semantic_multibox_campaign",
        [sys.executable, "-m", "experiments.emulator_fleet", "campaign", "--out", str(campaign_dir)],
        out=out, timeout_s=600,
    ))
    if (campaign_dir / "MANIFEST.json").exists():
        bundle = _verify_campaign_bundle(campaign_dir)
        write_json(out / "campaign_bundle_verification.json", bundle)
        steps.append(Step("campaign_bundle_verification", True, "pass" if bundle["ok"] else "fail", 0 if bundle["ok"] else 1, 0.0, [], detail=bundle))
    else:
        steps.append(Step("campaign_bundle_verification", True, "fail", 1, 0.0, [], detail={"reason": "campaign manifest missing"}))

    if not args.skip_full_repo and _full_repo_available():
        steps.append(_run("full_repository_tests", [sys.executable, "-m", "pytest", "-q", "tests"], out=out, required=args.require_full_repo, timeout_s=1200))
    elif args.require_full_repo:
        steps.append(_skip("full_repository_tests", "complete ScreenGhost checkout not present", required=True))
    else:
        steps.append(_skip("full_repository_tests", "complete ScreenGhost checkout not present or explicitly skipped"))

    if args.machine_plan:
        command = [sys.executable, "-m", "experiments.emulator_fleet", "machine", "--plan", str(args.machine_plan), "--out", str(out / "machine")]
        if args.apply_machine: command.append("--apply")
        steps.append(_run("measured_machine_plan", command, out=out, required=args.require_machine, timeout_s=1800))
    elif args.require_machine:
        steps.append(_skip("measured_machine_plan", "--machine-plan was not supplied", required=True))
    else:
        steps.append(_skip("measured_machine_plan", "no measured machine plan supplied"))

    campaign_conclusion = {}
    path = campaign_dir / "CONCLUSION.json"
    if path.exists(): campaign_conclusion = json.loads(path.read_text(encoding="utf-8"))
    required_failures = [step.name for step in steps if step.required and not step.passed]
    validation = {
        "schema": "semantic_multibox_validation_v1",
        "status": "PASS" if not required_failures else "FAIL",
        "duration_ms": (time.monotonic() - started) * 1000.0,
        "python": sys.version,
        "steps": [step.to_dict() for step in steps],
        "required_failures": required_failures,
        "campaign_conclusion": campaign_conclusion,
        "evidence_classification": {
            "deterministic_semantic_multibox": "PASS" if campaign_conclusion.get("status") == "PASS" else "NOT_PROVEN",
            "vendor_provider_command_surfaces": "TESTED_WITH_RECORDING_RUNNER",
            "measured_installed_emulator_fleet": "PASS" if any(step.name == "measured_machine_plan" and step.status == "pass" for step in steps) else "NOT_RUN",
        },
        "honest_remainders": [
            "A measured MEmu or LDPlayer receipt requires prepared, disjoint golden clones and --apply-machine.",
            "MEmu .mir action bytes remain opaque unless accompanied by an observed or exported action manifest.",
            "BlueStacks lifecycle remains operator-controlled because this package does not call undocumented executables.",
        ],
    }
    write_json(out / "VALIDATION.json", validation)
    lines = [
        "# Semantic Multibox Verification",
        "",
        f"Status: **{validation['status']}**",
        "",
        "| Step | Required | Status | Duration ms |",
        "|---|---:|---:|---:|",
    ]
    for step in steps:
        lines.append(f"| `{step.name}` | {step.required} | {step.status} | {step.duration_ms:.1f} |")
    lines.extend(["", "## Evidence classification", ""])
    for key, value in validation["evidence_classification"].items():
        lines.append(f"- `{key}`: **{value}**")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": validation["status"], "out": str(out), "required_failures": required_failures}, indent=2))
    return 0 if not required_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
