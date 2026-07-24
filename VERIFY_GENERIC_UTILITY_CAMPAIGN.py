#!/usr/bin/env python3
"""One-command verifier for ScreenGhost's Generic Utility campaign.

The verifier is deliberately layered:

* deterministic emulation proves the original amortization premise;
* real local Chromium proves DPR geometry and bounded dynamic-region handling;
* an oracle worker proves the grounding benchmark protocol, but is classified as
  simulated because the worker receives the hidden answer;
* optional local-model, AndroidWorld, and physical-device runs add measured
  machine-specific receipts without changing the emulated conclusion.

No service or listener is started. Model workers are attached child processes and
must terminate before the verifier exits.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
FOCUSED_TESTS = ("tests/surface_teacher_v1", "tests/generic_utility")
FORBIDDEN_SERVER_MARKERS = (
    "fastapi",
    "uvicorn",
    "flask",
    "socketserver",
    "http.server",
    "websockets.serve",
    ".listen(",
    "create_server(",
)
SCAN_ROOTS = (
    "core/surface_alignment.py",
    "core/surface_curriculum.py",
    "core/surface_evaluator.py",
    "core/surface_graph.py",
    "core/surface_perception.py",
    "core/surface_temporal_teacher.py",
    "experiments/generic_utility",
    "examples/generic_utility_campaign.py",
    "examples/live_playwright_receipt.py",
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
    stdout_path: str | None = None
    stderr_path: str | None = None
    detail: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.status in {"pass", "skipped"} and (not self.required or self.status == "pass")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


class VerificationError(RuntimeError):
    pass


def _run(
    name: str,
    command: Sequence[str],
    *,
    out_dir: Path,
    required: bool = True,
    timeout_s: float = 300.0,
    env: Mapping[str, str] | None = None,
) -> Step:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{name}.stdout.txt"
    stderr_path = logs / f"{name}.stderr.txt"
    started = time.monotonic()
    merged_env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        **dict(env or {}),
    }
    try:
        proc = subprocess.run(
            list(command),
            cwd=ROOT,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        status = "pass" if proc.returncode == 0 else "fail"
        return Step(
            name=name,
            required=required,
            status=status,
            returncode=proc.returncode,
            duration_ms=(time.monotonic() - started) * 1000.0,
            command=list(command),
            stdout_path=str(stdout_path.relative_to(out_dir)),
            stderr_path=str(stderr_path.relative_to(out_dir)),
            detail={
                "stdout_tail": proc.stdout[-1200:],
                "stderr_tail": proc.stderr[-1200:],
            },
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        return Step(
            name=name,
            required=required,
            status="timeout",
            returncode=None,
            duration_ms=(time.monotonic() - started) * 1000.0,
            command=list(command),
            stdout_path=str(stdout_path.relative_to(out_dir)),
            stderr_path=str(stderr_path.relative_to(out_dir)),
            detail={"timeout_s": timeout_s},
        )


def _skip(name: str, detail: str, *, required: bool = False) -> Step:
    return Step(name, required, "skipped", None, 0.0, [], detail={"reason": detail})


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _doctor(out_dir: Path) -> tuple[Step, dict[str, Any]]:
    path = out_dir / "doctor.json"
    step = _run(
        "doctor",
        [sys.executable, "-m", "experiments.generic_utility", "doctor", "--output", str(path)],
        out_dir=out_dir,
        timeout_s=30,
    )
    report = _load(path) if path.exists() else {}
    return step, report


def _authority_scan(out_dir: Path) -> Step:
    started = time.monotonic()
    findings: list[dict[str, str]] = []
    scanned = 0
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
            for marker in FORBIDDEN_SERVER_MARKERS:
                if marker.casefold() in text:
                    findings.append({"path": path.relative_to(ROOT).as_posix(), "marker": marker})
    result = {
        "schema": "screenghost_generic_utility_authority_scan_v1",
        "files_scanned": scanned,
        "forbidden_server_findings": findings,
        "input_authority_note": (
            "The campaign includes a transactional motor abstraction because input behavior is under test. "
            "Surface Teacher adapters remain read-only; no listener or independent command API is introduced."
        ),
        "passed": not findings,
    }
    path = out_dir / "authority_scan.json"
    path.write_bytes(_json_bytes(result))
    return Step(
        "authority_scan",
        True,
        "pass" if not findings else "fail",
        0 if not findings else 1,
        (time.monotonic() - started) * 1000.0,
        [],
        detail=result,
    )


def _run_optional_local_model(args: argparse.Namespace, out_dir: Path, doctor: Mapping[str, Any]) -> Step:
    if not args.local_model:
        return _skip("local_grounder", "no --local-model supplied")
    if not doctor.get("local_grounder_ready") and not args.force_local_model:
        return _skip(
            "local_grounder",
            "doctor reports missing GPU or model dependencies; use --force-local-model to attempt anyway",
            required=args.require_local_model,
        )
    target = out_dir / "grounding-local"
    command = [
        sys.executable,
        "-m",
        "experiments.generic_utility",
        "grounding-local",
        "--out",
        str(target),
        "--model",
        args.local_model,
        "--dtype",
        args.model_dtype,
        "--timeout-ms",
        str(args.model_timeout_ms),
        "--startup-timeout-ms",
        str(args.model_startup_timeout_ms),
    ]
    if args.model_quantization:
        command.extend(["--quantization", args.model_quantization])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return _run(
        "local_grounder",
        command,
        out_dir=out_dir,
        required=args.require_local_model,
        timeout_s=max(300.0, (args.model_startup_timeout_ms + 11 * args.model_timeout_ms) / 1000.0 + 120.0),
    )


def _run_optional_androidworld(args: argparse.Namespace, out_dir: Path, doctor: Mapping[str, Any]) -> Step:
    if not args.androidworld_adb_path:
        return _skip("androidworld_smoke", "no --androidworld-adb-path supplied")
    command = [
        sys.executable,
        "-m",
        "experiments.generic_utility",
        "androidworld-smoke",
        "--out",
        str(out_dir / "androidworld"),
        "--adb-path",
        args.androidworld_adb_path,
        "--console-port",
        str(args.androidworld_console_port),
    ]
    if args.androidworld_setup:
        command.append("--perform-emulator-setup")
    return _run(
        "androidworld_smoke",
        command,
        out_dir=out_dir,
        required=args.require_androidworld,
        timeout_s=600,
    )


def _run_optional_physical(args: argparse.Namespace, out_dir: Path) -> Step:
    if args.physical_device is None:
        return _skip("physical_smoke", "no --physical-device supplied")
    command = [
        sys.executable,
        "-m",
        "experiments.generic_utility",
        "physical-smoke",
        "--out",
        str(out_dir / "physical"),
        "--surface-id",
        args.physical_surface_id,
    ]
    if args.physical_device:
        command.extend(["--device", args.physical_device])
    return _run(
        "physical_smoke",
        command,
        out_dir=out_dir,
        required=args.require_physical,
        timeout_s=180,
    )


def _receipt_path(base: Path, *candidates: str) -> Path | None:
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("log/generic_utility/verification"))
    p.add_argument("--skip-browser", action="store_true")
    p.add_argument("--require-browser", action="store_true")
    p.add_argument("--skip-full-repo", action="store_true")
    p.add_argument("--require-full-repo", action="store_true")
    p.add_argument("--local-model")
    p.add_argument("--model-dtype", default="float16")
    p.add_argument("--model-quantization")
    p.add_argument("--model-timeout-ms", type=float, default=30000)
    p.add_argument("--model-startup-timeout-ms", type=float, default=180000)
    p.add_argument("--require-local-model", action="store_true")
    p.add_argument("--force-local-model", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--androidworld-adb-path")
    p.add_argument("--androidworld-console-port", type=int, default=5554)
    p.add_argument("--androidworld-setup", action="store_true")
    p.add_argument("--require-androidworld", action="store_true")
    p.add_argument("--physical-device", nargs="?", const="")
    p.add_argument("--physical-surface-id", default="physical.current")
    p.add_argument("--require-physical", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out_dir = args.out.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[Step] = []
    started = time.monotonic()

    doctor_step, doctor = _doctor(out_dir)
    steps.append(doctor_step)
    steps.append(_authority_scan(out_dir))

    with tempfile.TemporaryDirectory(prefix="sg-compile-") as pycache:
        steps.append(
            _run(
                "compileall",
                [
                    sys.executable,
                    "-m",
                    "compileall",
                    "-q",
                    "core",
                    "experiments",
                    "examples",
                    "tests",
                    "VERIFY_GENERIC_UTILITY_CAMPAIGN.py",
                ],
                out_dir=out_dir,
                timeout_s=120,
                env={"PYTHONPYCACHEPREFIX": pycache},
            )
        )

    steps.append(
        _run(
            "focused_tests",
            [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS],
            out_dir=out_dir,
            timeout_s=300,
        )
    )

    campaign_dir = out_dir / "campaign"
    steps.append(
        _run(
            "emulated_campaign",
            [sys.executable, "-m", "experiments.generic_utility", "emulate", "--out", str(campaign_dir)],
            out_dir=out_dir,
            timeout_s=180,
        )
    )
    steps.append(
        _run(
            "campaign_integrity",
            [sys.executable, "-m", "experiments.generic_utility", "verify", str(campaign_dir)],
            out_dir=out_dir,
            timeout_s=30,
        )
    )

    oracle_dir = out_dir / "grounding-emulated"
    steps.append(
        _run(
            "grounding_emulated",
            [
                sys.executable,
                "-m",
                "experiments.generic_utility",
                "grounding-emulated",
                "--out",
                str(oracle_dir),
            ],
            out_dir=out_dir,
            timeout_s=120,
        )
    )

    browser_receipt: Path | None = None
    if args.skip_browser:
        steps.append(_skip("browser_smoke", "explicitly skipped", required=args.require_browser))
    elif doctor.get("browser_smoke_ready"):
        browser_dir = out_dir / "browser"
        step = _run(
            "browser_smoke",
            [
                sys.executable,
                "-m",
                "experiments.generic_utility",
                "browser-smoke",
                "--out",
                str(browser_dir),
                "--chromium",
                str(doctor["chromium"]["path"]),
            ],
            out_dir=out_dir,
            required=args.require_browser,
            timeout_s=120,
        )
        steps.append(step)
        browser_receipt = browser_dir / "receipt.json" if step.status == "pass" else None
    else:
        steps.append(
            _skip(
                "browser_smoke",
                "Playwright or Chromium unavailable",
                required=args.require_browser,
            )
        )

    steps.append(_run_optional_local_model(args, out_dir, doctor))
    steps.append(_run_optional_androidworld(args, out_dir, doctor))
    steps.append(_run_optional_physical(args, out_dir))

    full_repo_present = (ROOT / "tests" / "test_drivers.py").exists() and (ROOT / "screenghost.py").exists()
    if args.skip_full_repo:
        steps.append(_skip("full_repository_tests", "explicitly skipped", required=args.require_full_repo))
    elif full_repo_present:
        steps.append(
            _run(
                "full_repository_tests",
                [sys.executable, "-m", "pytest", "-q", "tests"],
                out_dir=out_dir,
                required=True,
                timeout_s=900,
            )
        )
    else:
        steps.append(
            _skip(
                "full_repository_tests",
                "package verifier is not running inside a complete ScreenGhost checkout",
                required=args.require_full_repo,
            )
        )

    grounding_receipt = _receipt_path(
        out_dir,
        "grounding-local/benchmark/benchmark_receipt.json",
        "grounding-emulated/benchmark/benchmark_receipt.json",
    )
    androidworld_receipt = _receipt_path(out_dir, "androidworld/receipt.json")
    physical_receipt = _receipt_path(out_dir, "physical/receipt.json")
    conclusion_path = out_dir / "conclusion.json"
    conclude = [
        sys.executable,
        "-m",
        "experiments.generic_utility",
        "conclude",
        "--campaign",
        str(campaign_dir),
        "--out",
        str(conclusion_path),
    ]
    if browser_receipt:
        conclude.extend(["--browser", str(browser_receipt)])
    if grounding_receipt:
        conclude.extend(["--grounding", str(grounding_receipt)])
    if androidworld_receipt:
        conclude.extend(["--androidworld", str(androidworld_receipt)])
    if physical_receipt:
        conclude.extend(["--physical", str(physical_receipt)])
    steps.append(_run("conclusion", conclude, out_dir=out_dir, timeout_s=30))

    conclusion = _load(conclusion_path) if conclusion_path.exists() else {}
    required_steps_passed = all(step.passed for step in steps if step.required)
    success = required_steps_passed and bool(conclusion.get("premise_conclusion_ready"))
    validation = {
        "schema": "screenghost_generic_utility_validation_v1",
        "status": "pass" if success else "fail",
        "duration_ms": (time.monotonic() - started) * 1000.0,
        "repository_root": str(ROOT),
        "full_screen_ghost_checkout": full_repo_present,
        "original_premise": (
            "after expensive first teaching, a phone should become a generic utility whose known operation is "
            "cheaper, patient, teacher-blind, and transferable"
        ),
        "evidence_ledger": {
            "emulated_system_behavior": "executed",
            "browser_geometry_and_dynamic_regions": (
                "measured" if browser_receipt else "not_run"
            ),
            "oracle_grounding_protocol": "executed_simulated_answer",
            "local_student_model": (
                "measured" if _receipt_path(out_dir, "grounding-local/benchmark/benchmark_receipt.json") else "not_run"
            ),
            "androidworld_transport": "measured" if androidworld_receipt else "not_run",
            "physical_adb_transport": "measured" if physical_receipt else "not_run",
        },
        "doctor": doctor,
        "steps": [step.to_dict() for step in steps],
        "conclusion": conclusion,
    }
    (out_dir / "VALIDATION.json").write_bytes(_json_bytes(validation))
    print(json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
