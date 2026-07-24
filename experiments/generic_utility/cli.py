"""Command-line front door for the complete Generic Utility campaign."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.generic_utility.androidworld_smoke import run_androidworld_read_smoke
from experiments.generic_utility.campaign import EmulatedCampaignConfig, run_emulated_campaign
from experiments.generic_utility.conclusion import assemble_conclusion
from experiments.generic_utility.doctor import capability_report
from experiments.generic_utility.grounding_benchmark import (
    build_phoneworld_grounding_suite,
    run_grounding_benchmark,
)
from experiments.generic_utility.metrics import verify_campaign_bundle
from experiments.generic_utility.model_runtime import AttachedJsonModelProvider
from experiments.generic_utility.physical_smoke import run_physical_read_smoke


ROOT = Path(__file__).resolve().parents[2]
FAKE_WORKER = ROOT / "experiments" / "generic_utility" / "model_workers" / "fake_grounder_worker.py"
HF_WORKER = ROOT / "experiments" / "generic_utility" / "model_workers" / "hf_gui_grounder_worker.py"
BROWSER_SCRIPT = ROOT / "examples" / "live_playwright_receipt.py"


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def command_doctor(args) -> int:
    report = capability_report()
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(report)
    return 0 if report["emulated_campaign_ready"] else 2


def command_emulate(args) -> int:
    out = run_emulated_campaign(
        args.out,
        seed=args.seed,
        minimum_visual_confidence=args.minimum_confidence,
        minimum_visual_margin=args.minimum_margin,
    )
    receipt = json.loads((out / "campaign_receipt.json").read_text())
    _print(receipt)
    return 0 if receipt.get("all_gates_passed") else 1


def command_verify(args) -> int:
    result = verify_campaign_bundle(args.path)
    _print(result)
    return 0 if result.get("ok") else 1


def command_browser(args) -> int:
    chromium = args.chromium or os.environ.get("SG_CHROMIUM") or shutil.which("chromium")
    if not chromium:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                chromium = playwright.chromium.executable_path
        except Exception:
            chromium = None
    if not chromium or not Path(chromium).exists():
        raise SystemExit("Chromium not found; pass --chromium or run playwright install chromium")
    command = [sys.executable, str(BROWSER_SCRIPT), "--out", str(args.out), "--chromium", str(chromium)]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def command_grounding_emulated(args) -> int:
    suite = build_phoneworld_grounding_suite(Path(args.out) / "suite")
    with AttachedJsonModelProvider(
        "emulated_oracle_grounder",
        [sys.executable, str(FAKE_WORKER), "--mode", "oracle"],
        startup_timeout_ms=5000,
    ) as provider:
        out = run_grounding_benchmark(
            provider,
            suite,
            Path(args.out) / "benchmark",
            timeout_ms=args.timeout_ms,
            emulated_oracle_payload=True,
        )
    receipt = json.loads((out / "benchmark_receipt.json").read_text())
    _print(receipt)
    return 0 if receipt.get("hit_rate") == 1.0 else 1


def command_grounding_local(args) -> int:
    suite = build_phoneworld_grounding_suite(Path(args.out) / "suite")
    command = [
        sys.executable,
        str(HF_WORKER),
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    with AttachedJsonModelProvider(
        args.model,
        command,
        startup_timeout_ms=args.startup_timeout_ms,
        quantization=args.quantization,
    ) as provider:
        out = run_grounding_benchmark(
            provider,
            suite,
            Path(args.out) / "benchmark",
            timeout_ms=args.timeout_ms,
        )
    receipt = json.loads((out / "benchmark_receipt.json").read_text())
    _print(receipt)
    return 0 if receipt.get("completed") == receipt.get("case_count") else 1


def command_androidworld(args) -> int:
    out = run_androidworld_read_smoke(
        args.out,
        adb_path=args.adb_path,
        console_port=args.console_port,
        perform_emulator_setup=args.perform_emulator_setup,
    )
    _print(json.loads((out / "receipt.json").read_text()))
    return 0


def command_physical(args) -> int:
    out = run_physical_read_smoke(
        args.out,
        device=args.device,
        surface_id=args.surface_id,
        burst_count=args.burst_count,
    )
    _print(json.loads((out / "receipt.json").read_text()))
    return 0


def command_conclude(args) -> int:
    out = assemble_conclusion(
        args.out,
        campaign_dir=args.campaign,
        browser_receipt=args.browser,
        grounding_receipt=args.grounding,
        androidworld_receipt=args.androidworld,
        physical_receipt=args.physical,
    )
    result = json.loads(out.read_text())
    _print(result)
    return 0 if result.get("premise_conclusion_ready") else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m experiments.generic_utility")
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="side-effect-free capability report")
    p.add_argument("--output", type=Path)
    p.set_defaults(func=command_doctor)

    p = sub.add_parser("emulate", help="run the complete deterministic premise campaign")
    p.add_argument("--out", type=Path, default=Path("log/generic_utility/emulated"))
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--minimum-confidence", type=float, default=0.90)
    p.add_argument("--minimum-margin", type=float, default=0.05)
    p.set_defaults(func=command_emulate)

    p = sub.add_parser("verify", help="verify a campaign bundle manifest")
    p.add_argument("path", type=Path)
    p.set_defaults(func=command_verify)

    p = sub.add_parser("browser-smoke", help="real local Chromium DPR and dynamic-region receipt")
    p.add_argument("--out", type=Path, default=Path("log/generic_utility/browser"))
    p.add_argument("--chromium")
    p.set_defaults(func=command_browser)

    p = sub.add_parser("grounding-emulated", help="validate the hidden-teacher grounding benchmark")
    p.add_argument("--out", type=Path, default=Path("log/generic_utility/grounding-emulated"))
    p.add_argument("--timeout-ms", type=float, default=5000)
    p.set_defaults(func=command_grounding_emulated)

    p = sub.add_parser("grounding-local", help="measure a run-scoped Hugging Face GUI grounder")
    p.add_argument("--out", type=Path, default=Path("log/generic_utility/grounding-local"))
    p.add_argument("--model", default="osunlp/UGround-V1-2B")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quantization")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--startup-timeout-ms", type=float, default=180000)
    p.add_argument("--timeout-ms", type=float, default=30000)
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(func=command_grounding_local)

    p = sub.add_parser("androidworld-smoke", help="read-only live AndroidWorld state capture")
    p.add_argument("--out", type=Path, default=Path("log/generic_utility/androidworld"))
    p.add_argument("--adb-path", required=True)
    p.add_argument("--console-port", type=int, default=5554)
    p.add_argument("--perform-emulator-setup", action="store_true")
    p.set_defaults(func=command_androidworld)

    p = sub.add_parser("physical-smoke", help="read-only attached ADB lesson capture")
    p.add_argument("--out", type=Path, default=Path("log/generic_utility/physical"))
    p.add_argument("--device")
    p.add_argument("--surface-id", default="physical.current")
    p.add_argument("--burst-count", type=int, default=3)
    p.set_defaults(func=command_physical)

    p = sub.add_parser("conclude", help="assemble the explicit conclusion receipt")
    p.add_argument("--out", type=Path, default=Path("log/generic_utility/conclusion.json"))
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--browser", type=Path)
    p.add_argument("--grounding", type=Path)
    p.add_argument("--androidworld", type=Path)
    p.add_argument("--physical", type=Path)
    p.set_defaults(func=command_conclude)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
