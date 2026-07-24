"""Command-line entrypoint for ScreenGhost semantic multibox experiments."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments.emulator_fleet.campaign import (
    run_semantic_multibox_campaign,
    verify_bundle,
)
from experiments.emulator_fleet.command import SubprocessCommandRunner
from experiments.emulator_fleet.doctor import doctor_report
from experiments.emulator_fleet.machine import MachinePlan, run_machine_plan
from experiments.emulator_fleet.macro import catalog_memu_macros, load_macro
from experiments.emulator_fleet.providers.ldplayer import LDPlayerFleetProvider
from experiments.emulator_fleet.providers.memu import MEmuFleetProvider
from experiments.emulator_fleet.schema import EmulatorVendor, write_json


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _provider(vendor: str, executable: str, *, apply: bool):
    runner = SubprocessCommandRunner()
    kind = EmulatorVendor(vendor)
    if kind is EmulatorVendor.MEMU:
        return MEmuFleetProvider(executable, runner=runner, apply=apply)
    if kind is EmulatorVendor.LDPLAYER:
        return LDPlayerFleetProvider(executable, runner=runner, apply=apply)
    raise SystemExit("provider commands support memu or ldplayer")


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(
        memuc=args.memuc or os.getenv("SG_MEMUC"),
        ldconsole=args.ldconsole or os.getenv("SG_LDCONSOLE"),
        bluestacks=args.bluestacks or os.getenv("SG_BLUESTACKS"),
    )
    if args.output:
        write_json(args.output, report)
    _json(report)
    return 0 if report["ready_for_emulated_campaign"] else 2


def _cmd_campaign(args: argparse.Namespace) -> int:
    out = run_semantic_multibox_campaign(args.out)
    print(out)
    return 0


def _cmd_verify_bundle(args: argparse.Namespace) -> int:
    result = verify_bundle(args.bundle)
    _json(result)
    return 0 if result["ok"] else 1


def _cmd_parse_macro(args: argparse.Namespace) -> int:
    resolution = tuple(args.default_resolution) if args.default_resolution else None
    macro = load_macro(
        args.path,
        format_hint=args.format,
        default_resolution=resolution,
    )
    payload = macro.to_dict()
    if args.output:
        write_json(args.output, payload)
    _json(payload)
    return 0 if not macro.unsupported_actions else 3


def _cmd_memu_catalog(args: argparse.Namespace) -> int:
    payload = {
        "schema": "memu_macro_catalog_v1",
        "scripts_dir": str(Path(args.scripts_dir).resolve()),
        "macros": list(catalog_memu_macros(args.scripts_dir)),
    }
    if args.output:
        write_json(args.output, payload)
    _json(payload)
    return 0


def _cmd_inventory(args: argparse.Namespace) -> int:
    provider = _provider(args.vendor, args.executable, apply=False)
    capability = provider.capability().to_dict()
    try:
        rows = [row.to_dict() for row in provider.list_instances(running_only=args.running_only)]
        error = None
    except Exception as exc:
        rows = []
        error = str(exc)
    payload = {
        "schema": "emulator_fleet_inventory_v1",
        "capability": capability,
        "instances": rows,
        "error": error,
    }
    if args.output:
        write_json(args.output, payload)
    _json(payload)
    return 0 if error is None else 1


def _cmd_machine(args: argparse.Namespace) -> int:
    plan = MachinePlan.load(args.plan)
    out = run_machine_plan(plan, out_dir=args.out, apply=args.apply)
    print(out)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="report installed vendor and test capabilities")
    d.add_argument("--memuc")
    d.add_argument("--ldconsole")
    d.add_argument("--bluestacks")
    d.add_argument("--output", type=Path)
    d.set_defaults(func=_cmd_doctor)

    c = sub.add_parser("campaign", help="run the full deterministic multibox campaign")
    c.add_argument("--out", type=Path, default=Path("log/semantic_multibox/campaign"))
    c.set_defaults(func=_cmd_campaign)

    v = sub.add_parser("verify-bundle", help="verify a campaign bundle manifest")
    v.add_argument("bundle", type=Path)
    v.set_defaults(func=_cmd_verify_bundle)

    m = sub.add_parser("parse-macro", help="parse an exported coordinate macro")
    m.add_argument("path", type=Path)
    m.add_argument("--format")
    m.add_argument("--default-resolution", nargs=2, type=int)
    m.add_argument("--output", type=Path)
    m.set_defaults(func=_cmd_parse_macro)

    mc = sub.add_parser("memu-catalog", help="catalog opaque MEmu .mir artifacts")
    mc.add_argument("scripts_dir", type=Path)
    mc.add_argument("--output", type=Path)
    mc.set_defaults(func=_cmd_memu_catalog)

    i = sub.add_parser("inventory", help="list instances through a vendor CLI")
    i.add_argument("--vendor", choices=("memu", "ldplayer"), required=True)
    i.add_argument("--executable", required=True)
    i.add_argument("--running-only", action="store_true")
    i.add_argument("--output", type=Path)
    i.set_defaults(func=_cmd_inventory)

    r = sub.add_parser("machine", help="run a measured fleet plan over prepared clones")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--out", type=Path, default=Path("log/semantic_multibox/machine"))
    r.add_argument("--apply", action="store_true", help="allow instance-scoped Android input")
    r.set_defaults(func=_cmd_machine)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))
