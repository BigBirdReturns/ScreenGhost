"""Small command line for profile validation, theme resolution, and Windows preflight."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from experiments.dti.profile import DTIProfile
from experiments.dti.schema import sha256_bytes
from experiments.dti.theme_kernel import ThemeCatalog, seed_theme_cards
from experiments.dti.windows_driver import WindowTarget, WindowsGameDriver


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.dti")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-profile")
    validate.add_argument("profile", type=Path)

    resolve = sub.add_parser("resolve-theme")
    resolve.add_argument("theme")

    doctor = sub.add_parser("doctor")
    doctor.add_argument("profile", type=Path)

    capture = sub.add_parser("capture")
    capture.add_argument("profile", type=Path)
    capture.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-profile":
        profile = DTIProfile.load(args.profile)
        print(
            json.dumps(
                {
                    "profile_id": profile.profile_id,
                    "app_version": profile.app_version,
                    "client_size": list(profile.client_size),
                    "regions": sorted(profile.observation_regions),
                    "anchors": len(profile.anchors),
                    "wardrobe_items": len(profile.wardrobe.items),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "resolve-theme":
        result = ThemeCatalog(seed_theme_cards()).resolve(args.theme)
        print(
            json.dumps(
                {
                    "raw_text": result.raw_text,
                    "normalized_text": result.normalized_text,
                    "resolved": result.resolved,
                    "theme": result.card.canonical_name if result.card else None,
                    "confidence": result.confidence,
                    "margin": result.margin,
                    "alternatives": result.alternatives,
                    "reason": result.reason,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.resolved else 2

    if args.command in {"doctor", "capture"}:
        profile = DTIProfile.load(args.profile)
        driver = WindowsGameDriver(
            WindowTarget(
                title_pattern=profile.window_title_pattern,
                expected_client_size=profile.client_size,
            )
        )
        result = driver.doctor()
        if args.command == "doctor":
            print(json.dumps(asdict(result) | {"ready": result.ready}, indent=2, sort_keys=True))
            return 0 if result.ready else 3
        if not result.ready:
            print(json.dumps(asdict(result) | {"ready": False}, indent=2, sort_keys=True))
            return 3
        try:
            image = driver.capture()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            image.save(args.output, format="PNG")
            payload = args.output.read_bytes()
            print(
                json.dumps(
                    {
                        "profile_id": profile.profile_id,
                        "output": str(args.output),
                        "width": image.width,
                        "height": image.height,
                        "sha256": sha256_bytes(payload),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        finally:
            driver.close()

    raise AssertionError("argparse returned an unknown command")
