#!/usr/bin/env python3
"""Compile one ScreenGhost Surface Teacher lesson.

The file modes are offline and deterministic. The live Android mode performs a
bounded read-only capture through ScreenGhost's existing AndroidAdbDriver. No mode
issues input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from core.pixel_evidence import png_dimensions
from core.surface_teacher import (
    LessonArtifact,
    LessonPolicy,
    SourceKind,
    SurfaceAtlas,
    compile_lesson,
    stage_lesson,
)
from core.teacher_sources import (
    capture_android_lesson,
    parse_dom_snapshot,
    parse_uiautomator_xml,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _policy(args: argparse.Namespace) -> LessonPolicy:
    return LessonPolicy(
        retain_values=bool(args.retain_values),
        write_element_crops=not bool(args.no_crops),
    )


def _compile_android_files(args: argparse.Namespace) -> LessonArtifact:
    png_bytes = args.png.read_bytes()
    viewport = png_dimensions(png_bytes)
    if viewport is None:
        raise ValueError(f"not a readable PNG: {args.png}")
    xml_bytes = args.xml.read_bytes()
    xml_text = xml_bytes.decode("utf-8")
    nodes = parse_uiautomator_xml(xml_text, viewport=viewport)
    return compile_lesson(
        png_bytes,
        surface_id=args.surface_id,
        source_kind=SourceKind.ANDROID_UIAUTOMATOR,
        source_payload_sha256=_sha256(xml_bytes),
        nodes=nodes,
        policy=_policy(args),
        app_version=args.app_version,
        locale=args.locale,
    )


def _compile_dom_files(args: argparse.Namespace) -> LessonArtifact:
    png_bytes = args.png.read_bytes()
    viewport = png_dimensions(png_bytes)
    if viewport is None:
        raise ValueError(f"not a readable PNG: {args.png}")
    dom_bytes = args.dom_json.read_bytes()
    payload: Any = json.loads(dom_bytes.decode("utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("DOM JSON must be a list or an object containing a records list")
    nodes = parse_dom_snapshot(records, viewport=viewport)
    canonical = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return compile_lesson(
        png_bytes,
        surface_id=args.surface_id,
        source_kind=SourceKind.WEB_DOM,
        source_payload_sha256=_sha256(canonical.encode("utf-8")),
        nodes=nodes,
        policy=_policy(args),
        app_version=args.app_version,
        locale=args.locale,
    )


def _capture_android_live(args: argparse.Namespace) -> LessonArtifact:
    from drivers import AndroidAdbDriver

    return capture_android_lesson(
        AndroidAdbDriver(),
        surface_id=args.surface_id,
        device=args.device,
        policy=_policy(args),
        app_version=args.app_version,
        locale=args.locale,
        alignment_attempts=args.alignment_attempts,
    )


def _emit(artifact: LessonArtifact, args: argparse.Namespace) -> None:
    out = stage_lesson(artifact, args.out)
    if args.atlas is not None:
        SurfaceAtlas(args.atlas).add(artifact.lesson)
    lesson = artifact.lesson
    print(
        json.dumps(
            {
                "lesson_dir": str(out),
                "lesson_id": lesson.lesson_id,
                "screen_key": lesson.screen_key,
                "surface_id": lesson.surface_id,
                "source_kind": lesson.source_kind,
                "grammar_hash": lesson.grammar_hash,
                "control_hash": lesson.control_hash,
                "content_hash": lesson.content_hash,
                "observation_hash": lesson.observation_hash,
                "element_count": len(lesson.elements),
                "explanation": lesson.explanation,
                "atlas": str(args.atlas) if args.atlas is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--surface-id", required=True, help="Stable app/activity or site surface identity")
    parser.add_argument("--out", type=Path, required=True, help="Lesson bundle directory")
    parser.add_argument("--atlas", type=Path, help="Optional teacher-blind SurfaceAtlas JSON path")
    parser.add_argument("--app-version", help="Optional declared application version")
    parser.add_argument("--locale", help="Optional declared UI locale")
    parser.add_argument(
        "--retain-values",
        action="store_true",
        help="Retain non-sensitive field values in the teacher record; runtime remains redacted",
    )
    parser.add_argument("--no-crops", action="store_true", help="Do not write derived per-element PNG crops")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile one ScreenGhost Surface Teacher lesson")
    modes = parser.add_subparsers(dest="mode", required=True)

    android_files = modes.add_parser("android-files", help="Compile an existing PNG and UI Automator XML")
    _common(android_files)
    android_files.add_argument("--png", type=Path, required=True)
    android_files.add_argument("--xml", type=Path, required=True)
    android_files.set_defaults(handler=_compile_android_files)

    dom_files = modes.add_parser("dom-files", help="Compile an existing PNG and DOM snapshot JSON")
    _common(dom_files)
    dom_files.add_argument("--png", type=Path, required=True)
    dom_files.add_argument("--dom-json", type=Path, required=True)
    dom_files.set_defaults(handler=_compile_dom_files)

    android_live = modes.add_parser("android-live", help="Read one aligned frame/tree pair from local ADB")
    _common(android_live)
    android_live.add_argument("--device", help="ADB serial; omit only when exactly one device is available")
    android_live.add_argument("--alignment-attempts", type=int, default=3)
    android_live.set_defaults(handler=_capture_android_live)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = args.handler(args)
    _emit(artifact, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
