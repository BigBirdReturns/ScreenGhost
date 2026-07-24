"""Import vendor coordinate macros into one bounded, auditable representation."""
from __future__ import annotations

import configparser
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from experiments.emulator_fleet.schema import (
    CoordinateMacro,
    EmulatorVendor,
    MacroAction,
    MacroActionKind,
    clean_text,
    sha256_bytes,
)


class MacroParseError(ValueError):
    pass


def _without_comment(line: str) -> str:
    # LDPlayer examples use both # and // comments. Keep URL-like text intact by
    # only treating an inline // as a comment when preceded by whitespace.
    stripped = line.strip()
    if stripped.startswith("#") or stripped.startswith("//"):
        return ""
    line = re.split(r"\s+#", line, maxsplit=1)[0]
    line = re.split(r"\s+//", line, maxsplit=1)[0]
    return line.strip()


def _norm(value: float, extent: int) -> float:
    if extent <= 0:
        raise MacroParseError("macro resolution must be positive")
    result = float(value) / float(extent)
    if not 0.0 <= result <= 1.0:
        raise MacroParseError(f"coordinate {value} lies outside source extent {extent}")
    return round(result, 8)


def parse_ldplayer_macro(
    text: str,
    *,
    name: str = "LDPlayer macro",
    text_refs: Optional[Mapping[int, str]] = None,
) -> CoordinateMacro:
    """Parse LDPlayer's documented keyboard-macro language.

    Supported deterministic commands: ``size``, ``touch`` (tap or path), ``wait``,
    ``press``/``release``, ``key``, and ``text``. Trigger-dependent constructs such
    as ``loop``, ``ondown``, ``onup``, mouse-lock, and switch-mouse are retained as
    unsupported control actions rather than silently ignored.
    """

    source = text.encode("utf-8")
    width = height = None
    cursor_ms = 0.0
    actions: list[MacroAction] = []
    refs = dict(text_refs or {})
    pending_press: Optional[tuple[tuple[float, float], float, str]] = None

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = _without_comment(raw)
        if not line:
            continue
        parts = line.split()
        command = parts[0].lower()
        args = parts[1:]
        if command == "size":
            if len(args) != 2:
                raise MacroParseError(f"line {line_number}: size requires width height")
            try:
                width, height = int(args[0]), int(args[1])
            except ValueError as exc:
                raise MacroParseError(f"line {line_number}: invalid size") from exc
            if width <= 0 or height <= 0:
                raise MacroParseError(f"line {line_number}: invalid size")
            continue
        if width is None or height is None:
            raise MacroParseError(
                f"line {line_number}: coordinate command encountered before size"
            )
        sequence = len(actions)
        if command == "wait":
            if len(args) != 1:
                raise MacroParseError(f"line {line_number}: wait requires milliseconds")
            try:
                duration = float(args[0])
            except ValueError as exc:
                raise MacroParseError(f"line {line_number}: invalid wait") from exc
            actions.append(
                MacroAction(
                    sequence=sequence,
                    kind=MacroActionKind.WAIT,
                    at_ms=cursor_ms,
                    duration_ms=duration,
                    raw=line,
                )
            )
            cursor_ms += duration
            continue
        if command == "touch":
            try:
                nums = [float(v) for v in args]
            except ValueError as exc:
                raise MacroParseError(f"line {line_number}: non-numeric touch argument") from exc
            if len(nums) < 2:
                raise MacroParseError(f"line {line_number}: touch requires coordinates")
            duration = 0.0
            coord_values = nums
            # An odd count means the final value is the optional duration.
            if len(nums) % 2 == 1:
                duration = nums[-1]
                coord_values = nums[:-1]
            if len(coord_values) % 2:
                raise MacroParseError(f"line {line_number}: touch coordinate pairs are incomplete")
            points = tuple(
                (_norm(coord_values[i], width), _norm(coord_values[i + 1], height))
                for i in range(0, len(coord_values), 2)
            )
            if len(points) == 1:
                actions.append(
                    MacroAction(
                        sequence=sequence,
                        kind=MacroActionKind.TAP,
                        at_ms=cursor_ms,
                        point=points[0],
                        duration_ms=duration,
                        raw=line,
                    )
                )
            else:
                actions.append(
                    MacroAction(
                        sequence=sequence,
                        kind=MacroActionKind.SWIPE,
                        at_ms=cursor_ms,
                        path=points,
                        duration_ms=duration,
                        raw=line,
                    )
                )
            cursor_ms += duration
            continue
        if command == "press":
            try:
                nums = [float(v) for v in args]
            except ValueError as exc:
                raise MacroParseError(f"line {line_number}: invalid press") from exc
            if len(nums) < 3 or len(nums) % 2 == 0:
                raise MacroParseError(
                    f"line {line_number}: press requires coordinate pairs plus duration"
                )
            duration = nums[-1]
            coords = nums[:-1]
            if len(coords) != 2:
                # Multi-point long presses are retained as an unsupported control
                # rather than split into simultaneous actions incorrectly.
                actions.append(
                    MacroAction(
                        sequence=sequence,
                        kind=MacroActionKind.CONTROL,
                        at_ms=cursor_ms,
                        duration_ms=duration,
                        raw=line,
                        supported=False,
                    )
                )
                cursor_ms += duration
                continue
            pending_press = (
                (_norm(coords[0], width), _norm(coords[1], height)),
                duration,
                line,
            )
            continue
        if command == "release":
            if pending_press is None:
                raise MacroParseError(f"line {line_number}: release without press")
            point, duration, press_raw = pending_press
            actions.append(
                MacroAction(
                    sequence=sequence,
                    kind=MacroActionKind.LONG_PRESS,
                    at_ms=cursor_ms,
                    point=point,
                    duration_ms=duration,
                    raw=f"{press_raw}\n{line}",
                )
            )
            cursor_ms += duration
            pending_press = None
            continue
        if command == "key":
            key = clean_text(" ".join(args))
            if not key:
                raise MacroParseError(f"line {line_number}: key requires a value")
            actions.append(
                MacroAction(
                    sequence=sequence,
                    kind=MacroActionKind.KEY,
                    at_ms=cursor_ms,
                    key=key,
                    raw=line,
                )
            )
            continue
        if command == "text":
            raw_text = raw.strip()[len(parts[0]) :].lstrip()
            ref = refs.get(line_number) or f"macro_text_line_{line_number}"
            actions.append(
                MacroAction(
                    sequence=sequence,
                    kind=MacroActionKind.TEXT,
                    at_ms=cursor_ms,
                    text_ref=ref,
                    text_length=len(raw_text),
                    text_sha256=sha256_bytes(raw_text.encode("utf-8")),
                    raw="text <redacted>",
                )
            )
            continue
        if command in {
            "loop",
            "ondown",
            "onup",
            "switch-mouse",
            "type=mouse-lock",
            "type=cancel",
        } or command.startswith("press-"):
            actions.append(
                MacroAction(
                    sequence=sequence,
                    kind=MacroActionKind.CONTROL,
                    at_ms=cursor_ms,
                    raw=line,
                    supported=False,
                )
            )
            continue
        actions.append(
            MacroAction(
                sequence=sequence,
                kind=MacroActionKind.UNKNOWN,
                at_ms=cursor_ms,
                raw=line,
                supported=False,
            )
        )
    if pending_press is not None:
        raise MacroParseError("macro ended with press but no release")
    if width is None or height is None:
        raise MacroParseError("LDPlayer macro has no size command")
    return CoordinateMacro.create(
        name=name,
        vendor=EmulatorVendor.LDPLAYER,
        source_bytes=source,
        source_resolution=(width, height),
        actions=actions,
        metadata={"format": "ldplayer_keyboard_macro", "line_count": len(text.splitlines())},
    )


def _first_list(value: Any) -> Optional[list[Mapping[str, Any]]]:
    if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
        return list(value)
    if isinstance(value, Mapping):
        for key in (
            "events",
            "Events",
            "actions",
            "Actions",
            "recordings",
            "Recordings",
            "macro",
            "Macro",
        ):
            found = _first_list(value.get(key))
            if found is not None:
                return found
        for nested in value.values():
            found = _first_list(nested)
            if found is not None:
                return found
    return None


def _number(row: Mapping[str, Any], *names: str) -> Optional[float]:
    for name in names:
        if row.get(name) is None:
            continue
        try:
            return float(row[name])
        except (TypeError, ValueError):
            return None
    return None


def parse_bluestacks_macro_json(
    payload: str | bytes | Mapping[str, Any],
    *,
    name: str = "BlueStacks macro",
    default_resolution: Optional[tuple[int, int]] = None,
) -> CoordinateMacro:
    """Parse exported BlueStacks macro JSON conservatively.

    BlueStacks has changed field names across releases.  The parser accepts common
    aliases, records unknown events, and never guesses coordinates when the source
    resolution is absent.
    """

    if isinstance(payload, Mapping):
        data = dict(payload)
        source = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    else:
        source = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        try:
            data = json.loads(source.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MacroParseError(f"invalid BlueStacks macro JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise MacroParseError("BlueStacks macro root must be an object")
    width = _number(data, "width", "Width", "screenWidth", "ScreenWidth")
    height = _number(data, "height", "Height", "screenHeight", "ScreenHeight")
    resolution = data.get("resolution") or data.get("Resolution")
    if (width is None or height is None) and isinstance(resolution, Mapping):
        width = _number(resolution, "width", "Width", "x")
        height = _number(resolution, "height", "Height", "y")
    if (width is None or height is None) and default_resolution is not None:
        width, height = default_resolution
    if width is None or height is None or width <= 0 or height <= 0:
        raise MacroParseError("BlueStacks macro requires a source resolution")
    events = _first_list(data)
    if events is None:
        raise MacroParseError("BlueStacks macro contains no event list")
    actions: list[MacroAction] = []
    cursor = 0.0
    for row in events:
        event_type = str(
            row.get("type")
            or row.get("Type")
            or row.get("eventType")
            or row.get("EventType")
            or row.get("action")
            or "unknown"
        ).strip().lower()
        at = _number(row, "time", "Time", "timestamp", "Timestamp", "delay", "Delay")
        at_ms = cursor if at is None else max(cursor, at)
        duration = _number(row, "duration", "Duration", "durationMs", "DurationMs") or 0.0
        seq = len(actions)
        x = _number(row, "x", "X", "posX", "PosX")
        y = _number(row, "y", "Y", "posY", "PosY")
        x2 = _number(row, "x2", "X2", "endX", "EndX")
        y2 = _number(row, "y2", "Y2", "endY", "EndY")

        def point(px: float, py: float) -> tuple[float, float]:
            # Some exported formats already use normalized 0..1 values.
            return (
                round(px, 8) if 0 <= px <= 1 else _norm(px, int(width)),
                round(py, 8) if 0 <= py <= 1 else _norm(py, int(height)),
            )

        if event_type in {"tap", "click", "mouseclick", "mousedown", "mouse_up", "mouseup"} and x is not None and y is not None:
            actions.append(
                MacroAction(seq, MacroActionKind.TAP, at_ms, point=point(x, y), duration_ms=duration, raw=json.dumps(row, sort_keys=True))
            )
        elif event_type in {"swipe", "drag", "gesture"} and None not in (x, y, x2, y2):
            actions.append(
                MacroAction(
                    seq,
                    MacroActionKind.SWIPE,
                    at_ms,
                    path=(point(float(x), float(y)), point(float(x2), float(y2))),
                    duration_ms=duration,
                    raw=json.dumps(row, sort_keys=True),
                )
            )
        elif event_type in {"wait", "delay", "sleep"}:
            wait_ms = duration or _number(row, "value", "Value", "milliseconds") or 0.0
            actions.append(MacroAction(seq, MacroActionKind.WAIT, at_ms, duration_ms=wait_ms, raw=json.dumps(row, sort_keys=True)))
            cursor = at_ms + wait_ms
            continue
        elif event_type in {"key", "keypress", "keyboard"}:
            actions.append(
                MacroAction(seq, MacroActionKind.KEY, at_ms, key=clean_text(row.get("key") or row.get("Key") or row.get("value")), raw=json.dumps(row, sort_keys=True))
            )
        elif event_type in {"text", "input", "type"}:
            value = str(row.get("text") or row.get("Text") or row.get("value") or "")
            actions.append(
                MacroAction(
                    seq,
                    MacroActionKind.TEXT,
                    at_ms,
                    text_ref=str(row.get("textRef") or row.get("TextRef") or f"macro_text_event_{seq}"),
                    text_length=len(value),
                    text_sha256=sha256_bytes(value.encode("utf-8")),
                    raw="text <redacted>",
                )
            )
        else:
            actions.append(
                MacroAction(seq, MacroActionKind.UNKNOWN, at_ms, raw=json.dumps(row, sort_keys=True), supported=False)
            )
        cursor = max(cursor, at_ms + duration)
    return CoordinateMacro.create(
        name=name,
        vendor=EmulatorVendor.BLUESTACKS,
        source_bytes=source,
        source_resolution=(int(width), int(height)),
        actions=actions,
        metadata={"format": "bluestacks_export_json", "event_count": len(events)},
    )


def catalog_memu_macros(scripts_dir: str | Path) -> tuple[dict[str, Any], ...]:
    """Read MEmu Operation Recorder metadata without interpreting opaque ``.mir`` bytes."""

    root = Path(scripts_dir)
    info = root / "info.ini"
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if info.exists():
        parser.read(info, encoding="utf-8")
    records = []
    for path in sorted(root.glob("*.mir")):
        stem = path.stem
        section = next(
            (name for name in parser.sections() if name.replace("%20", " ").replace("%3A", ":") == stem or name == stem),
            None,
        )
        meta = dict(parser[section]) if section else {}
        records.append(
            {
                "schema": "memu_macro_artifact_v1",
                "path": str(path),
                "file_name": path.name,
                "sha256": sha256_bytes(path.read_bytes()),
                "size_bytes": path.stat().st_size,
                "name": meta.get("name") or stem,
                "replayTime": meta.get("replayTime"),
                "replayCycles": meta.get("replayCycles"),
                "replayAccelRates": meta.get("replayAccelRates"),
                "replayInterval": meta.get("replayInterval"),
                "actions_available": False,
                "reason": "MEmu .mir is retained as opaque vendor evidence; provide an observed action manifest for distillation",
            }
        )
    return tuple(records)


def load_macro(path: str | Path, *, format_hint: Optional[str] = None, default_resolution: Optional[tuple[int, int]] = None) -> CoordinateMacro:
    source = Path(path)
    hint = (format_hint or source.suffix.lstrip(".")).lower()
    if hint in {"ld", "ldplayer", "txt", "macro"}:
        return parse_ldplayer_macro(source.read_text(encoding="utf-8"), name=source.stem)
    if hint in {"json", "bluestacks", "bs"}:
        return parse_bluestacks_macro_json(source.read_bytes(), name=source.stem, default_resolution=default_resolution)
    raise MacroParseError(f"unsupported macro format: {hint!r}")
