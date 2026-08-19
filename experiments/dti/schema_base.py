"""Domain contracts for the ScreenGhost Dress to Impress cartridge.

The cartridge consumes ordinary rendered pixels and emits bounded desktop input.
It does not read Roblox memory, inject code, modify the client, or own planning
authority outside the declared DTI round.  Every durable record is canonical and
content-addressed so the live adapter can be replaced without changing custody.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
Point = Tuple[float, float]
Bounds = Tuple[float, float, float, float]


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact or None


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def content_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{sha256_json(value)}"


def normalized_point(value: Sequence[Any]) -> Point:
    if len(value) != 2:
        raise ValueError("point requires two values")
    x, y = (float(v) for v in value)
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError("point contains a non-finite value")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("normalized point must lie inside [0,1]")
    return (x, y)


def normalized_bounds(value: Sequence[Any]) -> Bounds:
    if len(value) != 4:
        raise ValueError("bounds require four values")
    x1, y1, x2, y2 = (float(v) for v in value)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("bounds contain a non-finite value")
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("normalized bounds must lie inside [0,1]")
    return (x1, y1, x2, y2)


class PlayVenue(str, Enum):
    FREEPLAY = "freeplay"
    PRIVATE_SERVER = "private_server"
    PUBLIC_SERVER = "public_server"


class ControlMode(str, Enum):
    FAMILY_COPILOT = "family_copilot"
    ASSISTED = "assisted"
    AUTONOMOUS = "autonomous"


class RoundPhase(str, Enum):
    UNKNOWN = "unknown"
    LOBBY = "lobby"
    FREEPLAY = "freeplay"
    THEME_BRIEF = "theme_brief"
    DRESSING = "dressing"
    RUNWAY_READY = "runway_ready"
    RUNWAY = "runway"
    VOTING = "voting"
    RESULTS = "results"
    RECOVERY = "recovery"
    HALTED = "halted"


class GameActionKind(str, Enum):
    PRESS = "press"
    HOLD = "hold"
    CLICK = "click"
    MOVE_MOUSE = "move_mouse"
    SCROLL = "scroll"
    WAIT = "wait"


class AccessTier(str, Enum):
    STANDARD = "standard"
    CODE = "code"
    EVENT = "event"
    VIP = "vip"
    REMOVED = "removed"


