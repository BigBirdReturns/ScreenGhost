"""Shared read-only contracts for Surface Teacher source adapters."""
from __future__ import annotations

import hashlib
import io
import json
from typing import Any, Optional, Tuple

from PIL import Image

from core.surface_teacher import LessonRefused


class SourceParseError(LessonRefused):
    """The privileged source could not be interpreted without guessing."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_pixel_sha256(png_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
        image = image.convert("RGB")
    except Exception as exc:
        raise SourceParseError(f"capture is not a decodable PNG: {exc}") from exc
    header = json.dumps({"mode": image.mode, "size": image.size}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def image_png(image: Any) -> Tuple[bytes, Tuple[int, int]]:
    if not hasattr(image, "save") or not hasattr(image, "size"):
        raise SourceParseError("screencap did not return an image-like object")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), tuple(image.size)


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact or None
