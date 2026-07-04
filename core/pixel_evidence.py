"""ScreenGhost Pixel Evidence v0 — seal the rendered surface the user actually saw.

Rendered-surface capture, NOT clipboard ingestion. A clipboard copy is polluted
by DOM, hidden spans, rich text, embeds, quote cards, and platform formatting; a
screenshot is the actual visual surface the user saw. That pixel surface is the
evidence object this module produces.

The evidence tier is EXPLICIT and narrow -- ``pixel_capture``:
  - rendered surface only
  - NOT DOM truth
  - NOT API truth
  - NOT an authenticated platform record
  - NOT legal-grade provenance by itself.

Boundaries this module holds:
  - The PNG is treated as OPAQUE BYTES: hashed and carried verbatim, never
    decoded, re-encoded, OCR'd, or rewritten. (PNG width/height are read from the
    IHDR header only -- a read, never a rewrite.)
  - No author identity, timestamp, or URL is INFERRED. Those appear only when
    user-supplied via an optional sidecar.
  - No DOM/clipboard/view-tree parser is imported or invoked here.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EVIDENCE_TIER = "pixel_capture"
EVIDENCE_TIER_LIMITS: Tuple[str, ...] = (
    "rendered surface only",
    "not DOM truth",
    "not API truth",
    "not an authenticated platform record",
    "not legal-grade provenance by itself",
)

# Known capture methods (free-form is accepted; these are the documented ones).
CAPTURE_METHODS: Tuple[str, ...] = (
    "sharex_scrolling",
    "manual_screenshot",
    "browser_screenshot",
    "os_capture",
)

# Sidecar keys a user/tool may supply next to the PNG. NOTHING here is inferred.
SIDECAR_FIELDS: Tuple[str, ...] = (
    "url",
    "page_title",
    "app_name",
    "capture_tool",
    "user_note",
    "captured_at",
)

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

MANIFEST_NAME = "pixel_capture_manifest.json"
EVENT_NAME = "evidence_event.json"
PNG_NAME = "capture.png"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_png(data: bytes) -> bool:
    return data[:8] == _PNG_SIG


def png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Read (width, height) from the PNG IHDR header. Read-only: it parses the
    declared dimensions, it never decodes pixels or rewrites the file. Returns
    None if the bytes are not a PNG with a readable IHDR."""
    if not is_png(data) or len(data) < 24 or data[12:16] != b"IHDR":
        return None
    try:
        width, height = struct.unpack(">II", data[16:24])
    except struct.error:  # pragma: no cover - defensive
        return None
    return int(width), int(height)


@dataclass(frozen=True)
class PixelEvidenceCapture:
    """A rendered-surface evidence object. The PNG bytes live in the sealed
    ``capture.png``; this is the capture manifest that rides with them."""

    image_sha256: str                    # sha256 of the PNG bytes (the image hash)
    image_bytes: int                     # length of the PNG bytes
    source_label: str                    # source app or browser label (or "unknown")
    capture_method: str                  # sharex_scrolling | manual_screenshot | ...
    evidence_tier: str = EVIDENCE_TIER   # ALWAYS pixel_capture in v0
    image_format: str = "png"            # "png" if the signature is valid, else "unknown"
    image_width: Optional[int] = None    # declared PNG width (IHDR), if readable
    image_height: Optional[int] = None
    png_filename: str = PNG_NAME
    # --- user/sidecar-supplied only; never inferred -------------------------
    url: Optional[str] = None
    page_title: Optional[str] = None
    app_name: Optional[str] = None
    capture_tool: Optional[str] = None
    captured_at: Optional[str] = None
    capture_notes: Optional[str] = None

    def to_manifest(self) -> Dict[str, Any]:
        """The canonical, deterministic manifest dict. No wall-clock, no inferred
        fields -- so the manifest hash is stable for the same PNG + sidecar."""
        m = asdict(self)
        m["evidence_tier_limits"] = list(EVIDENCE_TIER_LIMITS)
        return m

    def manifest_bytes(self) -> bytes:
        """Byte-stable serialization of the manifest (sorted keys)."""
        return (json.dumps(self.to_manifest(), sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    def manifest_sha256(self) -> str:
        return sha256_hex(self.manifest_bytes())


@dataclass(frozen=True)
class EvidenceEvent:
    """A ScreenGhost evidence event pointing at a pixel capture. Deterministic:
    ``event_id`` is the image hash, so re-importing the same PNG is the same
    event."""

    event_id: str                        # == image_sha256 (stable identity)
    kind: str
    evidence_tier: str
    image_sha256: str
    manifest_sha256: str
    source_label: str
    capture_method: str
    url: Optional[str] = None
    captured_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sidecar_value(sidecar: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    if not sidecar:
        return None
    val = sidecar.get(key)
    return str(val) if val is not None else None


def build_capture(
    png_bytes: bytes,
    *,
    capture_method: str,
    source_label: Optional[str] = None,
    sidecar: Optional[Dict[str, Any]] = None,
) -> PixelEvidenceCapture:
    """Hash the PNG and assemble the capture manifest. Sidecar fields (url,
    page_title, app_name, capture_tool, user_note, captured_at) are carried ONLY
    if supplied -- nothing is inferred. The PNG bytes are not modified."""
    dims = png_dimensions(png_bytes)
    app_name = _sidecar_value(sidecar, "app_name")
    label = source_label or app_name or "unknown"
    return PixelEvidenceCapture(
        image_sha256=sha256_hex(png_bytes),
        image_bytes=len(png_bytes),
        source_label=label,
        capture_method=capture_method,
        image_format="png" if is_png(png_bytes) else "unknown",
        image_width=dims[0] if dims else None,
        image_height=dims[1] if dims else None,
        url=_sidecar_value(sidecar, "url"),
        page_title=_sidecar_value(sidecar, "page_title"),
        app_name=app_name,
        capture_tool=_sidecar_value(sidecar, "capture_tool"),
        captured_at=_sidecar_value(sidecar, "captured_at"),
        capture_notes=_sidecar_value(sidecar, "user_note"),
    )


def build_event(capture: PixelEvidenceCapture) -> EvidenceEvent:
    """Create the EvidenceEvent for a capture. Deterministic identity."""
    return EvidenceEvent(
        event_id=capture.image_sha256,
        kind="pixel_evidence",
        evidence_tier=capture.evidence_tier,
        image_sha256=capture.image_sha256,
        manifest_sha256=capture.manifest_sha256(),
        source_label=capture.source_label,
        capture_method=capture.capture_method,
        url=capture.url,
        captured_at=capture.captured_at,
    )


def stage_capture(
    png_bytes: bytes,
    capture: PixelEvidenceCapture,
    out_dir: str | Path,
) -> Path:
    """Write the pre-seal bundle: capture.png (VERBATIM bytes), the manifest, and
    the evidence event. The PNG is copied byte-for-byte; it is never re-encoded.
    Returns the staged directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / capture.png_filename).write_bytes(png_bytes)          # verbatim, no rewrite
    (out / MANIFEST_NAME).write_bytes(capture.manifest_bytes())
    event = build_event(capture)
    (out / EVENT_NAME).write_text(
        json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


class FilesystemIntakeProvider:
    """Read-only intake over a local folder where an external tool (e.g. ShareX)
    saves PNGs. ScreenGhost consumes the PNG and an OPTIONAL ``<stem>.json``
    sidecar. There is NO ShareX dependency and NO write back into the folder.
    """

    def __init__(self, intake_dir: str | Path) -> None:
        self._dir = Path(intake_dir)

    def list_captures(self) -> List[str]:
        """PNG filenames in the intake folder, sorted. Read-only."""
        if not self._dir.exists():
            return []
        return sorted(p.name for p in self._dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")

    def read_capture(self, name: str) -> Tuple[bytes, Optional[Dict[str, Any]]]:
        """Return (png_bytes, sidecar_or_None). Reads only; never modifies the
        intake folder or the PNG."""
        png_path = self._dir / name
        png_bytes = png_path.read_bytes()
        sidecar_path = png_path.with_suffix(".json")
        sidecar = None
        if sidecar_path.exists():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return png_bytes, sidecar

    def import_capture(
        self, name: str, *, capture_method: str, source_label: Optional[str] = None
    ) -> Tuple[bytes, PixelEvidenceCapture, EvidenceEvent]:
        """Read a capture from the folder and build (png_bytes, capture, event).
        Pure with respect to the source folder: nothing is written back."""
        png_bytes, sidecar = self.read_capture(name)
        capture = build_capture(
            png_bytes, capture_method=capture_method, source_label=source_label, sidecar=sidecar
        )
        return png_bytes, capture, build_event(capture)
