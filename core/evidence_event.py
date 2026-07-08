"""ScreenGhost's level-1 boundary object: the EvidenceEvent emitter.

ScreenGhost is intake, not truth (see docs/THREAT_GEOMETRY_INTAKE.md). Its job
in the AXM layering is to turn a surface observation into a structured evidence
event that the level-2 attention layer (GhostBox) can ingest. The canonical
shape of that event is defined by GhostBox's interop contract at
``ghostbox/interop/contracts.py``. This module is the ScreenGhost-side emitter.

Why a mirror instead of an import
---------------------------------
ScreenGhost has no dependency on GhostBox and must not grow one — the repos are
peers across a contract seam, not a monolith. So the ID derivation is
*reimplemented here to be byte-for-byte identical* to the contract, and the
conformance test (tests/test_evidence_event.py) pins that equality against golden
values produced by the GhostBox contract itself. Same content in either repo →
same content-addressed ``event_id``, with no shared code. If the two ever drift,
the test fails; the contract module wins and this mirror is the bug.

Honesty note
------------
``provenance = PROVEN`` on an EvidenceEvent asserts that the *capture of the
surface state is faithful* — the bytes on screen were recorded without
corruption. It asserts nothing about whether the captured content is accurate or
true. That distinction is the same one the rest of ScreenGhost enforces
(no category borrows trust) carried onto the level-1 boundary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Hashing / stable content-addressed IDs — mirror of ghostbox/interop/contracts.py.
# Kept byte-for-byte compatible on purpose; pinned by the conformance test.
#
# The algorithm is pinned to SHA-256 as a fixed contract constant. It must NOT be
# an environment-dependent choice: an optional blake3 backend would make the same
# EvidenceEvent produce evt:b3:... where blake3 is installed and evt:sha256:...
# where it is not, breaking the cross-repo promise that one observation yields
# one event_id everywhere. This mirrors the same pin on the GhostBox side.
# ---------------------------------------------------------------------------

import hashlib

_HASH_TAG = "sha256"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    raise TypeError(f"non-canonicalizable type: {type(value).__name__}")


def canonical_bytes(payload: Dict[str, Any]) -> bytes:
    """Deterministic serialization for content addressing.

    Sorted keys, no insignificant whitespace, utf-8. Identical to the contract.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def content_id(prefix: str, identity: Dict[str, Any]) -> str:
    """Stable, content-addressed ID of the form ``<prefix>:<tag>:<hex>``."""
    return f"{prefix}:{_HASH_TAG}:{_digest(canonical_bytes(identity))[:32]}"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProvenanceState(str, Enum):
    """Mirror of the contract vocabulary. PROVEN = faithful capture, not truth."""

    PROVEN = "proven"
    SIMULATED = "simulated"
    FROZEN = "frozen"
    UNTESTED = "untested"


@dataclass
class EvidenceEvent:
    """What the surface showed. ScreenGhost's level-1 emission.

    Field order and the ``event_id`` identity set are fixed by the contract.
    Do not reorder the identity dict in __post_init__ without updating the
    conformance golden — the ID depends on canonical bytes, not field order,
    but the *set* of identity fields is the contract.
    """

    source: str
    surface: str  # api | view_tree | screen | ui | chat | ...
    observation: Dict[str, Any]
    captured_at: str = field(default_factory=now_utc)
    raw_ref: Optional[str] = None
    provenance: ProvenanceState = ProvenanceState.PROVEN
    event_id: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = content_id(
                "evt",
                {
                    "source": self.source,
                    "surface": self.surface,
                    "observation": self.observation,
                    "captured_at": self.captured_at,
                    "raw_ref": self.raw_ref,
                },
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "surface": self.surface,
            "observation": self.observation,
            "captured_at": self.captured_at,
            "raw_ref": self.raw_ref,
            "provenance": self.provenance.value,
            "event_id": self.event_id,
        }


# ---------------------------------------------------------------------------
# Mapping: observer-mode output -> EvidenceEvent
# ---------------------------------------------------------------------------

# ScreenGhost inherits the user's own access, tried in order (see README:
# "The ghost inherits the user's system"). The capture path a state came from
# determines the boundary's `surface` value, so downstream attention knows how
# trustworthy the extraction path was without re-deriving it.
_SURFACE_BY_SOURCE_TYPE = {
    "api": "api",              # official platform event the user's session got
    "view_tree": "view_tree",  # exact on-screen text from the UI tree
    "photonic": "screen",      # last-resort vision capture of the screen
}


def evidence_from_observation(
    observation: Dict[str, Any],
    *,
    source: Optional[str] = None,
    raw_ref: Optional[str] = None,
    provenance: ProvenanceState = ProvenanceState.PROVEN,
) -> EvidenceEvent:
    """Turn one ScreenGhost observer-mode observation into an EvidenceEvent.

    ``observation`` is the structured state ScreenGhost already produces
    (``{"app", "screen", "elements", "topic", "confidence", ...}`` plus a
    ``source_type`` describing the capture path). The capture path is lifted to
    the boundary ``surface`` field; everything else rides in ``observation``
    verbatim so the content-addressed ID covers the full captured state.

    This does not assert the observation is *true*. It asserts the capture is
    faithful. A vision-path capture and a view-tree capture of the same screen
    are different surfaces with different trust, and that difference is preserved
    here rather than flattened.
    """
    src = source or observation.get("source") or "screen_ghost"
    source_type = observation.get("source_type", "photonic")
    surface = _SURFACE_BY_SOURCE_TYPE.get(source_type, source_type)
    captured_at = observation.get("timestamp") or now_utc()
    return EvidenceEvent(
        source=src,
        surface=surface,
        observation=observation,
        captured_at=captured_at,
        raw_ref=raw_ref,
        provenance=provenance,
    )


if __name__ == "__main__":  # pragma: no cover - manual self-test
    obs = {
        "topic": "Settings.Display",
        "source": "screen_ghost",
        "source_type": "photonic",
        "app": "Settings",
        "screen": "Display",
        "confidence": 0.8,
        "elements": [{"type": "toggle", "label": "Dark Mode", "value": "off"}],
        "timestamp": "2025-12-10T03:45:00Z",
    }
    ev = evidence_from_observation(obs, raw_ref="log/screenshots/0001.png")
    print(json.dumps(ev.to_dict(), ensure_ascii=False, indent=2))
    print("hash algorithm in use:", _HASH_TAG)
