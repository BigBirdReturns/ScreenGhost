"""Seal a ScreenGhost pixel-evidence capture through genesis; verify with an
out-of-band key.

Custody and verification are genesis's. This module only DRIVES the real
``axm-build`` / ``axm-verify`` surface: it seals ``capture.png`` (verbatim bytes)
plus ``pixel_capture_manifest.json`` into a normal ``axm-hybrid1`` shard and
verifies it with an out-of-band publisher key. It reproduces the frozen
``PASS / FAIL / MALFORMED / NO_TRUSTED_KEY`` taxonomy WITHOUT importing GhostBox
or any attention-layer code -- ScreenGhost intake must not depend on GhostBox.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from core.pixel_evidence import (
    MANIFEST_NAME,
    PNG_NAME,
    PixelEvidenceCapture,
    stage_capture,
)

AXM_BUILD = "axm-build"
AXM_VERIFY = "axm-verify"


class VerifyStatus(str, Enum):
    """Mirror of the genesis custody-seam taxonomy (frozen exit codes)."""

    PASS = "pass"
    FAIL = "fail"
    MALFORMED = "malformed"
    NO_TRUSTED_KEY = "no_trusted_key"


@dataclass(frozen=True)
class SealedPixelShard:
    shard_id: str            # genesis-derived sh1_, the ONLY custody identity
    shard_dir: str
    trusted_key_path: str    # out-of-band publisher pub (sibling to the shard)
    suite: str
    image_sha256: str        # the sealed image hash, carried for the receipt


def kernel_available() -> bool:
    return shutil.which(AXM_BUILD) is not None and shutil.which(AXM_VERIFY) is not None


def _candidates_and_source(capture: PixelEvidenceCapture, namespace: str) -> Tuple[List[dict], str]:
    """Turn the pixel capture into genesis candidates + a source.txt the claims
    cite. The PNG file and the manifest are sealed as content; the claims record
    WHAT the sealed pixels are, at the ``pixel_capture`` tier -- never that they
    are DOM/API/platform truth. Byte offsets are computed so evidence matches."""
    surface = capture.url or capture.source_label or "rendered-surface"
    png = capture.png_filename
    tier = capture.evidence_tier

    def _ent(label: str, etype: str) -> dict:
        return {"type": "entity", "namespace": namespace, "label": label, "entity_type": etype}

    entities = {
        png: _ent(png, "pixel_capture"),
        surface: _ent(surface, "rendered_surface"),
        tier: _ent(tier, "evidence_tier"),
    }
    statements = [
        (f"{png} is a pixel_capture of {surface}",
         {"subject_label": png, "predicate": "is_pixel_capture_of", "object_label": surface}),
        (f"{png} has evidence tier {capture.evidence_tier}",
         {"subject_label": png, "predicate": "has_evidence_tier", "object_label": capture.evidence_tier}),
    ]

    claims: List[dict] = []
    source = ""
    for stmt, base in statements:
        start = len(source.encode("utf-8"))
        source += stmt
        end = len(source.encode("utf-8"))
        source += "\n"
        claims.append(
            {
                "type": "claim",
                "subject_label": base["subject_label"],
                "predicate": base["predicate"],
                "object_label": base["object_label"],
                "object_type": "entity",
                "tier": 1,
                "evidence": {"source_file": "source.txt", "byte_start": start, "byte_end": end, "text": stmt},
            }
        )
    return list(entities.values()) + claims, source


def seal_pixel_evidence(
    png_bytes: bytes,
    capture: PixelEvidenceCapture,
    out_shard_dir: str | Path,
    *,
    work_dir: Optional[str | Path] = None,
    namespace: str = "screenghost/pixel",
    title: str = "ScreenGhost pixel evidence",
    created_at: str = "2026-07-04T00:00:00Z",
) -> SealedPixelShard:
    """Seal capture.png (verbatim) + the manifest into an axm-hybrid1 shard.

    The PNG bytes are copied byte-for-byte into content/; genesis never rewrites
    them. Returns the sealed shard with the genesis-derived ``sh1_`` custody id.
    """
    out_shard_dir = Path(out_shard_dir)
    work = Path(work_dir) if work_dir else out_shard_dir.parent
    content_dir = work / "_content"
    key_dir = work / "keys"
    if content_dir.exists():
        shutil.rmtree(content_dir)
    content_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    # content/ = the pixel bundle (png verbatim + manifest + event) + source.txt.
    stage_capture(png_bytes, capture, content_dir)
    candidates, source_text = _candidates_and_source(capture, namespace)
    (content_dir / "source.txt").write_text(source_text, encoding="utf-8")
    candidates_path = work / "candidates.jsonl"
    candidates_path.write_text("\n".join(json.dumps(c) for c in candidates) + "\n", encoding="utf-8")

    key_path = key_dir / "publisher.key"
    pub_path = key_dir / "publisher.pub"
    if not (key_path.exists() and pub_path.exists()):
        subprocess.run([AXM_BUILD, "keygen", str(key_dir), "--name", "publisher"], check=True, capture_output=True, text=True)

    subprocess.run(
        [
            AXM_BUILD, "compile", str(candidates_path), str(content_dir), str(out_shard_dir),
            "--private-key", str(key_path),
            "--namespace", namespace, "--title", title, "--created-at", created_at,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_bytes = (out_shard_dir / "manifest.json").read_bytes()
    m = json.loads(manifest_bytes)
    from axm_verify.crypto import derive_shard_id  # genesis owns custody id derivation

    return SealedPixelShard(
        shard_id=derive_shard_id(manifest_bytes),
        shard_dir=str(out_shard_dir),
        trusted_key_path=str(pub_path),
        suite=m.get("suite", "axm-hybrid1"),
        image_sha256=capture.image_sha256,
    )


def verify_pixel_evidence(shard_dir: str | Path, trusted_key: Optional[str | Path]) -> VerifyStatus:
    """Verify through genesis with an out-of-band key.

    No key -> NO_TRUSTED_KEY, decided before invoking the CLI (never falls back to
    the shard's own embedded key). Otherwise map genesis's frozen exit code.
    """
    if not trusted_key:
        return VerifyStatus.NO_TRUSTED_KEY
    code = subprocess.run(
        [AXM_VERIFY, "shard", str(shard_dir), "--trusted-key", str(trusted_key)],
        capture_output=True,
        text=True,
    ).returncode
    if code == 0:
        return VerifyStatus.PASS
    if code == 2:
        return VerifyStatus.MALFORMED
    return VerifyStatus.FAIL
