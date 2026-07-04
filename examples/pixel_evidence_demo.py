"""ScreenGhost Pixel Evidence v0 — end-to-end demo.

    intake folder (ShareX/manual PNG + optional sidecar json)
      -> read-only filesystem intake
      -> hash + pixel_capture_manifest.json + EvidenceEvent (nothing inferred)
      -> seal capture.png (VERBATIM) + manifest through genesis
      -> verify with an out-of-band key
      -> exit test: verify with only shard bytes + oob pub
         (no ShareX, no ScreenGhost, no GhostBox, no browser).

ShareX stays OUTSIDE the codebase: this demo can synthesize a tiny sample PNG
(`--make-sample`) so it runs with no external tool, or point `--intake` at a real
folder where ShareX saves PNGs. The genesis kernel (axm-build / axm-verify) must
be on PATH for the seal/verify steps.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pixel_evidence import FilesystemIntakeProvider, stage_capture
from core.pixel_exit_test import verify_detached
from core.pixel_seal import kernel_available, seal_pixel_evidence, verify_pixel_evidence


def _sample_png(w: int = 6, h: int = 4) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(t: bytes, d: bytes) -> bytes:
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x22\x44\x66" * w for _ in range(h))
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _make_sample_intake(intake: Path) -> str:
    intake.mkdir(parents=True, exist_ok=True)
    (intake / "capture.png").write_bytes(_sample_png())
    (intake / "capture.json").write_text(
        json.dumps(
            {
                "url": "https://example.social/status/123",
                "page_title": "a post that looked edited",
                "app_name": "Chrome",
                "capture_tool": "ShareX",
                "user_note": "quote card rendering differed from the copied text",
                "captured_at": "2026-07-04T12:00:00Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return "capture.png"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run ScreenGhost Pixel Evidence v0 end to end.")
    ap.add_argument("--intake", default=None, help="folder where PNGs (+ optional .json sidecars) land")
    ap.add_argument("--capture", default=None, help="PNG filename inside the intake folder")
    ap.add_argument("--capture-method", default="sharex_scrolling")
    ap.add_argument("--make-sample", action="store_true", help="synthesize a tiny sample PNG + sidecar and use it")
    ap.add_argument("--out", default="pixel_evidence_out")
    args = ap.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="pixel_evidence_"))
    if args.make_sample or not args.intake:
        intake = Path(args.intake) if args.intake else work / "intake"
        name = _make_sample_intake(intake)
    else:
        intake, name = Path(args.intake), args.capture
        if not name:
            names = FilesystemIntakeProvider(intake).list_captures()
            if not names:
                print(f"no PNGs found in {intake}", file=sys.stderr)
                return 2
            name = names[0]

    prov = FilesystemIntakeProvider(intake)
    png_bytes, capture, event = prov.import_capture(name, capture_method=args.capture_method)

    out = Path(args.out)
    stage_capture(png_bytes, capture, out / "bundle")

    receipt = {
        "artifact": "ScreenGhost Pixel Evidence v0",
        "evidence_tier": capture.evidence_tier,
        "evidence_tier_limits": list(capture.to_manifest()["evidence_tier_limits"]),
        "capture_method": capture.capture_method,
        "source_label": capture.source_label,
        "image_sha256": capture.image_sha256,
        "manifest_sha256": capture.manifest_sha256(),
        "image_bytes": capture.image_bytes,
        "image_dimensions": [capture.image_width, capture.image_height],
        "sidecar_supplied": {
            "url": capture.url,
            "page_title": capture.page_title,
            "captured_at": capture.captured_at,
        },
        "event_id": event.event_id,
    }

    if kernel_available():
        shard = seal_pixel_evidence(png_bytes, capture, out / "shard", work_dir=out / "seal")
        status = verify_pixel_evidence(shard.shard_dir, shard.trusted_key_path)
        detached = verify_detached(shard.shard_dir, shard.trusted_key_path)
        png_verbatim = (Path(shard.shard_dir) / "content" / "capture.png").read_bytes() == png_bytes
        receipt.update(
            sealed=True,
            shard_id=shard.shard_id,
            suite=shard.suite,
            verification=status.value,
            png_verbatim_in_shard=png_verbatim,
            intake_png_untouched=(intake / name).read_bytes() == png_bytes,
            detached={
                "status": detached["status"],
                "sharex_involved": detached["sharex_involved"],
                "screenghost_involved": detached["screenghost_involved"],
                "ghostbox_involved": detached["ghostbox_involved"],
                "browser_involved": detached["browser_involved"],
            },
        )
        ok = status.value == "pass" and detached["status"] == "PASS" and png_verbatim
    else:
        receipt.update(sealed=False, note="genesis kernel not on PATH; hashed + staged but not sealed")
        ok = True

    print(json.dumps(receipt, indent=2))
    print(f"[pixel evidence v0: {'OK' if ok else 'INCOMPLETE'}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
