"""ScreenGhost Pixel Evidence v0 — rendered-surface capture, sealed through genesis.

Pure tests (import, manifest, hashes, boundaries) always run. Seal/verify tests
skip cleanly without the genesis kernel (axm-build / axm-verify). Evidence tier:
``pixel_capture`` -- rendered surface only, not DOM/API/platform truth.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from core.pixel_evidence import (
    EVIDENCE_TIER,
    EvidenceEvent,
    FilesystemIntakeProvider,
    PixelEvidenceCapture,
    build_capture,
    build_event,
    png_dimensions,
    stage_capture,
)
from core.pixel_seal import (
    SealedPixelShard,
    VerifyStatus,
    kernel_available,
    seal_pixel_evidence,
    verify_pixel_evidence,
)
from core.pixel_exit_test import verify_detached

requires_kernel = pytest.mark.skipif(
    not kernel_available(), reason="axm-genesis kernel (axm-build / axm-verify) not on PATH"
)

SIDECAR = {
    "url": "https://example.social/status/123",
    "page_title": "a post that looked edited",
    "app_name": "Chrome",
    "capture_tool": "ShareX",
    "user_note": "quote card rendering differed from the copied text",
    "captured_at": "2026-07-04T12:00:00Z",
}


def _png(w: int = 4, h: int = 3, fill: bytes = b"\x10\x20\x30") -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(t: bytes, d: bytes) -> bytes:
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + fill * w for _ in range(h))
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


@pytest.fixture
def intake(tmp_path):
    d = tmp_path / "intake"
    d.mkdir()
    (d / "capture.png").write_bytes(_png())
    (d / "capture.json").write_text(__import__("json").dumps(SIDECAR), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    if not kernel_available():
        pytest.skip("kernel not available")
    work = tmp_path_factory.mktemp("pix")
    pb = _png()
    capture = build_capture(pb, capture_method="sharex_scrolling", sidecar=SIDECAR)
    shard = seal_pixel_evidence(pb, capture, work / "shard", work_dir=work / "seal")
    return pb, capture, shard


# === requirement: PNG import produces a manifest ============================


def test_png_import_produces_a_manifest(intake):
    prov = FilesystemIntakeProvider(intake)
    assert prov.list_captures() == ["capture.png"]
    pb, capture, event = prov.import_capture("capture.png", capture_method="sharex_scrolling")
    m = capture.to_manifest()
    assert m["image_sha256"] == capture.image_sha256
    assert m["capture_method"] == "sharex_scrolling"
    assert m["evidence_tier"] == "pixel_capture"
    assert m["image_bytes"] == len(pb)
    assert isinstance(event, EvidenceEvent)


# === requirement: evidence tier is pixel_capture (explicit + bounded) =======


def test_evidence_tier_is_pixel_capture():
    capture = build_capture(_png(), capture_method="manual_screenshot")
    assert capture.evidence_tier == EVIDENCE_TIER == "pixel_capture"
    limits = capture.to_manifest()["evidence_tier_limits"]
    assert "not DOM truth" in limits and "not API truth" in limits
    assert "not legal-grade provenance by itself" in limits


# === requirement: image hash is stable ======================================


def test_image_hash_is_stable():
    pb = _png()
    a = build_capture(pb, capture_method="os_capture")
    b = build_capture(pb, capture_method="os_capture")
    assert a.image_sha256 == b.image_sha256
    assert a.image_sha256 == __import__("hashlib").sha256(pb).hexdigest()


# === requirement: manifest hash is stable ===================================


def test_manifest_hash_is_stable():
    pb = _png()
    a = build_capture(pb, capture_method="os_capture", sidecar=SIDECAR)
    b = build_capture(pb, capture_method="os_capture", sidecar=SIDECAR)
    assert a.manifest_sha256() == b.manifest_sha256()
    assert a.manifest_bytes() == b.manifest_bytes()  # byte-deterministic


# === requirement: PNG bytes are unchanged after import ======================


def test_png_bytes_unchanged_after_import_and_staging(intake, tmp_path):
    prov = FilesystemIntakeProvider(intake)
    original = (intake / "capture.png").read_bytes()
    pb, capture, _ = prov.import_capture("capture.png", capture_method="sharex_scrolling")
    assert pb == original
    staged = stage_capture(pb, capture, tmp_path / "stage")
    assert (staged / "capture.png").read_bytes() == original      # staged verbatim
    assert (intake / "capture.png").read_bytes() == original      # intake untouched (read-only)


def test_png_dimensions_from_ihdr():
    assert png_dimensions(_png(7, 5)) == (7, 5)
    assert png_dimensions(b"not a png") is None


# === requirement: nothing is inferred without a sidecar =====================


def test_no_url_timestamp_or_identity_inferred_without_sidecar():
    capture = build_capture(_png(), capture_method="manual_screenshot")
    assert capture.url is None
    assert capture.captured_at is None
    assert capture.page_title is None
    assert capture.app_name is None
    assert capture.source_label == "unknown"


def test_sidecar_fields_are_carried_only_when_supplied():
    capture = build_capture(_png(), capture_method="browser_screenshot", sidecar=SIDECAR)
    assert capture.url == SIDECAR["url"]
    assert capture.page_title == SIDECAR["page_title"]
    assert capture.capture_tool == "ShareX"
    assert capture.captured_at == SIDECAR["captured_at"]
    assert capture.capture_notes == SIDECAR["user_note"]
    event = build_event(capture)
    assert event.url == SIDECAR["url"] and event.captured_at == SIDECAR["captured_at"]


# === requirement: sealed shard verifies with correct key ====================


@requires_kernel
def test_sealed_shard_verifies_with_correct_key(sealed):
    _pb, _cap, shard = sealed
    assert isinstance(shard, SealedPixelShard)
    assert shard.shard_id.startswith("sh1_") and shard.suite == "axm-hybrid1"
    assert verify_pixel_evidence(shard.shard_dir, shard.trusted_key_path) is VerifyStatus.PASS


# === requirement: wrong key fails ===========================================


@requires_kernel
def test_wrong_key_fails(sealed, tmp_path):
    _pb, _cap, shard = sealed
    subprocess.run(["axm-build", "keygen", str(tmp_path), "--name", "attacker"], check=True, capture_output=True, text=True)
    assert verify_pixel_evidence(shard.shard_dir, tmp_path / "attacker.pub") is VerifyStatus.FAIL


# === requirement: missing key refuses =======================================


def test_missing_key_refuses(tmp_path):
    # No key -> NO_TRUSTED_KEY, decided before any CLI call (kernel not needed).
    assert verify_pixel_evidence(tmp_path, None) is VerifyStatus.NO_TRUSTED_KEY


# === requirement: PNG verbatim inside the sealed shard ======================


@requires_kernel
def test_sealed_png_is_byte_identical_to_the_capture(sealed):
    pb, _cap, shard = sealed
    assert (Path(shard.shard_dir) / "content" / "capture.png").read_bytes() == pb


@requires_kernel
def test_shard_id_is_genesis_derived(sealed):
    from axm_verify.crypto import derive_shard_id

    _pb, _cap, shard = sealed
    manifest_bytes = (Path(shard.shard_dir) / "manifest.json").read_bytes()
    assert shard.shard_id == derive_shard_id(manifest_bytes)


# === requirement: detached verification survives ShareX/ScreenGhost/GhostBox/browser


@requires_kernel
def test_detached_verification_survives_everything(sealed):
    _pb, _cap, shard = sealed
    res = verify_detached(shard.shard_dir, shard.trusted_key_path)
    assert res["status"] == "PASS" and res["exit_code"] == 0
    assert res["sharex_involved"] is False
    assert res["screenghost_involved"] is False
    assert res["ghostbox_involved"] is False
    assert res["browser_involved"] is False


# === requirement: no GhostBox, no DOM/clipboard parser ======================

_PIXEL_MODULES = ("core.pixel_evidence", "core.pixel_seal", "core.pixel_exit_test")
_FORBIDDEN_MODULES = (
    "ghostbox",
    "core.adapter",     # the view-tree / DOM (xml.etree) parser
    "core.texttree",    # xml text-tree parser
    "core.capture",     # the candidate capture ladder
    "core.ingest",
    "xml.etree.ElementTree",
)


def test_pixel_path_imports_no_ghostbox_no_dom_or_clipboard_parser():
    # Import ONLY the pixel modules in a clean subprocess and assert none of the
    # DOM/clipboard/view-tree parsers or ghostbox were pulled in.
    code = (
        "import importlib, sys\n"
        f"for m in {_PIXEL_MODULES!r}: importlib.import_module(m)\n"
        f"bad=[m for m in {_FORBIDDEN_MODULES!r} "
        "if m in sys.modules or any(k==m or k.startswith(m+'.') for k in sys.modules)]\n"
        "print('BAD:'+','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    line = [l for l in out.stdout.splitlines() if l.startswith("BAD:")][0]
    assert line == "BAD:", f"pixel path pulled in forbidden modules: {line}"


def test_pixel_source_imports_no_dom_or_clipboard_parser_or_ghostbox():
    # Check IMPORT statements and CALL sites, not docstring prose -- the modules
    # legitimately NAME clipboard/DOM to say they exclude them.
    import inspect

    from core import pixel_evidence, pixel_exit_test, pixel_seal

    # forbidden import-statement forms + call names that never appear in prose
    forbidden = [
        "import ghostbox", "from ghostbox",
        "from core.adapter", "import core.adapter",
        "from core.texttree", "import core.texttree",
        "from core.capture", "import core.capture",
        "from core.ingest", "import core.ingest",
        "from xml.etree", "import xml.etree",
        "classify_payload(", "extract(",
    ]
    for mod in (pixel_evidence, pixel_seal, pixel_exit_test):
        src = inspect.getsource(mod)
        for token in forbidden:
            assert token not in src, f"{mod.__name__} must not use {token!r}"
