"""Interface Procedure v0 — bounded operation, drift discipline, sealed trace.

Three gates of tests:
  - pure: runner logic against a call-logging FakeDriver (always runs). This is
    where the non-improvisation property is proven: after any drift, the driver
    log shows no further actions.
  - kernel: the trace seals through real genesis and verifies with an
    out-of-band key (skips without axm-build / axm-verify).
  - browser: the full chain against real Chromium rendering the local fixture
    dashboard (skips without playwright + a chromium executable).

Evidence tier: ``interface_procedure_trace`` -- rendered pixels + one bounded
action + visible verification only; never vendor backend state or API truth.
"""
from __future__ import annotations

import inspect
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from core.interface_procedure import (
    ApprovedBounds,
    DriftReason,
    InterfaceProcedure,
    Outcome,
    ProcedureRefused,
    ProcedureRunner,
    TRACE_TIER,
)
from core.pixel_seal import VerifyStatus, kernel_available
from core.procedure_seal import build_trace_bundle, seal_trace, verify_trace

requires_kernel = pytest.mark.skipif(
    not kernel_available(), reason="axm-genesis kernel (axm-build / axm-verify) not on PATH"
)

FIXTURES = Path(__file__).resolve().parent.parent / "examples" / "fixtures" / "interface_surface"
CHROMIUM = "/opt/pw-browsers/chromium"


def _browser_available() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return Path(CHROMIUM).exists()


requires_browser = pytest.mark.skipif(
    not _browser_available(), reason="playwright + chromium not available"
)


def _png(fill: bytes = b"\x10\x20\x30") -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(t: bytes, d: bytes) -> bytes:
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 4, 3, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + fill * 4 for _ in range(3))
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


ANCHOR_SEL, TARGET_SEL, VERIFY_SEL = "#dashboard-title", "#tile-living-room-lamp", "#lamp-state"


def _procedure(**over) -> InterfaceProcedure:
    kw = dict(
        procedure_id="proc-lamp-on-v0",
        surface_label="local-fixture-surface",
        anchor_selector=ANCHOR_SEL,
        anchor_text="Home Dashboard",
        target_selector=TARGET_SEL,
        target_label="Living Room Lamp",
        approved_bounds=ApprovedBounds(x=20, y=100, width=300, height=160),
        verify_selector=VERIFY_SEL,
        verify_expected_text="On",
        verify_timeout_ms=200,
    )
    kw.update(over)
    return InterfaceProcedure(**kw)


class FakeDriver:
    """A scriptable surface with a full call log — the non-improvisation proof."""

    def __init__(self, *, anchor="Home Dashboard", label="Living Room Lamp Off",
                 box=(40.0, 120.0, 224.0, 104.0), toggles=True) -> None:
        self._texts = {ANCHOR_SEL: anchor, TARGET_SEL: label, VERIFY_SEL: "Off"}
        self._box = box
        self._toggles = toggles
        self.calls: list = []

    def screenshot(self) -> bytes:
        self.calls.append("screenshot")
        return _png(bytes([len(self.calls) % 256, 0x20, 0x30]))  # distinct frames

    def text(self, selector):
        self.calls.append(("text", selector))
        return self._texts.get(selector)

    def find_box(self, selector):
        self.calls.append(("find_box", selector))
        return self._box if selector == TARGET_SEL else None

    def click(self, x, y):
        self.calls.append(("click", x, y))
        if self._toggles:
            self._texts[VERIFY_SEL] = "On"

    def wait_text(self, selector, expected, timeout_ms):
        self.calls.append(("wait_text", selector, expected))
        return self._texts.get(selector, "").strip() == expected

    def actions(self):
        return [c for c in self.calls if isinstance(c, tuple) and c[0] == "click"]


# === pure: the happy path ====================================================


def test_completed_run_produces_full_trace():
    driver = FakeDriver()
    trace = ProcedureRunner().run(driver, _procedure())
    assert trace.outcome is Outcome.COMPLETED and trace.drift_reason is None
    assert trace.before_png and trace.after_png and trace.drift_png is None
    assert trace.before_png != trace.after_png            # two distinct frames
    assert trace.clicked and trace.click_point == (40 + 224 / 2, 120 + 104 / 2)
    assert all(s.ok for s in trace.steps)
    assert [s.step for s in trace.steps] == [
        "wait_for_anchor", "capture_before", "locate_target", "check_label",
        "check_bounds", "click", "capture_after", "verify",
    ]


# === pure: every drift mode stops the run at the failed check ===============


def test_anchor_missing_drifts_before_any_action():
    driver = FakeDriver(anchor="Welcome Screen v2")   # UI changed
    trace = ProcedureRunner().run(driver, _procedure())
    assert trace.outcome is Outcome.DRIFT and trace.drift_reason is DriftReason.ANCHOR_MISSING
    assert trace.drift_png and trace.before_png is None   # stopped before 'before'
    assert driver.actions() == []                          # no click, ever


def test_target_missing_drifts_without_clicking():
    driver = FakeDriver()
    trace = ProcedureRunner().run(driver, _procedure(target_selector="#tile-lr-lamp"))
    assert trace.drift_reason is DriftReason.TARGET_MISSING
    assert driver.actions() == []


def test_label_mismatch_drifts_without_clicking():
    driver = FakeDriver(label="Bedroom Lamp Off")     # tile renamed / rearranged
    trace = ProcedureRunner().run(driver, _procedure())
    assert trace.drift_reason is DriftReason.TARGET_LABEL_MISMATCH
    assert driver.actions() == []


def test_moved_target_drifts_and_never_clicks_elsewhere():
    driver = FakeDriver(box=(500.0, 400.0, 224.0, 104.0))  # tile moved off-approval
    trace = ProcedureRunner().run(driver, _procedure())
    assert trace.drift_reason is DriftReason.TARGET_OUTSIDE_BOUNDS
    assert trace.clicked is False and driver.actions() == []  # the click is NOT re-aimed


def test_verification_failure_is_recorded_with_after_evidence():
    driver = FakeDriver(toggles=False)                 # click lands, state never changes
    trace = ProcedureRunner().run(driver, _procedure())
    assert trace.outcome is Outcome.DRIFT
    assert trace.drift_reason is DriftReason.VERIFICATION_FAILED
    assert trace.clicked is True and trace.after_png   # the failed state IS the evidence


def test_no_improvisation_after_drift():
    # After the failed check, the ONLY driver call is the drift screenshot:
    # no retry, no fallback locator, no second click.
    driver = FakeDriver(box=(500.0, 400.0, 224.0, 104.0))
    ProcedureRunner().run(driver, _procedure())
    idx = driver.calls.index(("find_box", TARGET_SEL))
    tail = driver.calls[idx + 1:]
    assert tail[-1] == "screenshot" and all(
        not (isinstance(c, tuple) and c[0] in ("click", "find_box")) for c in tail
    )


def test_unbounded_procedure_is_refused_before_the_driver_is_touched():
    driver = FakeDriver()
    with pytest.raises(ProcedureRefused):
        ProcedureRunner().run(driver, _procedure(verify_expected_text=""))
    with pytest.raises(ProcedureRefused):
        ProcedureRunner().run(driver, _procedure(anchor_text="  "))
    assert driver.calls == []                          # never touched


# === pure: the trace bundle ==================================================


def test_trace_bundle_carries_tier_procedure_and_verbatim_pixels(tmp_path):
    trace = ProcedureRunner().run(FakeDriver(), _procedure())
    out = build_trace_bundle(trace, tmp_path / "bundle")
    manifest = json.loads((out / "interface_trace_manifest.json").read_text())
    assert manifest["evidence_tier"] == TRACE_TIER == "interface_procedure_trace"
    assert "not vendor backend state" in manifest["evidence_tier_limits"]
    assert manifest["outcome"] == "completed"
    assert (out / "before.png").read_bytes() == trace.before_png   # verbatim
    assert (out / "after.png").read_bytes() == trace.after_png
    # every image rides with its unchanged Pixel Evidence v0 manifest
    for name in ("before.png", "after.png"):
        img = manifest["images"][name]
        assert img["evidence_tier"] == "pixel_capture"
        import hashlib
        assert img["image_sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    proc_doc = json.loads((out / "procedure.json").read_text())
    assert proc_doc["procedure_id"] == "proc-lamp-on-v0"           # the approved artifact, sealed


# === kernel: custody =========================================================


@pytest.fixture(scope="module")
def sealed_completed(tmp_path_factory):
    if not kernel_available():
        pytest.skip("kernel not available")
    work = tmp_path_factory.mktemp("trace")
    trace = ProcedureRunner().run(FakeDriver(), _procedure())
    shard = seal_trace(trace, work / "shard", work_dir=work)
    return trace, shard


@requires_kernel
def test_sealed_trace_verifies_with_out_of_band_key(sealed_completed):
    _trace, shard = sealed_completed
    assert shard.shard_id.startswith("sh1_") and shard.suite == "axm-hybrid1"
    assert verify_trace(shard.shard_dir, shard.trusted_key_path) is VerifyStatus.PASS


@requires_kernel
def test_wrong_key_fails_and_missing_key_refuses(sealed_completed, tmp_path):
    _trace, shard = sealed_completed
    subprocess.run(["axm-build", "keygen", str(tmp_path), "--name", "attacker"],
                   check=True, capture_output=True, text=True)
    assert verify_trace(shard.shard_dir, tmp_path / "attacker.pub") is VerifyStatus.FAIL
    assert verify_trace(shard.shard_dir, None) is VerifyStatus.NO_TRUSTED_KEY


@requires_kernel
def test_pixels_are_verbatim_inside_the_sealed_trace(sealed_completed):
    trace, shard = sealed_completed
    content = Path(shard.shard_dir) / "content"
    assert (content / "before.png").read_bytes() == trace.before_png
    assert (content / "after.png").read_bytes() == trace.after_png


@requires_kernel
def test_drift_trace_seals_too(tmp_path):
    # Drift is evidence, not an error: a verification-failed trace seals and
    # verifies exactly like a completed one.
    trace = ProcedureRunner().run(FakeDriver(toggles=False), _procedure())
    assert trace.drift_reason is DriftReason.VERIFICATION_FAILED
    shard = seal_trace(trace, tmp_path / "shard", work_dir=tmp_path)
    assert shard.outcome == "drift"
    assert verify_trace(shard.shard_dir, shard.trusted_key_path) is VerifyStatus.PASS
    manifest = json.loads((Path(shard.shard_dir) / "content" / "interface_trace_manifest.json").read_text())
    assert manifest["drift_reason"] == "verification_failed"


@requires_kernel
def test_detached_verification_without_browser_screenghost_or_ghostbox(sealed_completed):
    from core.pixel_exit_test import verify_detached   # reused, stdlib-only

    _trace, shard = sealed_completed
    res = verify_detached(shard.shard_dir, shard.trusted_key_path)
    assert res["status"] == "PASS" and res["exit_code"] == 0
    assert res["screenghost_involved"] is False
    assert res["ghostbox_involved"] is False
    assert res["browser_involved"] is False


# === browser: the full chain against real Chromium ==========================


@requires_browser
def test_real_chromium_completed_run(tmp_path):
    from core.interface_procedure import PlaywrightSurfaceDriver

    driver = PlaywrightSurfaceDriver(
        (FIXTURES / "dashboard.html").as_uri(), executable_path=CHROMIUM
    )
    try:
        trace = ProcedureRunner().run(driver, _procedure())
    finally:
        driver.close()
    assert trace.outcome is Outcome.COMPLETED
    assert trace.before_png[:8] == b"\x89PNG\r\n\x1a\n"      # real rendered pixels
    assert trace.after_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert trace.before_png != trace.after_png                # the surface visibly changed
    if kernel_available():
        shard = seal_trace(trace, tmp_path / "shard", work_dir=tmp_path)
        assert verify_trace(shard.shard_dir, shard.trusted_key_path) is VerifyStatus.PASS


@requires_browser
def test_real_chromium_stuck_tile_records_verification_drift():
    from core.interface_procedure import PlaywrightSurfaceDriver

    driver = PlaywrightSurfaceDriver(
        (FIXTURES / "dashboard_stuck.html").as_uri(), executable_path=CHROMIUM
    )
    try:
        trace = ProcedureRunner().run(driver, _procedure(verify_timeout_ms=400))
    finally:
        driver.close()
    assert trace.outcome is Outcome.DRIFT
    assert trace.drift_reason is DriftReason.VERIFICATION_FAILED
    assert trace.clicked is True and trace.after_png          # evidence of the dead control


@requires_browser
def test_real_chromium_moved_bounds_never_clicks():
    from core.interface_procedure import PlaywrightSurfaceDriver

    driver = PlaywrightSurfaceDriver(
        (FIXTURES / "dashboard.html").as_uri(), executable_path=CHROMIUM
    )
    try:
        trace = ProcedureRunner().run(
            driver, _procedure(approved_bounds=ApprovedBounds(x=500, y=400, width=100, height=80))
        )
        # the tile still says Off: no click happened anywhere
        assert driver.text(VERIFY_SEL).strip() == "Off"
    finally:
        driver.close()
    assert trace.drift_reason is DriftReason.TARGET_OUTSIDE_BOUNDS
    assert trace.clicked is False


# === boundaries ==============================================================


def test_no_ghostbox_ocr_or_vision_in_the_procedure_path():
    code = (
        "import importlib, sys\n"
        "importlib.import_module('core.interface_procedure')\n"
        "importlib.import_module('core.procedure_seal')\n"
        "bad=[m for m in ('ghostbox','pytesseract','PIL','cv2','torch','easyocr',"
        "'transformers','xml.etree.ElementTree','playwright') "
        "if any(k==m or k.startswith(m+'.') for k in sys.modules)]\n"
        "print('BAD:'+','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    # playwright itself must be lazy: importing the modules pulls in NO browser stack
    assert [l for l in out.stdout.splitlines() if l.startswith("BAD:")][0] == "BAD:"


def test_procedure_source_has_no_forbidden_imports():
    from core import interface_procedure, procedure_seal

    for mod in (interface_procedure, procedure_seal):
        src = inspect.getsource(mod)
        for token in ("import ghostbox", "from ghostbox", "import pytesseract",
                      "from PIL", "import cv2", "import torch", "core.adapter",
                      "core.texttree", "core.capture", "core.ingest"):
            assert token not in src, f"{mod.__name__} must not use {token!r}"
