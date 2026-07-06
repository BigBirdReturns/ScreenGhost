"""ScreenGhost Interface Procedure v0 — end-to-end demo.

    approved procedure (anchor, target, approved bounds, verification)
      -> real Chromium renders the local fixture dashboard
      -> wait for anchor -> BEFORE pixels -> locate + label + bounds check
      -> ONE click inside approved bounds -> visible state Off -> On
      -> AFTER pixels -> verify -> seal the whole trace through genesis
      -> verify with an out-of-band key -> detached exit test.

The surface is a deterministic LOCAL fixture (file://), labeled as such -- not a
vendor app. Same runner, same procedure shape, later targets an emulator or a
vendor dashboard. Run `--drift` to see the drift discipline instead: a dead tile
records verification_failed, stops, and the drift trace seals as evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interface_procedure import (
    ApprovedBounds,
    InterfaceProcedure,
    PlaywrightSurfaceDriver,
    ProcedureRunner,
)
from core.pixel_exit_test import verify_detached
from core.pixel_seal import kernel_available
from core.procedure_seal import seal_trace, verify_trace

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "interface_surface"
DEFAULT_CHROMIUM = "/opt/pw-browsers/chromium"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run ScreenGhost Interface Procedure v0.")
    ap.add_argument("--drift", action="store_true", help="run against the dead-tile fixture instead")
    ap.add_argument("--chromium", default=os.environ.get("SCREENGHOST_CHROMIUM", DEFAULT_CHROMIUM))
    ap.add_argument("--out", default="interface_procedure_out")
    args = ap.parse_args(argv)

    fixture = FIXTURES / ("dashboard_stuck.html" if args.drift else "dashboard.html")
    procedure = InterfaceProcedure(
        procedure_id="proc-lamp-on-v0",
        surface_label="local-fixture-surface",
        anchor_selector="#dashboard-title",
        anchor_text="Home Dashboard",
        target_selector="#tile-living-room-lamp",
        target_label="Living Room Lamp",
        approved_bounds=ApprovedBounds(x=20, y=100, width=300, height=160),
        verify_selector="#lamp-state",
        verify_expected_text="On",
        verify_timeout_ms=1000,
    )

    exe = args.chromium if Path(args.chromium).exists() else None
    driver = PlaywrightSurfaceDriver(fixture.as_uri(), executable_path=exe)
    try:
        trace = ProcedureRunner().run(driver, procedure)
    finally:
        driver.close()

    receipt = {
        "artifact": "ScreenGhost Interface Procedure v0",
        "surface": procedure.surface_label + " (LOCAL fixture, not a vendor app)",
        "procedure_id": procedure.procedure_id,
        "outcome": trace.outcome.value,
        "drift_reason": trace.drift_reason.value if trace.drift_reason else None,
        "clicked": trace.clicked,
        "click_point": trace.click_point,
        "steps": [f"{s.step}: {'ok' if s.ok else 'FAILED'} — {s.detail}" for s in trace.steps],
        "evidence": {
            "before_png_bytes": len(trace.before_png or b""),
            "after_png_bytes": len(trace.after_png or b""),
            "drift_png_bytes": len(trace.drift_png or b""),
        },
    }

    if kernel_available():
        work = Path(tempfile.mkdtemp(prefix="iface_trace_"))
        shard = seal_trace(trace, Path(args.out) / "shard", work_dir=work)
        status = verify_trace(shard.shard_dir, shard.trusted_key_path)
        detached = verify_detached(shard.shard_dir, shard.trusted_key_path)
        receipt.update(
            sealed=True,
            shard_id=shard.shard_id,
            suite=shard.suite,
            verification=status.value,
            detached={
                "status": detached["status"],
                "screenghost_involved": detached["screenghost_involved"],
                "ghostbox_involved": detached["ghostbox_involved"],
                "browser_involved": detached["browser_involved"],
            },
        )
        ok = status.value == "pass" and detached["status"] == "PASS"
    else:
        receipt.update(sealed=False, note="genesis kernel not on PATH; trace captured but not sealed")
        ok = True

    print(json.dumps(receipt, indent=2))
    print(f"[interface procedure v0: {'OK' if ok else 'INCOMPLETE'} — outcome={trace.outcome.value}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
