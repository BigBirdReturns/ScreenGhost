"""The pixel-evidence exit property: the sealed visual record survives ShareX,
ScreenGhost, GhostBox, and the browser.

Verify the sealed shard using ONLY the shard bytes plus the out-of-band public
key, through the genesis verifier CLI. This module imports NOTHING from
ScreenGhost ``core`` and nothing from ``ghostbox``, and touches no capture tool or
browser -- it is the proof that the visual record's verifiability does not depend
on any of them.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def verify_detached(shard_dir: str | Path, trusted_key: str | Path, axm_verify: str = "axm-verify") -> Dict[str, Any]:
    """Genesis-verify the sealed pixel shard with only its bytes + the oob pub.

    The returned receipt records that no capture tool, ScreenGhost runtime,
    GhostBox, or browser was in the verification path.
    """
    proc = subprocess.run(
        [axm_verify, "shard", str(shard_dir), "--trusted-key", str(trusted_key)],
        capture_output=True,
        text=True,
    )
    result: Dict[str, Any] = {}
    body = proc.stdout.strip()
    if body:
        try:
            result = json.loads(body.splitlines()[-1])
        except json.JSONDecodeError:
            result = {"raw_stdout": proc.stdout, "raw_stderr": proc.stderr}
    return {
        "exit_code": proc.returncode,
        "status": result.get("status"),
        "sharex_involved": False,
        "screenghost_involved": False,
        "ghostbox_involved": False,
        "browser_involved": False,
        "genesis_result": result,
    }
