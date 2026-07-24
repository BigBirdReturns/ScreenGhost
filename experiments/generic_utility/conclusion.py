"""Aggregate receipts into the explicit conclusion boundary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from experiments.generic_utility.metrics import verify_campaign_bundle
from experiments.generic_utility.schema import json_bytes, sha256_json


def _load(path: Optional[str | Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        for name in ("receipt.json", "benchmark_receipt.json", "campaign_receipt.json"):
            if (candidate / name).exists():
                candidate = candidate / name
                break
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def assemble_conclusion(
    out_path: str | Path,
    *,
    campaign_dir: str | Path,
    browser_receipt: Optional[str | Path] = None,
    grounding_receipt: Optional[str | Path] = None,
    androidworld_receipt: Optional[str | Path] = None,
    physical_receipt: Optional[str | Path] = None,
) -> Path:
    campaign = Path(campaign_dir)
    integrity = verify_campaign_bundle(campaign)
    campaign_receipt = _load(campaign / "campaign_receipt.json") or {}
    gates = _load(campaign / "gate_results.json") or {}
    browser = _load(browser_receipt)
    grounding = _load(grounding_receipt)
    androidworld = _load(androidworld_receipt)
    physical = _load(physical_receipt)
    premise_ready = bool(
        integrity.get("ok")
        and campaign_receipt.get("all_gates_passed")
        and gates
        and all(bool(row.get("passed")) for row in gates.values())
    )
    measured_grounder = bool(
        grounding
        and grounding.get("metric_kind") == "measured"
        and grounding.get("evidence_classification") == "teacher_hidden_student_measurement"
        and grounding.get("teacher_answers_visible_to_provider") is False
    )
    live_transport = bool(
        (androidworld and androidworld.get("status") == "pass")
        or (physical and physical.get("status") == "pass")
    )
    result = {
        "schema": "screenghost_generic_utility_conclusion_v1",
        "conclusion_id": "conclusion_" + sha256_json(
            {
                "campaign": campaign_receipt.get("campaign_id"),
                "browser": browser,
                "grounding": grounding,
                "androidworld": androidworld,
                "physical": physical,
            }
        ),
        "original_premise": (
            "a phone becomes a generic utility after expensive first teaching; "
            "known operation should become cheaper, calmer, teacher-blind, and transferable"
        ),
        "emulated_premise_proof": "pass" if premise_ready else "fail",
        "browser_geometry_and_dynamic_surface": (
            "pass" if browser and browser.get("dpr_geometry_pass") and browser.get("animated_surface_pass") else "not_run"
        ),
        "measured_local_grounder": "pass" if measured_grounder and grounding.get("completed", 0) > 0 else "not_run",
        "androidworld_transport": "pass" if androidworld and androidworld.get("status") == "pass" else "not_run",
        "physical_adb_transport": "pass" if physical and physical.get("status") == "pass" else "not_run",
        "premise_conclusion_ready": premise_ready,
        "physical_machine_is_final_smoke": premise_ready,
        "production_claim_ready": premise_ready and measured_grounder and live_transport,
        "claim_boundary": (
            "emulated receipts establish system behavior and amortization; measured model and live transport receipts "
            "establish hardware-specific latency, VRAM, and device compatibility"
        ),
        "campaign_integrity": integrity,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(json_bytes(result))
    return out
