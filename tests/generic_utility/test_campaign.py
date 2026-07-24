from __future__ import annotations

import json

from experiments.generic_utility.metrics import verify_campaign_bundle


def test_full_emulated_campaign_passes_all_premise_gates(campaign_dir):
    gates = json.loads((campaign_dir / "gate_results.json").read_text())
    assert gates and all(row["passed"] for row in gates.values())
    receipt = json.loads((campaign_dir / "campaign_receipt.json").read_text())
    assert receipt["all_gates_passed"] is True


def test_campaign_bundle_manifest_verifies(campaign_dir):
    result = verify_campaign_bundle(campaign_dir)
    assert result["ok"] is True


def test_warm_tasks_use_no_teacher_or_large_vlm(campaign_dir):
    rows = json.loads((campaign_dir / "task_receipts.json").read_text())["tasks"]
    warm = [row for row in rows if row["phase"] == "B_warm_replay"]
    assert warm and all(row["success"] for row in warm)
    assert all(row["metrics"]["teacher_reads"] == 0 for row in warm)
    assert all(row["metrics"]["large_vlm_calls"] == 0 for row in warm)
