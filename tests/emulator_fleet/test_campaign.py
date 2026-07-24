import json
from pathlib import Path

from experiments.emulator_fleet.campaign import (
    run_semantic_multibox_campaign,
    verify_bundle,
)


def test_full_semantic_multibox_campaign(tmp_path: Path):
    out = run_semantic_multibox_campaign(tmp_path / "campaign")
    conclusion = json.loads((out / "CONCLUSION.json").read_text())
    comparison = json.loads((out / "comparison.json").read_text())
    assert conclusion["status"] == "PASS"
    assert comparison["metrics"]["semantic_advantages"] >= 3
    assert comparison["metrics"]["runtime_teacher_reads"] == 0
    assert comparison["metrics"]["large_model_calls"] == 0
    assert all(comparison["gates"].values())
    assert verify_bundle(out)["ok"]


def test_campaign_tamper_gate(tmp_path: Path):
    out = run_semantic_multibox_campaign(tmp_path / "campaign")
    (out / "comparison.json").write_text("{}", encoding="utf-8")
    result = verify_bundle(out)
    assert not result["ok"] and result["mismatches"]
