from __future__ import annotations

import json
from pathlib import Path

from experiments.generic_utility.cli import parser
from experiments.generic_utility.conclusion import assemble_conclusion
from experiments.generic_utility.doctor import capability_report


def test_cli_exposes_complete_campaign_surfaces():
    root = parser()
    commands = root._subparsers._group_actions[0].choices  # argparse has no public choices API
    assert {
        "doctor",
        "emulate",
        "verify",
        "browser-smoke",
        "grounding-emulated",
        "grounding-local",
        "androidworld-smoke",
        "physical-smoke",
        "conclude",
    } <= set(commands)


def test_doctor_is_side_effect_free_capability_report():
    report = capability_report()
    assert report["schema"] == "screenghost_generic_utility_doctor_v1"
    assert isinstance(report["emulated_campaign_ready"], bool)
    assert "gpu" in report and "commands" in report


def test_oracle_grounder_cannot_satisfy_production_model_gate(campaign_dir, tmp_path):
    grounding = tmp_path / "oracle.json"
    grounding.write_text(
        json.dumps(
            {
                "schema": "screenghost_grounding_benchmark_v1",
                "metric_kind": "simulated",
                "process_timing_kind": "measured",
                "evidence_classification": "emulated_oracle_protocol_validation",
                "teacher_answers_visible_to_provider": True,
                "completed": 11,
            }
        ),
        encoding="utf-8",
    )
    out = assemble_conclusion(
        tmp_path / "conclusion.json",
        campaign_dir=campaign_dir,
        grounding_receipt=grounding,
    )
    receipt = json.loads(out.read_text())
    assert receipt["premise_conclusion_ready"] is True
    assert receipt["measured_local_grounder"] == "not_run"
    assert receipt["production_claim_ready"] is False


def test_teacher_hidden_measured_grounder_is_recognized_but_transport_still_required(campaign_dir, tmp_path):
    grounding = tmp_path / "student.json"
    grounding.write_text(
        json.dumps(
            {
                "schema": "screenghost_grounding_benchmark_v1",
                "metric_kind": "measured",
                "evidence_classification": "teacher_hidden_student_measurement",
                "teacher_answers_visible_to_provider": False,
                "completed": 11,
            }
        ),
        encoding="utf-8",
    )
    out = assemble_conclusion(
        tmp_path / "conclusion.json",
        campaign_dir=campaign_dir,
        grounding_receipt=grounding,
    )
    receipt = json.loads(out.read_text())
    assert receipt["measured_local_grounder"] == "pass"
    assert receipt["production_claim_ready"] is False
