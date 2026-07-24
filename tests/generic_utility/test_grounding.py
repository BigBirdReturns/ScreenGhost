from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.generic_utility.grounding_benchmark import (
    build_phoneworld_grounding_suite,
    run_grounding_benchmark,
)
from experiments.generic_utility.model_runtime import AttachedJsonModelProvider


WORKER = Path(__file__).resolve().parents[2] / "experiments" / "generic_utility" / "model_workers" / "fake_grounder_worker.py"


def test_grounding_suite_has_multiple_apps_and_variants(tmp_path):
    cases = build_phoneworld_grounding_suite(tmp_path / "suite")
    assert len(cases) >= 10
    assert {case.app_family for case in cases} >= {"settings", "profile", "timer", "connectivity"}
    assert any(case.variant["theme"] == "dark" for case in cases)


def test_oracle_worker_proves_benchmark_plumbing(tmp_path):
    cases = build_phoneworld_grounding_suite(tmp_path / "suite")
    with AttachedJsonModelProvider(
        "oracle-fixture",
        [sys.executable, str(WORKER), "--mode", "oracle"],
        startup_timeout_ms=5000,
    ) as provider:
        out = run_grounding_benchmark(
            provider,
            cases,
            tmp_path / "result",
            emulated_oracle_payload=True,
        )
    receipt = json.loads((out / "benchmark_receipt.json").read_text())
    assert receipt["hit_rate"] == 1.0
    assert receipt["teacher_answers_visible_to_provider"] is True
    assert receipt["metric_kind"] == "simulated"
    assert receipt["process_timing_kind"] == "measured"
    assert receipt["evidence_classification"] == "emulated_oracle_protocol_validation"
    assert receipt["motor_calls"] == 0


def test_non_oracle_center_worker_is_scored_honestly(tmp_path):
    cases = build_phoneworld_grounding_suite(tmp_path / "suite")[:2]
    with AttachedJsonModelProvider(
        "center",
        [sys.executable, str(WORKER), "--mode", "echo"],
        startup_timeout_ms=5000,
    ) as provider:
        out = run_grounding_benchmark(provider, cases, tmp_path / "result")
    receipt = json.loads((out / "benchmark_receipt.json").read_text())
    assert 0.0 <= receipt["hit_rate"] <= 1.0
    assert receipt["teacher_answers_visible_to_provider"] is False
    assert receipt["metric_kind"] == "measured"
    assert receipt["evidence_classification"] == "teacher_hidden_student_measurement"


def test_suite_manifest_hides_raw_bounds(tmp_path):
    build_phoneworld_grounding_suite(tmp_path / "suite")
    manifest = json.loads((tmp_path / "suite" / "suite_manifest.json").read_text())
    assert all("expected_bounds" not in row for row in manifest["cases"])
    assert all("hidden_answer_sha256" in row for row in manifest["cases"])
