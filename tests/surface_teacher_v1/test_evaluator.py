from __future__ import annotations

from core.surface_evaluator import evaluate_prediction
from tests.surface_teacher_v1.support import runtime_projection


def prediction_from_truth(truth):
    return {
        "screen_key": truth["screen_key"],
        "evidence_sources": ["pixels", "surface_atlas"],
        "elements": [
            {
                "role": element["role"],
                "label": element["label"],
                "normalized_bounds": element["normalized_bounds"],
                "states": element["states"],
            }
            for element in truth["elements"]
            if not element["sensitive"]
        ],
    }


def test_perfect_teacher_blind_prediction_scores_one():
    truth = runtime_projection("settings")
    receipt = evaluate_prediction(truth, prediction_from_truth(truth))
    assert receipt.screen_match
    assert receipt.privileged_leakage is False
    assert receipt.overall_score == 1.0
    assert receipt.grounding_accuracy == 1.0


def test_privileged_runtime_source_zeroes_score():
    truth = runtime_projection("settings")
    prediction = prediction_from_truth(truth)
    prediction["evidence_sources"].append("android_uiautomator")
    receipt = evaluate_prediction(truth, prediction)
    assert receipt.privileged_leakage
    assert receipt.overall_score == 0.0
    assert receipt.forbidden_sources_seen == ("android_uiautomator",)


def test_partial_prediction_reports_recall_without_inventing_matches():
    truth = runtime_projection("settings")
    prediction = prediction_from_truth(truth)
    prediction["elements"] = prediction["elements"][:1]
    receipt = evaluate_prediction(truth, prediction)
    assert receipt.matched_elements == 1
    assert receipt.element_precision == 1.0
    assert 0 < receipt.element_recall < 1
