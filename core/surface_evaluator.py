"""Hidden-teacher evaluation for pixel-only ScreenGhost predictions.

The teacher plane may know the structural answer while the evaluated runtime may
only see pixels and declared memory.  This module scores the runtime output
without feeding the answer back into action selection.  It reports screen-family,
element, grounding, and state metrics together with an evidence-source leakage
check.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

EVALUATION_SCHEMA = "surface_teacher_evaluation_v1"
DEFAULT_FORBIDDEN_RUNTIME_SOURCES = (
    "android_uiautomator",
    "accessibility_tree",
    "web_dom",
    "cdp_domsnapshot",
    "teacher_lesson",
)


class EvaluationRefused(ValueError):
    """Truth or prediction data is malformed."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact.casefold() if compact else None


def _bounds(value: Sequence[Any]) -> Tuple[float, float, float, float]:
    if len(value) != 4:
        raise EvaluationRefused(f"bounds require four values, got {value!r}")
    vals = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vals):
        raise EvaluationRefused(f"non-finite bounds: {vals!r}")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise EvaluationRefused(f"empty or inverted bounds: {vals!r}")
    return vals


def _iou(left: Sequence[Any], right: Sequence[Any]) -> float:
    ax1, ay1, ax2, ay2 = _bounds(left)
    bx1, by1, bx2, by2 = _bounds(right)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


def _point_inside(point: Sequence[Any], bounds: Sequence[Any]) -> bool:
    if len(point) != 2:
        return False
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = _bounds(bounds)
    return x1 <= x <= x2 and y1 <= y <= y2


def _safe_elements(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    elements = projection.get("elements")
    if not isinstance(elements, list):
        raise EvaluationRefused("teacher projection has no elements list")
    out = []
    for element in elements:
        if not isinstance(element, Mapping):
            raise EvaluationRefused("teacher element is not an object")
        if bool(element.get("sensitive", False)):
            continue
        row = dict(element)
        row["normalized_bounds"] = list(_bounds(row.get("normalized_bounds") or ()))
        row["role"] = str(row.get("role") or "unknown")
        row["label"] = _clean(row.get("label"))
        states = row.get("states") or {}
        row["states"] = {str(k): str(v) for k, v in states.items()} if isinstance(states, Mapping) else {}
        out.append(row)
    return out


def _prediction_elements(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    elements = prediction.get("elements")
    if not isinstance(elements, list):
        raise EvaluationRefused("prediction has no elements list")
    out = []
    for element in elements:
        if not isinstance(element, Mapping):
            raise EvaluationRefused("predicted element is not an object")
        row = dict(element)
        raw_bounds = row.get("normalized_bounds") or row.get("box")
        row["normalized_bounds"] = list(_bounds(raw_bounds)) if raw_bounds is not None else None
        raw_point = row.get("point")
        row["point"] = list(raw_point) if isinstance(raw_point, Sequence) and len(raw_point) == 2 else None
        row["role"] = str(row.get("role") or "unknown")
        row["label"] = _clean(row.get("label"))
        states = row.get("states") or {}
        row["states"] = {str(k): str(v) for k, v in states.items()} if isinstance(states, Mapping) else {}
        out.append(row)
    return out


@dataclass(frozen=True)
class ElementMatch:
    truth_index: int
    prediction_index: int
    role_match: bool
    label_match: bool
    iou: float
    point_hit: bool
    state_accuracy: Optional[float]


@dataclass(frozen=True)
class EvaluationReceipt:
    schema: str
    receipt_id: str
    lesson_id: Optional[str]
    screen_key_expected: Optional[str]
    screen_key_predicted: Optional[str]
    screen_match: bool
    truth_elements: int
    predicted_elements: int
    matched_elements: int
    element_precision: float
    element_recall: float
    role_accuracy: float
    label_accuracy: float
    grounding_accuracy: float
    state_accuracy: Optional[float]
    privileged_leakage: bool
    forbidden_sources_seen: Tuple[str, ...]
    overall_score: float
    matches: Tuple[ElementMatch, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["forbidden_sources_seen"] = list(self.forbidden_sources_seen)
        value["matches"] = [asdict(match) for match in self.matches]
        return value


def evaluate_prediction(
    teacher_projection: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    forbidden_runtime_sources: Sequence[str] = DEFAULT_FORBIDDEN_RUNTIME_SOURCES,
    grounding_iou_threshold: float = 0.5,
) -> EvaluationReceipt:
    if not 0 <= grounding_iou_threshold <= 1:
        raise EvaluationRefused("grounding_iou_threshold must be in [0,1]")
    truth = _safe_elements(teacher_projection)
    predicted = _prediction_elements(prediction)

    candidates: list[tuple[float, int, int, float, bool]] = []
    for truth_index, expected in enumerate(truth):
        for prediction_index, actual in enumerate(predicted):
            role_match = _clean(expected["role"]) == _clean(actual["role"])
            label_match = bool(expected["label"]) and expected["label"] == actual["label"]
            overlap = 0.0
            if actual["normalized_bounds"] is not None:
                overlap = _iou(expected["normalized_bounds"], actual["normalized_bounds"])
            point_hit = actual["point"] is not None and _point_inside(actual["point"], expected["normalized_bounds"])
            score = (2.0 if role_match else 0.0) + (3.0 if label_match else 0.0) + 2.0 * overlap + (1.0 if point_hit else 0.0)
            if score > 0:
                candidates.append((score, truth_index, prediction_index, overlap, point_hit))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))

    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    matches: list[ElementMatch] = []
    for _score, truth_index, prediction_index, overlap, point_hit in candidates:
        if truth_index in used_truth or prediction_index in used_prediction:
            continue
        expected = truth[truth_index]
        actual = predicted[prediction_index]
        role_match = _clean(expected["role"]) == _clean(actual["role"])
        label_match = expected["label"] == actual["label"] if expected["label"] else actual["label"] is None
        state_scores = [
            1.0 if actual["states"].get(key) == value else 0.0
            for key, value in expected["states"].items()
            if key in actual["states"]
        ]
        state_accuracy = sum(state_scores) / len(state_scores) if state_scores else None
        matches.append(
            ElementMatch(
                truth_index=truth_index,
                prediction_index=prediction_index,
                role_match=role_match,
                label_match=label_match,
                iou=round(overlap, 6),
                point_hit=bool(point_hit),
                state_accuracy=(round(state_accuracy, 6) if state_accuracy is not None else None),
            )
        )
        used_truth.add(truth_index)
        used_prediction.add(prediction_index)

    matched = len(matches)
    precision = matched / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = matched / len(truth) if truth else 1.0
    role_accuracy = sum(1 for match in matches if match.role_match) / matched if matched else 0.0
    label_accuracy = sum(1 for match in matches if match.label_match) / matched if matched else 0.0
    grounding_accuracy = (
        sum(1 for match in matches if match.point_hit or match.iou >= grounding_iou_threshold) / matched
        if matched
        else 0.0
    )
    state_values = [match.state_accuracy for match in matches if match.state_accuracy is not None]
    state_accuracy = sum(state_values) / len(state_values) if state_values else None

    expected_screen = teacher_projection.get("screen_key")
    predicted_screen = prediction.get("screen_key")
    screen_match = expected_screen is not None and str(expected_screen) == str(predicted_screen)
    sources = prediction.get("evidence_sources") or []
    if isinstance(sources, str):
        sources = [sources]
    normalized_sources = {_clean(source) for source in sources}
    forbidden = {_clean(source) for source in forbidden_runtime_sources}
    seen = tuple(sorted(source for source in normalized_sources if source and source in forbidden))
    leakage = bool(seen)

    state_component = state_accuracy if state_accuracy is not None else 1.0
    overall = (
        0.20 * float(screen_match)
        + 0.20 * recall
        + 0.10 * precision
        + 0.15 * role_accuracy
        + 0.15 * label_accuracy
        + 0.15 * grounding_accuracy
        + 0.05 * state_component
    )
    if leakage:
        overall = 0.0

    payload = {
        "schema": EVALUATION_SCHEMA,
        "lesson_id": teacher_projection.get("lesson_id"),
        "screen_key_expected": expected_screen,
        "screen_key_predicted": predicted_screen,
        "screen_match": screen_match,
        "truth_elements": len(truth),
        "predicted_elements": len(predicted),
        "matched_elements": matched,
        "metrics": {
            "element_precision": precision,
            "element_recall": recall,
            "role_accuracy": role_accuracy,
            "label_accuracy": label_accuracy,
            "grounding_accuracy": grounding_accuracy,
            "state_accuracy": state_accuracy,
        },
        "forbidden_sources_seen": list(seen),
        "overall_score": overall,
        "matches": [asdict(match) for match in matches],
    }
    return EvaluationReceipt(
        schema=EVALUATION_SCHEMA,
        receipt_id="evaluation_" + _sha256(_json_bytes(payload)),
        lesson_id=(str(teacher_projection["lesson_id"]) if teacher_projection.get("lesson_id") else None),
        screen_key_expected=(str(expected_screen) if expected_screen is not None else None),
        screen_key_predicted=(str(predicted_screen) if predicted_screen is not None else None),
        screen_match=screen_match,
        truth_elements=len(truth),
        predicted_elements=len(predicted),
        matched_elements=matched,
        element_precision=round(precision, 6),
        element_recall=round(recall, 6),
        role_accuracy=round(role_accuracy, 6),
        label_accuracy=round(label_accuracy, 6),
        grounding_accuracy=round(grounding_accuracy, 6),
        state_accuracy=(round(state_accuracy, 6) if state_accuracy is not None else None),
        privileged_leakage=leakage,
        forbidden_sources_seen=seen,
        overall_score=round(overall, 6),
        matches=tuple(matches),
    )
