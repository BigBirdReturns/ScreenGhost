"""Teacher-hidden GUI grounding benchmark for local 2B-class students."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from experiments.generic_utility.model_runtime import ModelProvider, ModelRequest, ModelReceipt
from experiments.generic_utility.phone_world import DisplayVariant, PhoneWorld
from experiments.generic_utility.schema import MetricKind, json_bytes, sha256_bytes, sha256_json


BENCHMARK_SCHEMA = "screenghost_grounding_benchmark_v1"


@dataclass(frozen=True)
class GroundingCase:
    case_id: str
    image_path: str
    image_sha256: str
    instruction: str
    app_family: str
    screen_name: str
    target_role: str
    target_label: Optional[str]
    expected_bounds: tuple[float, float, float, float]
    variant: Mapping[str, Any]

    @property
    def expected_point(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.expected_bounds
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def public_dict(self) -> dict[str, Any]:
        """Student-visible case fields, excluding the hidden answer."""
        return {
            "case_id": self.case_id,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "instruction": self.instruction,
            "app_family": self.app_family,
            "screen_name": self.screen_name,
            "variant": dict(self.variant),
        }


@dataclass(frozen=True)
class GroundingResult:
    case_id: str
    status: str
    predicted_point: Optional[tuple[float, float]]
    hit: bool
    normalized_center_error: Optional[float]
    model_receipt: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["predicted_point"] = list(self.predicted_point) if self.predicted_point else None
        value["model_receipt"] = dict(self.model_receipt)
        return value


def _point(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if max(abs(x), abs(y)) > 1.5:
        x, y = x / 1000.0, y / 1000.0
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return x, y


def _inside(point: tuple[float, float], bounds: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def _prompt(case: GroundingCase) -> str:
    return (
        "You are grounding one control on a phone screenshot. "
        f"Instruction: {case.instruction}. Return only the center point as JSON "
        '{"x": <0-1000>, "y": <0-1000>}. Do not describe the screen.'
    )


def build_phoneworld_grounding_suite(out_dir: str | Path) -> tuple[GroundingCase, ...]:
    out = Path(out_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    world = PhoneWorld(seed=41)
    specifications = [
        ("settings-home-display", "settings", "home", "button", "Display", DisplayVariant()),
        ("settings-display-dark", "settings", "display", "switch", "Dark mode", DisplayVariant()),
        ("settings-display-save", "settings", "display", "button", "Save", DisplayVariant()),
        ("profile-home-edit", "profile", "home", "button", "Edit profile", DisplayVariant()),
        ("profile-edit-name", "profile", "edit", "text_field", "Display name", DisplayVariant()),
        ("profile-edit-save", "profile", "edit", "button", "Save profile", DisplayVariant()),
        ("timer-home-start", "timer", "home", "button", "Start", DisplayVariant()),
        ("timer-running-stop", "timer", "running", "button", "Stop", DisplayVariant()),
        ("holdout-connectivity", "connectivity", "home", "switch", None, DisplayVariant()),
        ("settings-dark-theme", "settings", "display", "switch", "Dark mode", DisplayVariant(theme="dark", variant_id="dark")),
        ("settings-font-scale", "settings", "display", "button", "Save", DisplayVariant(font_scale=1.15, variant_id="font-115")),
    ]
    cases: list[GroundingCase] = []
    for case_id, app, screen, role, label, variant in specifications:
        world.reset(app_family=app, screen_name=screen)
        world.set_variant(variant)
        if app == "timer" and screen == "running":
            world._timer_running = True
        frame = world.teacher_snapshot()
        candidates = [
            node
            for node in frame.nodes
            if node.interactive
            and node.role == role
            and (label is None or (node.label or "").casefold() == label.casefold())
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"grounding fixture {case_id} resolved {len(candidates)} candidates")
        target = candidates[0]
        path = images / f"{case_id}.png"
        path.write_bytes(frame.png_bytes)
        instruction = f"Activate the {label}" if label else f"Activate the only {role} control"
        cases.append(
            GroundingCase(
                case_id=case_id,
                image_path=str(path),
                image_sha256=sha256_bytes(frame.png_bytes),
                instruction=instruction,
                app_family=app,
                screen_name=screen,
                target_role=role,
                target_label=label,
                expected_bounds=target.normalized(frame.width, frame.height),
                variant=variant.__dict__,
            )
        )
    manifest = {
        "schema": BENCHMARK_SCHEMA,
        "evidence_classification": "emulated_phone_pixels_with_hidden_teacher_answers",
        "cases": [
            {
                **case.public_dict(),
                "hidden_answer_sha256": sha256_json(
                    {"bounds": list(case.expected_bounds), "role": case.target_role, "label": case.target_label}
                ),
            }
            for case in cases
        ],
    }
    (out / "suite_manifest.json").write_bytes(json_bytes(manifest))
    return tuple(cases)


def run_grounding_benchmark(
    provider: ModelProvider,
    cases: Iterable[GroundingCase],
    out_dir: str | Path,
    *,
    timeout_ms: float = 20000.0,
    emulated_oracle_payload: bool = False,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[GroundingResult] = []
    receipts: list[ModelReceipt] = []
    for case in cases:
        payload: dict[str, Any] = {
            "image_path": case.image_path,
            "image_sha256": case.image_sha256,
            "instruction": case.instruction,
            "prompt": _prompt(case),
            "input_resolution": "phoneworld",
        }
        if emulated_oracle_payload:
            payload["fixture_point"] = list(case.expected_point)
        receipt = provider.run(ModelRequest.create("gui_grounding", payload, timeout_ms=timeout_ms))
        receipts.append(receipt)
        point = _point((receipt.result or {}).get("point")) if receipt.status == "done" else None
        center_error = None
        hit = False
        if point is not None:
            expected = case.expected_point
            center_error = math.dist(point, expected)
            hit = _inside(point, case.expected_bounds)
        results.append(
            GroundingResult(
                case_id=case.case_id,
                status=receipt.status,
                predicted_point=point,
                hit=hit,
                normalized_center_error=center_error,
                model_receipt=receipt.to_dict(),
            )
        )
    hits = sum(result.hit for result in results)
    completed = sum(result.status == "done" for result in results)
    measured_process = any((receipt.metric_kind == MetricKind.MEASURED.value) for receipt in receipts)
    evidence_classification = (
        "emulated_oracle_protocol_validation"
        if emulated_oracle_payload
        else "teacher_hidden_student_measurement"
    )
    summary = {
        "schema": BENCHMARK_SCHEMA,
        "provider": getattr(provider, "name", type(provider).__name__),
        # A real child process was timed in both modes, but an oracle that receives
        # the answer is still simulated benchmark evidence. Keep runtime timing and
        # benchmark validity separate so it can never satisfy a measured-model gate.
        "metric_kind": (
            MetricKind.SIMULATED.value
            if emulated_oracle_payload
            else MetricKind.MEASURED.value if measured_process else MetricKind.SIMULATED.value
        ),
        "process_timing_kind": (
            MetricKind.MEASURED.value if measured_process else MetricKind.SIMULATED.value
        ),
        "evidence_classification": evidence_classification,
        "case_count": len(results),
        "completed": completed,
        "hits": hits,
        "hit_rate": hits / len(results) if results else 0.0,
        "mean_center_error": (
            sum(result.normalized_center_error for result in results if result.normalized_center_error is not None)
            / max(1, sum(result.normalized_center_error is not None for result in results))
        ),
        "motor_calls": 0,
        "teacher_answers_visible_to_provider": bool(emulated_oracle_payload),
        "results": [result.to_dict() for result in results],
    }
    (out / "benchmark_receipt.json").write_bytes(json_bytes(summary))
    return out
