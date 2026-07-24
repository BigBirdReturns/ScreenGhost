#!/usr/bin/env python3
"""Offline proof of the Surface Teacher v1 research graft."""
from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from core.surface_alignment import AlignmentNode, FrameObservation, certify_alignment, stage_alignment
from core.surface_curriculum import build_curriculum, stage_curriculum
from core.surface_evaluator import evaluate_prediction
from core.surface_graph import ActionDescriptor, SurfaceTransitionGraph, make_transition
from core.surface_perception import PerceptionRequest, route_perception


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def frame(clock: int, checked: bool = False) -> bytes:
    image = Image.new("RGB", (360, 720), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 359, 64), fill=(232, 232, 232))
    draw.text((18, 22), "Settings", fill=(0, 0, 0))
    draw.text((300, 22), f"{clock:02d}", fill=(0, 0, 0))
    draw.text((24, 150), "Dark mode", fill=(0, 0, 0))
    draw.rounded_rectangle((275, 136, 335, 176), radius=18, fill=(45, 125, 255) if checked else (155, 155, 155))
    knob_x = 315 if checked else 295
    draw.ellipse((knob_x - 14, 142, knob_x + 14, 170), fill=(255, 255, 255))
    draw.rounded_rectangle((210, 610, 330, 668), radius=8, outline=(0, 0, 0), width=2)
    draw.text((246, 630), "Save", fill=(0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def nodes(checked: bool = False):
    return (
        AlignmentNode("title", "heading", (0, 0, 250, 64), label="Settings"),
        AlignmentNode("clock", "text", (290, 8, 350, 54), label="clock", dynamic=True),
        AlignmentNode(
            "dark-mode",
            "switch",
            (20, 125, 340, 190),
            label="Dark mode",
            interactive=True,
            states=(("checked", str(checked).lower()),),
        ),
        AlignmentNode("save", "button", (210, 610, 330, 668), label="Save", interactive=True),
    )


def projection(checked: bool = False):
    return {
        "schema": "surface_teacher_runtime_v0",
        "lesson_id": f"lesson-settings-{checked}",
        "surface_id": "demo.settings",
        "screen_key": "screen_settings",
        "grammar_hash": "grammar_settings",
        "control_hash": "controls_settings",
        "content_hash": f"content-{checked}",
        "explanation": "Settings screen with a Dark mode switch and Save button.",
        "elements": [
            {
                "element_id": "dark-mode",
                "role": "switch",
                "label": "Dark mode",
                "normalized_bounds": [0.0556, 0.1736, 0.9444, 0.2639],
                "interactive": True,
                "enabled": True,
                "states": {"checked": str(checked).lower()},
                "parent_element_id": None,
                "pixel_crop_sha256": "a" * 64,
                "sensitive": False,
            },
            {
                "element_id": "save",
                "role": "button",
                "label": "Save",
                "normalized_bounds": [0.5833, 0.8472, 0.9167, 0.9278],
                "interactive": True,
                "enabled": True,
                "states": {},
                "parent_element_id": None,
                "pixel_crop_sha256": "b" * 64,
                "sensitive": False,
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("surface_teacher_v1_demo"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    observations = [
        FrameObservation(frame(index, checked=False), nodes(False), timestamp)
        for index, timestamp in zip((12, 13, 14), (0.0, 110.0, 230.0))
    ]
    alignment = certify_alignment(observations)
    stage_alignment(alignment, out / "alignment")

    before = projection(False)
    after = projection(True)
    curriculum = build_curriculum(alignment.representative_png, before)
    stage_curriculum(curriculum, out / "curriculum", source_png=alignment.representative_png)

    graph = SurfaceTransitionGraph(out / "surface_graph.json")
    action = ActionDescriptor("tap", target_key="dark-mode", target_role="switch", target_label="Dark mode")
    transition = make_transition(
        before,
        action,
        controller_receipt_id="external-procedure-receipt-demo",
        outcome="verified",
        verified=True,
        after_projection=after,
        postcondition={"label": "Dark mode", "expected_state": "true"},
        settlement_ms=420,
        evidence={"before_lesson_id": before["lesson_id"], "after_lesson_id": after["lesson_id"]},
    )
    graph.record(before, transition, after_projection=after)

    prediction = {
        "screen_key": before["screen_key"],
        "evidence_sources": ["pixels", "surface_atlas"],
        "elements": [
            {
                "role": element["role"],
                "label": element["label"],
                "normalized_bounds": element["normalized_bounds"],
                "states": element["states"],
            }
            for element in before["elements"]
        ],
    }
    evaluation = evaluate_prediction(before, prediction)
    (out / "evaluation_receipt.json").write_bytes(json_bytes(evaluation.to_dict()))

    routing = route_perception(
        PerceptionRequest.create(
            "toggle dark mode",
            before["screen_key"],
            atlas_confidence=0.98,
            prototype_confidence=0.96,
            novelty=0.04,
            changed_fraction=alignment.certificate.dynamic_pixel_fraction,
        )
    )
    (out / "perception_route.json").write_bytes(json_bytes(asdict(routing)))

    summary = {
        "alignment_certificate_id": alignment.certificate.certificate_id,
        "dynamic_pixel_fraction": alignment.certificate.dynamic_pixel_fraction,
        "curriculum_id": curriculum.curriculum_id,
        "curriculum_tasks": len(curriculum.tasks),
        "graph_states": graph.state_count,
        "graph_transitions": graph.transition_count,
        "evaluation_score": evaluation.overall_score,
        "perception_tier": routing.selected_tier.value,
        "action_authority": "external_controller_only",
    }
    (out / "demo_summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
