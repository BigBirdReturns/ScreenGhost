from __future__ import annotations

import json

from core.surface_curriculum import (
    build_curriculum,
    build_revision_task,
    stage_curriculum,
)
from tests.surface_teacher_v1.support import png_frame, runtime_projection


def test_curriculum_emits_grounding_marks_crops_and_no_sensitive_target():
    projection = runtime_projection("settings")
    bundle = build_curriculum(png_frame(), projection)
    tasks = [task.to_dict() for task in bundle.tasks]
    task_types = {task["task"] for task in tasks}
    assert {
        "classify_screen_family",
        "extract_visible_controls",
        "ground_instruction_to_point",
        "ground_instruction_to_box",
        "select_set_of_marks_target",
        "classify_element_crop",
    } <= task_types
    serialized = json.dumps(tasks)
    assert "Password" not in serialized
    assert len(bundle.crop_pngs) == 2
    assert bundle.marked_png.startswith(b"\x89PNG")
    boxes = [task["target"] for task in tasks if task["task"] == "ground_instruction_to_box"]
    assert all(all(0 <= coordinate <= 1000 for coordinate in box) for box in boxes)


def test_curriculum_is_deterministic():
    projection = runtime_projection("settings")
    first = build_curriculum(png_frame(), projection)
    second = build_curriculum(png_frame(), projection)
    assert first.curriculum_id == second.curriculum_id
    assert [task.task_id for task in first.tasks] == [task.task_id for task in second.tasks]
    assert first.marked_png == second.marked_png


def test_curriculum_bundle_manifest(tmp_path):
    source = png_frame()
    bundle = build_curriculum(source, runtime_projection("settings"))
    stage_curriculum(bundle, tmp_path, source_png=source)
    manifest = json.loads((tmp_path / "curriculum_manifest.json").read_text())
    assert manifest["curriculum_id"] == bundle.curriculum_id
    assert manifest["task_count"] == len(bundle.tasks)
    assert "surface.png" in manifest["files"]
    assert "surface_marks.png" in manifest["files"]


def test_revision_task_is_content_addressed():
    before = runtime_projection("settings", content="off")
    after = runtime_projection("settings", content="on")
    task = build_revision_task(before, after, classification="dynamic_content")
    assert task.task == "classify_surface_revision"
    assert task.target["classification"] == "dynamic_content"
    assert task.task_id.startswith("curr_")
