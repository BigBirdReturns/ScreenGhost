from __future__ import annotations

import hashlib
import json

from core.surface_teacher import AtlasMatchKind, SurfaceAtlas, stage_lesson, training_records
from .support import ANDROID_XML, compile_fixture, nodes, png_bytes


def test_staged_bundle_preserves_surface_bytes_and_writes_training_records(tmp_path):
    artifact = compile_fixture()
    out = stage_lesson(artifact, tmp_path / "lesson")
    assert (out / "surface.png").read_bytes() == artifact.png_bytes
    manifest = json.loads((out / "lesson_manifest.json").read_text())
    assert manifest["lesson_id"] == artifact.lesson.lesson_id
    assert manifest["files"]["surface.png"] == hashlib.sha256(
        artifact.png_bytes
    ).hexdigest()
    lines = [
        json.loads(line)
        for line in (out / "training.jsonl").read_text().splitlines()
    ]
    assert [line["task"] for line in lines] == [
        "explain_surface",
        "ground_visible_elements",
    ]
    assert all((out / path).exists() for path in manifest["files"])


def test_training_records_are_teacher_blind():
    records = training_records(compile_fixture().lesson)
    encoded = json.dumps(records)
    assert "source_ref" not in encoded
    assert "parent_ref" not in encoded
    assert "raw_type" not in encoded


def test_atlas_persists_runtime_projection_only_and_tracks_repeated_observations(tmp_path):
    path = tmp_path / "atlas.json"
    first = compile_fixture(payload="a").lesson
    second = compile_fixture(payload="b", nodes=nodes(field_value="Ada")).lesson
    atlas = SurfaceAtlas(path)
    atlas.add(first)
    atlas.add(second)
    exact = atlas.match(
        surface_id=second.surface_id,
        grammar_hash=second.grammar_hash,
        control_hash=second.control_hash,
    )
    assert exact.kind is AtlasMatchKind.EXACT
    assert exact.observation_count == 2
    serialized = path.read_text()
    assert "source_ref" not in serialized and "raw_type" not in serialized
    assert "Jonathan" not in serialized and "Ada" not in serialized


def test_atlas_reports_grammar_only_when_stable_control_copy_changes(tmp_path):
    atlas = SurfaceAtlas(tmp_path / "atlas.json")
    original = compile_fixture(nodes=nodes(button_label="Continue"), payload="a").lesson
    changed = compile_fixture(nodes=nodes(button_label="Next"), payload="b").lesson
    atlas.add(original)
    match = atlas.match(
        surface_id=changed.surface_id,
        grammar_hash=changed.grammar_hash,
        control_hash=changed.control_hash,
    )
    assert match.kind is AtlasMatchKind.GRAMMAR_ONLY


def test_offline_demo_exercises_the_public_file_compiler(tmp_path):
    from examples.surface_teacher_demo import main

    png_path = tmp_path / "screen.png"
    xml_path = tmp_path / "window.xml"
    out_dir = tmp_path / "lesson"
    atlas_path = tmp_path / "atlas.json"
    png_path.write_bytes(png_bytes())
    xml_path.write_text(ANDROID_XML, encoding="utf-8")

    result = main(
        [
            "android-files",
            "--png",
            str(png_path),
            "--xml",
            str(xml_path),
            "--surface-id",
            "com.example/.Login",
            "--out",
            str(out_dir),
            "--atlas",
            str(atlas_path),
        ]
    )

    assert result == 0
    assert (out_dir / "teacher_lesson.json").exists()
    assert (out_dir / "runtime_projection.json").exists()
    assert (out_dir / "training.jsonl").exists()
    assert json.loads(atlas_path.read_text())["entries"][0]["observation_count"] == 1
