from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from core.surface_teacher import (
    LessonPolicy,
    LessonRefused,
    Rect,
    RevisionClass,
    SourceKind,
    TeacherNode,
    compare_lessons,
    compile_lesson,
)
from .support import compile_fixture, nodes, png_bytes


def test_compile_is_deterministic_and_independent_of_source_node_order():
    first = compile_fixture(nodes=nodes()).lesson
    second = compile_fixture(nodes=tuple(reversed(nodes()))).lesson
    assert first.lesson_id == second.lesson_id
    assert first.grammar_hash == second.grammar_hash
    assert first.control_hash == second.control_hash
    assert [element.element_id for element in first.elements] == [
        element.element_id for element in second.elements
    ]


def test_lesson_pairs_each_element_with_a_visual_fingerprint_and_explanation():
    artifact = compile_fixture()
    lesson = artifact.lesson
    assert lesson.pixel_sha256 == hashlib.sha256(artifact.png_bytes).hexdigest()
    assert len(artifact.crop_pngs) == len(lesson.elements)
    assert all(element.pixel_crop_sha256 for element in lesson.elements)
    assert "interactive controls" in lesson.explanation
    assert "Continue" in lesson.explanation


def test_runtime_projection_removes_privileged_locators_and_raw_values():
    projection = compile_fixture(
        policy=LessonPolicy(retain_values=True)
    ).lesson.runtime_projection()
    serialized = json.dumps(projection)
    assert "source_ref" not in serialized
    assert "parent_ref" not in serialized
    assert "raw_type" not in serialized
    assert "label_source" not in serialized
    assert "Jonathan" not in serialized
    assert projection["provenance"]["compiled_from"] == "android_uiautomator"


def test_default_policy_hashes_values_but_does_not_retain_them():
    lesson = compile_fixture().lesson
    field = next(element for element in lesson.elements if element.role == "text_field")
    assert field.value is None
    assert field.value_length == len("Jonathan")
    assert field.value_sha256 == hashlib.sha256(b"Jonathan").hexdigest()


def test_sensitive_value_is_never_retained_even_when_requested():
    augmented = nodes() + (
        TeacherNode(
            source_ref="password",
            role="text_field",
            bounds=Rect(40, 400, 360, 470),
            label="Password",
            value="supersecret",
            interactive=True,
            sensitive=True,
        ),
    )
    lesson = compile_fixture(
        nodes=augmented, policy=LessonPolicy(retain_values=True)
    ).lesson
    password = next(element for element in lesson.elements if element.label == "Password")
    assert password.value is None
    assert password.value_sha256 is None and password.value_length is None
    assert "supersecret" not in json.dumps(lesson.teacher_dict())


def test_partially_visible_node_is_explicitly_clipped():
    augmented = nodes() + (
        TeacherNode(
            source_ref="bottom-sheet",
            role="dialog",
            bounds=Rect(0, 760, 400, 900),
            label="Sheet",
            interactive=True,
        ),
    )
    lesson = compile_fixture(nodes=augmented).lesson
    sheet = next(element for element in lesson.elements if element.source_ref == "bottom-sheet")
    assert sheet.clipped is True
    assert sheet.normalized_bounds == (0.0, 0.95, 1.0, 1.0)


def test_offscreen_interactive_node_is_refused_instead_of_silently_dropped():
    augmented = nodes() + (
        TeacherNode(
            source_ref="offscreen",
            role="button",
            bounds=Rect(10, 900, 100, 960),
            label="Offscreen",
            interactive=True,
        ),
    )
    with pytest.raises(LessonRefused, match="entirely outside viewport"):
        compile_fixture(nodes=augmented)


def test_structural_identity_does_not_depend_on_privileged_source_ids():
    original = nodes()
    remap = {
        node.source_ref: f"renamed-{index}" for index, node in enumerate(original)
    }
    renamed = tuple(
        replace(
            node,
            source_ref=remap[node.source_ref],
            parent_ref=remap.get(node.parent_ref) if node.parent_ref else None,
        )
        for node in original
    )
    first = compile_fixture(nodes=original, payload="source-a").lesson
    second = compile_fixture(nodes=renamed, payload="source-b").lesson
    assert first.grammar_hash == second.grammar_hash
    assert first.control_hash == second.control_hash
    assert first.content_hash == second.content_hash
    assert first.screen_key == second.screen_key
    assert [element.element_id for element in first.elements] == [
        element.element_id for element in second.elements
    ]


def test_invalid_source_hash_and_non_png_are_refused():
    with pytest.raises(LessonRefused, match="source_payload_sha256"):
        compile_lesson(
            png_bytes(),
            surface_id="x",
            source_kind=SourceKind.WEB_DOM,
            source_payload_sha256="nope",
            nodes=nodes(),
        )
    with pytest.raises(LessonRefused, match="PNG evidence only"):
        compile_lesson(
            b"not png",
            surface_id="x",
            source_kind=SourceKind.WEB_DOM,
            source_payload_sha256="0" * 64,
            nodes=nodes(),
        )
    with pytest.raises(LessonRefused, match="unsupported source_kind"):
        compile_lesson(
            png_bytes(),
            surface_id="x",
            source_kind="screen_magic",
            source_payload_sha256="0" * 64,
            nodes=nodes(),
        )


def test_dynamic_values_do_not_invalidate_the_known_screen():
    before = compile_fixture(nodes=nodes(field_value="Jonathan"), payload="a").lesson
    after = compile_fixture(nodes=nodes(field_value="Ada"), payload="b").lesson
    finding = compare_lessons(before, after)
    assert finding.classification is RevisionClass.DYNAMIC_CONTENT
    assert before.grammar_hash == after.grammar_hash
    assert before.control_hash == after.control_hash
    assert before.content_hash != after.content_hash


def test_control_copy_change_is_control_drift_not_structural_drift():
    before = compile_fixture(nodes=nodes(button_label="Continue"), payload="a").lesson
    after = compile_fixture(nodes=nodes(button_label="Next"), payload="b").lesson
    finding = compare_lessons(before, after)
    assert finding.classification is RevisionClass.CONTROL_DRIFT
    assert finding.same_grammar and not finding.same_controls


def test_role_or_geometry_change_is_structural_drift():
    before = compile_fixture(nodes=nodes(), payload="a").lesson
    role_change = compile_fixture(nodes=nodes(button_role="link"), payload="b").lesson
    moved = compile_fixture(nodes=nodes(button_y=500), payload="c").lesson
    assert compare_lessons(before, role_change).classification is RevisionClass.STRUCTURAL_DRIFT
    assert compare_lessons(before, moved).classification is RevisionClass.STRUCTURAL_DRIFT


def test_pixel_change_is_a_new_observation_without_reteaching_the_grammar():
    before = compile_fixture(
        png=png_bytes(button_fill=(30, 80, 180)), payload="same"
    ).lesson
    after = compile_fixture(
        png=png_bytes(button_fill=(30, 120, 180)), payload="same"
    ).lesson
    assert before.observation_hash != after.observation_hash
    assert before.grammar_hash == after.grammar_hash
    assert before.control_hash == after.control_hash
    assert compare_lessons(before, after).classification is RevisionClass.SAME_SCREEN
