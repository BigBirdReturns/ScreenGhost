"""Curriculum compiler for Surface Teacher runtime projections.

A static lesson is useful evidence, but it does not by itself teach a local model
how to ground instructions.  This module converts a teacher-blind runtime
projection into several deterministic supervision tasks:

* screen-family classification;
* visible-control extraction;
* instruction-to-point and instruction-to-box grounding;
* set-of-marks selection;
* element-role and state recognition from local crops;
* contrastive screen-revision classification.

Coordinates use a 0..1000 integer space, which is resolution-independent and
compatible with common GUI-grounding datasets.  Sensitive elements are omitted.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

CURRICULUM_SCHEMA = "surface_teacher_curriculum_v1"
CURRICULUM_BUNDLE_SCHEMA = "surface_teacher_curriculum_bundle_v1"


class CurriculumRefused(ValueError):
    """The projection cannot be converted into honest grounding supervision."""


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
    return compact or None


def _normalized_bounds(value: Sequence[Any]) -> Tuple[float, float, float, float]:
    if len(value) != 4:
        raise CurriculumRefused(f"normalized bounds require four values, got {value!r}")
    vals = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vals):
        raise CurriculumRefused(f"non-finite normalized bounds: {vals!r}")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise CurriculumRefused(f"empty or inverted normalized bounds: {vals!r}")
    if x1 < -0.001 or y1 < -0.001 or x2 > 1.001 or y2 > 1.001:
        raise CurriculumRefused(f"normalized bounds fall outside [0,1]: {vals!r}")
    return tuple(max(0.0, min(1.0, item)) for item in vals)  # type: ignore[return-value]


@dataclass(frozen=True)
class CurriculumPolicy:
    coordinate_scale: int = 1000
    maximum_elements: int = 128
    include_unlabeled_interactive: bool = True
    include_state_tasks: bool = True
    include_element_crops: bool = True
    mark_padding_px: int = 2

    def require_valid(self) -> None:
        if self.coordinate_scale < 100:
            raise CurriculumRefused("coordinate_scale must be at least 100")
        if self.maximum_elements < 1:
            raise CurriculumRefused("maximum_elements must be positive")
        if self.mark_padding_px < 0:
            raise CurriculumRefused("mark_padding_px cannot be negative")


@dataclass(frozen=True)
class CurriculumTask:
    task_id: str
    task: str
    image: str
    prompt: str
    target: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CURRICULUM_SCHEMA,
            "task_id": self.task_id,
            "task": self.task,
            "image": self.image,
            "prompt": self.prompt,
            "target": self.target,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CurriculumBundle:
    curriculum_id: str
    tasks: Tuple[CurriculumTask, ...]
    marked_png: bytes
    crop_pngs: Mapping[str, bytes]
    source_image_sha256: str
    marked_image_sha256: str


def _decode_png(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise CurriculumRefused(f"source is not a decodable PNG: {exc}") from exc


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _reading_order(element: Mapping[str, Any]) -> tuple[float, float, float, float, str, str]:
    x1, y1, x2, y2 = _normalized_bounds(element.get("normalized_bounds") or ())
    return (y1, x1, y2, x2, str(element.get("role") or ""), str(element.get("label") or ""))


def _scaled_box(bounds: Sequence[Any], scale: int) -> list[int]:
    x1, y1, x2, y2 = _normalized_bounds(bounds)
    return [
        int(round(x1 * scale)),
        int(round(y1 * scale)),
        int(round(x2 * scale)),
        int(round(y2 * scale)),
    ]


def _scaled_point(bounds: Sequence[Any], scale: int) -> list[int]:
    box = _scaled_box(bounds, scale)
    return [int(round((box[0] + box[2]) / 2)), int(round((box[1] + box[3]) / 2))]


def _pixel_box(bounds: Sequence[Any], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _normalized_bounds(bounds)
    left = max(0, min(width - 1, int(math.floor(x1 * width))))
    top = max(0, min(height - 1, int(math.floor(y1 * height))))
    right = max(left + 1, min(width, int(math.ceil(x2 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y2 * height))))
    return left, top, right, bottom


def _element_phrase(element: Mapping[str, Any], ordinal: int) -> str:
    role = str(element.get("role") or "control").replace("_", " ")
    label = _clean(element.get("label"))
    if label:
        return f"the {role} labeled {label!r}"
    return f"unlabeled {role} number {ordinal} in reading order"


def _safe_elements(projection: Mapping[str, Any], policy: CurriculumPolicy) -> list[dict[str, Any]]:
    raw = projection.get("elements")
    if not isinstance(raw, list):
        raise CurriculumRefused("runtime projection has no elements list")
    elements: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise CurriculumRefused("runtime projection element is not an object")
        if bool(item.get("sensitive", False)):
            continue
        normalized = _normalized_bounds(item.get("normalized_bounds") or ())
        row = dict(item)
        row["normalized_bounds"] = list(normalized)
        row["role"] = str(item.get("role") or "unknown")
        row["label"] = _clean(item.get("label"))
        row["interactive"] = bool(item.get("interactive", False))
        row["enabled"] = bool(item.get("enabled", True))
        states = item.get("states") or {}
        if not isinstance(states, Mapping):
            raise CurriculumRefused("runtime projection states must be an object")
        row["states"] = {str(key): str(value) for key, value in states.items()}
        elements.append(row)
    elements.sort(key=_reading_order)
    if len(elements) > policy.maximum_elements:
        raise CurriculumRefused(
            f"projection exposes {len(elements)} elements, above maximum {policy.maximum_elements}"
        )
    return elements


def _task(
    task: str,
    image: str,
    prompt: str,
    target: Any,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CurriculumTask:
    payload = {
        "task": task,
        "image": image,
        "prompt": prompt,
        "target": target,
        "metadata": dict(metadata or {}),
    }
    return CurriculumTask(
        task_id="curr_" + _sha256(_json_bytes(payload)),
        task=task,
        image=image,
        prompt=prompt,
        target=target,
        metadata=dict(metadata or {}),
    )


def _render_marks(
    image: Image.Image,
    marked: Sequence[tuple[int, Mapping[str, Any]]],
    padding: int,
) -> bytes:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for mark, element in marked:
        left, top, right, bottom = _pixel_box(element["normalized_bounds"], image.width, image.height)
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.width - 1, right + padding)
        bottom = min(image.height - 1, bottom + padding)
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=2)
        label = str(mark)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        tag_right = min(image.width - 1, left + text_w + 6)
        tag_bottom = min(image.height - 1, top + text_h + 4)
        draw.rectangle((left, top, tag_right, tag_bottom), fill=(255, 255, 0))
        draw.text((left + 3, top + 2), label, fill=(0, 0, 0), font=font)
    return _encode_png(output)


def build_curriculum(
    png_bytes: bytes,
    runtime_projection: Mapping[str, Any],
    policy: CurriculumPolicy = CurriculumPolicy(),
) -> CurriculumBundle:
    """Compile one runtime projection into deterministic GUI supervision."""

    policy.require_valid()
    image = _decode_png(png_bytes)
    elements = _safe_elements(runtime_projection, policy)
    interactive = [
        element
        for element in elements
        if element["interactive"]
        and (policy.include_unlabeled_interactive or element.get("label"))
    ]
    marked = list(enumerate(interactive, start=1))
    marked_png = _render_marks(image, marked, policy.mark_padding_px)
    source_sha = _sha256(bytes(png_bytes))
    screen_key = str(runtime_projection.get("screen_key") or "unknown")
    surface_id = str(runtime_projection.get("surface_id") or "unknown")
    explanation = str(runtime_projection.get("explanation") or "")
    coordinate_metadata = {
        "coordinate_space": f"integer_0_{policy.coordinate_scale}",
        "source_image_sha256": source_sha,
        "screen_key": screen_key,
        "surface_id": surface_id,
    }

    tasks: list[CurriculumTask] = [
        _task(
            "classify_screen_family",
            "surface.png",
            "Identify the known screen family represented by this interface.",
            {"screen_key": screen_key, "surface_id": surface_id},
            coordinate_metadata,
        ),
        _task(
            "explain_surface",
            "surface.png",
            "Explain the visible interface and its exposed controls.",
            explanation,
            coordinate_metadata,
        ),
        _task(
            "extract_visible_controls",
            "surface.png",
            "Return the visible interactive controls in reading order.",
            [
                {
                    "mark": mark,
                    "role": element["role"],
                    "label": element.get("label"),
                    "enabled": element["enabled"],
                    "states": element["states"],
                    "box": _scaled_box(element["normalized_bounds"], policy.coordinate_scale),
                }
                for mark, element in marked
            ],
            coordinate_metadata,
        ),
    ]

    crops: dict[str, bytes] = {}
    for ordinal, (mark, element) in enumerate(marked, start=1):
        phrase = _element_phrase(element, ordinal)
        element_id = str(element.get("element_id") or f"mark-{mark}")
        metadata = dict(coordinate_metadata)
        metadata.update({"element_id": element_id, "mark": mark})
        tasks.extend(
            [
                _task(
                    "ground_instruction_to_point",
                    "surface.png",
                    f"Point to {phrase}.",
                    _scaled_point(element["normalized_bounds"], policy.coordinate_scale),
                    metadata,
                ),
                _task(
                    "ground_instruction_to_box",
                    "surface.png",
                    f"Return the bounding box of {phrase}.",
                    _scaled_box(element["normalized_bounds"], policy.coordinate_scale),
                    metadata,
                ),
                _task(
                    "select_set_of_marks_target",
                    "surface_marks.png",
                    f"Which numbered mark identifies {phrase}?",
                    mark,
                    metadata,
                ),
            ]
        )
        if policy.include_state_tasks and element["states"]:
            tasks.append(
                _task(
                    "read_element_state",
                    "surface.png",
                    f"Read the visible state of {phrase}.",
                    {
                        "enabled": element["enabled"],
                        "states": element["states"],
                    },
                    metadata,
                )
            )
        if policy.include_element_crops:
            crop_name = f"elements/{element_id}.png"
            crop = image.crop(_pixel_box(element["normalized_bounds"], image.width, image.height)).convert("RGB")
            crop_bytes = _encode_png(crop)
            crops[crop_name] = crop_bytes
            tasks.append(
                _task(
                    "classify_element_crop",
                    crop_name,
                    "Identify the semantic role and visible label of this interface element.",
                    {"role": element["role"], "label": element.get("label")},
                    metadata,
                )
            )

    payload = {
        "schema": CURRICULUM_SCHEMA,
        "source_image_sha256": source_sha,
        "marked_image_sha256": _sha256(marked_png),
        "projection_identity": {
            "screen_key": screen_key,
            "grammar_hash": runtime_projection.get("grammar_hash"),
            "control_hash": runtime_projection.get("control_hash"),
            "content_hash": runtime_projection.get("content_hash"),
        },
        "policy": asdict(policy),
        "task_ids": [task.task_id for task in tasks],
        "crop_sha256": {name: _sha256(data) for name, data in sorted(crops.items())},
    }
    curriculum_id = "curriculum_" + _sha256(_json_bytes(payload))
    return CurriculumBundle(
        curriculum_id=curriculum_id,
        tasks=tuple(tasks),
        marked_png=marked_png,
        crop_pngs=crops,
        source_image_sha256=source_sha,
        marked_image_sha256=_sha256(marked_png),
    )


def build_revision_task(
    before_projection: Mapping[str, Any],
    after_projection: Mapping[str, Any],
    *,
    classification: str,
) -> CurriculumTask:
    """Create one contrastive task from an externally classified lesson pair."""

    allowed = {"same_screen", "dynamic_content", "control_drift", "structural_drift"}
    if classification not in allowed:
        raise CurriculumRefused(f"unknown revision classification: {classification!r}")
    target = {
        "classification": classification,
        "before_screen_key": before_projection.get("screen_key"),
        "after_screen_key": after_projection.get("screen_key"),
    }
    return _task(
        "classify_surface_revision",
        "before_after_pair",
        "Classify how the second interface differs from the first.",
        target,
        {
            "before_lesson_id": before_projection.get("lesson_id"),
            "after_lesson_id": after_projection.get("lesson_id"),
        },
    )


def stage_curriculum(
    bundle: CurriculumBundle,
    out_dir: str | Path,
    *,
    source_png: Optional[bytes] = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {
        "surface_marks.png": bundle.marked_png,
        "curriculum.jsonl": b"".join(_json_bytes(task.to_dict()) for task in bundle.tasks),
    }
    if source_png is not None:
        if _sha256(source_png) != bundle.source_image_sha256:
            raise CurriculumRefused("source_png does not match the curriculum source hash")
        files["surface.png"] = bytes(source_png)
    files.update(bundle.crop_pngs)
    for name, data in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {
        "schema": CURRICULUM_BUNDLE_SCHEMA,
        "curriculum_id": bundle.curriculum_id,
        "task_count": len(bundle.tasks),
        "files": {name: _sha256(data) for name, data in sorted(files.items())},
    }
    (out / "curriculum_manifest.json").write_bytes(_json_bytes(manifest))
    return out
