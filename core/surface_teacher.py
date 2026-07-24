"""ScreenGhost Surface Teacher v0.

This module compiles one rendered surface plus one privileged structural snapshot
(UI Automator or DOM) into a deterministic lesson. The privileged source explains
what the pixels contain during teaching, but it has no input authority and the
runtime projection deliberately removes selectors, source node identifiers, raw
values, and source payloads.

The lesson is the boundary object:

    rendered PNG + privileged structure
        -> validated pixel/semantic correspondences
        -> stable grammar, control, and content signatures
        -> a human/model-readable explanation
        -> a teacher record and a teacher-blind runtime projection

It is an observation compiler, not a navigator. There are no tap, click, key,
launch, URL, listener, daemon, or model-service surfaces in this module.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from core.pixel_evidence import build_capture, is_png, png_dimensions, sha256_hex

LESSON_SCHEMA = "surface_teacher_lesson_v0"
RUNTIME_SCHEMA = "surface_teacher_runtime_v0"
ATLAS_SCHEMA = "surface_teacher_atlas_v0"
LESSON_TIER = "privileged_surface_lesson"
LESSON_TIER_LIMITS: Tuple[str, ...] = (
    "correlates rendered pixels with a declared privileged structure source",
    "teaching, annotation, scoring, and diagnosis only",
    "does not authorize, select, or execute input",
    "does not prove vendor backend or platform state",
    "runtime projection omits selectors, source identifiers, raw values, and source payloads",
    "live adapters refuse pixel/structure pairs that change during capture",
)

STABLE_CONTROL_ROLES = frozenset(
    {
        "button",
        "icon_button",
        "text_field",
        "checkbox",
        "radio",
        "switch",
        "tab",
        "menu_item",
        "navigation",
        "dialog",
    }
)


class LessonRefused(ValueError):
    """The evidence pair is malformed or internally contradictory."""


class SourceKind(str, Enum):
    ANDROID_UIAUTOMATOR = "android_uiautomator"
    WEB_DOM = "web_dom"


class RevisionClass(str, Enum):
    SAME_SCREEN = "same_screen"
    DYNAMIC_CONTENT = "dynamic_content"
    CONTROL_DRIFT = "control_drift"
    STRUCTURAL_DRIFT = "structural_drift"


class AtlasMatchKind(str, Enum):
    EXACT = "exact"
    GRAMMAR_ONLY = "grammar_only"
    NONE = "none"


@dataclass(frozen=True)
class Rect:
    """Pixel bounds using an exclusive right/bottom edge."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def require_valid(self) -> None:
        vals = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(v) for v in vals):
            raise LessonRefused(f"non-finite bounds: {vals}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise LessonRefused(f"empty or inverted bounds: {vals}")

    def visible_intersection(self, width: int, height: int) -> Optional["Rect"]:
        self.require_valid()
        clipped = Rect(
            max(0.0, self.x1),
            max(0.0, self.y1),
            min(float(width), self.x2),
            min(float(height), self.y2),
        )
        return clipped if clipped.area > 0 else None

    def normalized(self, width: int, height: int) -> Tuple[float, float, float, float]:
        if width <= 0 or height <= 0:
            raise LessonRefused(f"invalid viewport: {width}x{height}")
        return (
            round(self.x1 / width, 4),
            round(self.y1 / height, 4),
            round(self.x2 / width, 4),
            round(self.y2 / height, 4),
        )

    def crop_box(self) -> Tuple[int, int, int, int]:
        return (
            int(math.floor(self.x1)),
            int(math.floor(self.y1)),
            int(math.ceil(self.x2)),
            int(math.ceil(self.y2)),
        )


@dataclass(frozen=True)
class TeacherNode:
    """One visible structural node supplied by a privileged source adapter.

    ``source_ref`` and ``parent_ref`` are teacher-only identifiers. They are useful
    for provenance and tree reconstruction, but are never copied into the runtime
    projection.
    """

    source_ref: str
    role: str
    bounds: Rect
    label: Optional[str] = None
    value: Optional[str] = None
    interactive: bool = False
    enabled: bool = True
    visible: bool = True
    parent_ref: Optional[str] = None
    states: Tuple[Tuple[str, str], ...] = ()
    label_source: Optional[str] = None
    raw_type: Optional[str] = None
    sensitive: bool = False

    def state_dict(self) -> Dict[str, str]:
        return dict(self.states)


@dataclass(frozen=True)
class LessonPolicy:
    """Retention policy for compiled lessons.

    Labels are retained because they are the supervision signal. Raw field values
    are redacted by default. Sensitive values are never retained even when
    ``retain_values`` is true.
    """

    retain_values: bool = False
    retain_noninteractive_text: bool = True
    max_text_chars: int = 160
    write_element_crops: bool = True

    def require_valid(self) -> None:
        if self.max_text_chars < 1:
            raise LessonRefused("max_text_chars must be positive")


@dataclass(frozen=True)
class CompiledElement:
    element_id: str
    role: str
    label: Optional[str]
    value: Optional[str]
    value_sha256: Optional[str]
    value_length: Optional[int]
    normalized_bounds: Tuple[float, float, float, float]
    interactive: bool
    enabled: bool
    clipped: bool
    states: Tuple[Tuple[str, str], ...]
    parent_element_id: Optional[str]
    pixel_crop_sha256: str
    source_ref: str
    parent_ref: Optional[str]
    label_source: Optional[str]
    raw_type: Optional[str]
    sensitive: bool

    def teacher_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def runtime_dict(self) -> Dict[str, Any]:
        """Projection safe for runtime memory.

        The projection keeps the learned visual/semantic correlation but removes
        the privileged locator and source-specific fields.
        """
        return {
            "element_id": self.element_id,
            "role": self.role,
            "label": self.label,
            "normalized_bounds": list(self.normalized_bounds),
            "interactive": self.interactive,
            "enabled": self.enabled,
            "states": {k: v for k, v in self.states},
            "parent_element_id": self.parent_element_id,
            "pixel_crop_sha256": self.pixel_crop_sha256,
            "sensitive": self.sensitive,
        }


@dataclass(frozen=True)
class SurfaceLesson:
    schema: str
    lesson_id: str
    surface_id: str
    source_kind: str
    source_payload_sha256: str
    pixel_sha256: str
    pixel_manifest: Dict[str, Any]
    width: int
    height: int
    grammar_hash: str
    control_hash: str
    content_hash: str
    observation_hash: str
    screen_key: str
    explanation: str
    elements: Tuple[CompiledElement, ...]
    evidence_tier: str = LESSON_TIER
    evidence_tier_limits: Tuple[str, ...] = LESSON_TIER_LIMITS
    app_version: Optional[str] = None
    locale: Optional[str] = None

    def teacher_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "lesson_id": self.lesson_id,
            "surface_id": self.surface_id,
            "source_kind": self.source_kind,
            "source_payload_sha256": self.source_payload_sha256,
            "pixel_sha256": self.pixel_sha256,
            "pixel_manifest": self.pixel_manifest,
            "width": self.width,
            "height": self.height,
            "grammar_hash": self.grammar_hash,
            "control_hash": self.control_hash,
            "content_hash": self.content_hash,
            "observation_hash": self.observation_hash,
            "screen_key": self.screen_key,
            "explanation": self.explanation,
            "elements": [e.teacher_dict() for e in self.elements],
            "evidence_tier": self.evidence_tier,
            "evidence_tier_limits": list(self.evidence_tier_limits),
            "app_version": self.app_version,
            "locale": self.locale,
        }

    def runtime_projection(self) -> Dict[str, Any]:
        return {
            "schema": RUNTIME_SCHEMA,
            "lesson_id": self.lesson_id,
            "surface_id": self.surface_id,
            "grammar_hash": self.grammar_hash,
            "control_hash": self.control_hash,
            "content_hash": self.content_hash,
            "screen_key": self.screen_key,
            "explanation": self.explanation,
            "width": self.width,
            "height": self.height,
            "elements": [e.runtime_dict() for e in self.elements],
            "provenance": {
                "compiled_from": self.source_kind,
                "teacher_lesson_id": self.lesson_id,
                "runtime_visibility": "teacher_blind",
                "privileged_fields_removed": True,
            },
        }


@dataclass(frozen=True)
class LessonArtifact:
    lesson: SurfaceLesson
    png_bytes: bytes
    crop_pngs: Mapping[str, bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class RevisionFinding:
    classification: RevisionClass
    same_grammar: bool
    same_controls: bool
    same_content: bool
    detail: str


@dataclass(frozen=True)
class AtlasMatch:
    kind: AtlasMatchKind
    screen_key: Optional[str]
    lesson_id: Optional[str]
    observation_count: int = 0


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _hash_json(value: Any) -> str:
    return sha256_hex(_json_bytes(value))


def _clean_text(value: Optional[str], max_chars: int) -> Optional[str]:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    if not compact:
        return None
    return compact[:max_chars]


def _role(value: str) -> str:
    compact = "_".join((value or "unknown").strip().lower().replace("-", "_").split())
    return compact or "unknown"


def _reading_order(node: TeacherNode) -> Tuple[float, float, float, float, str, str, str]:
    return (
        node.bounds.y1,
        node.bounds.x1,
        node.bounds.y2,
        node.bounds.x2,
        _role(node.role),
        (node.label or "").casefold(),
        node.source_ref,
    )


def _position_name(bounds: Tuple[float, float, float, float]) -> str:
    x = (bounds[0] + bounds[2]) / 2
    y = (bounds[1] + bounds[3]) / 2
    horizontal = "left" if x < 1 / 3 else "center" if x < 2 / 3 else "right"
    vertical = "upper" if y < 1 / 3 else "middle" if y < 2 / 3 else "lower"
    return f"{vertical} {horizontal}"


def _crop_fingerprint(image: Image.Image, box: Rect) -> Tuple[str, bytes]:
    crop = image.crop(box.crop_box()).convert("RGB")
    canonical = _json_bytes({"mode": crop.mode, "size": crop.size}) + crop.tobytes()
    fingerprint = sha256_hex(canonical)
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=False, compress_level=9)
    return fingerprint, buf.getvalue()


def _explain(surface_id: str, elements: Sequence[CompiledElement]) -> str:
    visible = len(elements)
    interactive = [e for e in elements if e.interactive]
    role_counts: Dict[str, int] = {}
    for element in elements:
        role_counts[element.role] = role_counts.get(element.role, 0) + 1
    counts = ", ".join(f"{count} {role}" for role, count in sorted(role_counts.items()))
    clauses: List[str] = []
    for element in interactive[:18]:
        label = f" labeled {element.label!r}" if element.label else " with no exposed label"
        state = ""
        states = dict(element.states)
        if states:
            useful = ", ".join(f"{k}={v}" for k, v in sorted(states.items()) if v not in ("false", "none", ""))
            if useful:
                state = f" ({useful})"
        clauses.append(
            f"a {element.role}{label} at the {_position_name(element.normalized_bounds)}{state}"
        )
    if len(interactive) > 18:
        clauses.append(f"{len(interactive) - 18} additional interactive controls")
    controls = "; ".join(clauses) if clauses else "no interactive controls were exposed"
    return (
        f"Surface {surface_id!r} contains {visible} visible structural elements and "
        f"{len(interactive)} interactive controls. The structural inventory is {counts or 'empty'}. "
        f"In reading order, the exposed controls are {controls}."
    )


def compile_lesson(
    png_bytes: bytes,
    *,
    surface_id: str,
    source_kind: SourceKind | str,
    source_payload_sha256: str,
    nodes: Iterable[TeacherNode],
    policy: LessonPolicy = LessonPolicy(),
    app_version: Optional[str] = None,
    locale: Optional[str] = None,
) -> LessonArtifact:
    """Compile pixels and a privileged node snapshot into a deterministic lesson."""
    policy.require_valid()
    if not surface_id or not surface_id.strip():
        raise LessonRefused("surface_id is required")
    try:
        source_kind_value = SourceKind(source_kind).value
    except ValueError as exc:
        raise LessonRefused(f"unsupported source_kind: {source_kind!r}") from exc
    if not source_payload_sha256 or len(source_payload_sha256) != 64:
        raise LessonRefused("source_payload_sha256 must be a 64-character SHA-256 hex digest")
    try:
        int(source_payload_sha256, 16)
    except ValueError as exc:
        raise LessonRefused("source_payload_sha256 is not hexadecimal") from exc
    source_payload_sha256 = source_payload_sha256.lower()
    if not is_png(png_bytes):
        raise LessonRefused("surface teacher accepts PNG evidence only")
    declared = png_dimensions(png_bytes)
    if declared is None:
        raise LessonRefused("PNG has no readable IHDR dimensions")
    try:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
        image = image.convert("RGB")
    except Exception as exc:  # Pillow raises several format-specific exceptions
        raise LessonRefused(f"PNG could not be decoded: {exc}") from exc
    width, height = image.size
    if (width, height) != declared:
        raise LessonRefused(f"PNG decoded dimensions {image.size} disagree with IHDR {declared}")

    raw_nodes = sorted((n for n in nodes if n.visible), key=_reading_order)
    if not raw_nodes:
        raise LessonRefused("privileged source exposed no visible nodes")
    seen_refs: set[str] = set()
    retained: List[Tuple[TeacherNode, Rect, bool]] = []
    for node in raw_nodes:
        if not node.source_ref or not node.source_ref.strip():
            raise LessonRefused("every teacher node requires a source_ref")
        if node.source_ref in seen_refs:
            raise LessonRefused(f"duplicate source_ref: {node.source_ref}")
        seen_refs.add(node.source_ref)
        clipped = node.bounds.visible_intersection(width, height)
        if clipped is None:
            if node.interactive:
                raise LessonRefused(f"interactive node is entirely outside viewport: {node.source_ref}")
            continue
        retained.append((node, clipped, clipped != node.bounds))
    if not retained:
        raise LessonRefused("no nodes intersect the rendered viewport")

    retained_by_ref = {node.source_ref: (node, bounds, was_clipped) for node, bounds, was_clipped in retained}
    provisional: List[Dict[str, Any]] = []
    crop_pngs: Dict[str, bytes] = {}
    seed_counts: Dict[str, int] = {}

    def structural_descriptor(item: Optional[Tuple[TeacherNode, Rect, bool]]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        parent_node, parent_bounds, _ = item
        return {
            "role": _role(parent_node.role),
            "bounds": list(parent_bounds.normalized(width, height)),
            "interactive": bool(parent_node.interactive),
        }

    for node, bounds, was_clipped in retained:
        role = _role(node.role)
        label = _clean_text(node.label, policy.max_text_chars)
        if not node.interactive and not policy.retain_noninteractive_text:
            label = None
        raw_value = _clean_text(node.value, policy.max_text_chars)
        if node.sensitive:
            value = None
            value_hash = None
            value_length = None
        else:
            value = raw_value if policy.retain_values else None
            value_hash = sha256_hex(raw_value.encode("utf-8")) if raw_value is not None else None
            value_length = len(raw_value) if raw_value is not None else None
        normalized = bounds.normalized(width, height)
        element_seed = {
            "role": role,
            "bounds": list(normalized),
            "interactive": bool(node.interactive),
            "parent": structural_descriptor(retained_by_ref.get(node.parent_ref)),
        }
        seed_key = _json_bytes(element_seed).decode("utf-8")
        ordinal = seed_counts.get(seed_key, 0)
        seed_counts[seed_key] = ordinal + 1
        element_id = "el_" + _hash_json({"structure": element_seed, "ordinal": ordinal})[:20]
        crop_hash, crop_png = _crop_fingerprint(image, bounds)
        if element_id in crop_pngs:
            raise LessonRefused(f"internal element identity collision: {element_id}")
        crop_pngs[element_id] = crop_png
        provisional.append(
            {
                "node": node,
                "element_id": element_id,
                "role": role,
                "label": label,
                "value": value,
                "value_sha256": value_hash,
                "value_length": value_length,
                "normalized_bounds": normalized,
                "clipped": was_clipped,
                "pixel_crop_sha256": crop_hash,
            }
        )

    source_to_element = {row["node"].source_ref: row["element_id"] for row in provisional}
    compiled: List[CompiledElement] = []
    for row in provisional:
        node: TeacherNode = row["node"]
        compiled.append(
            CompiledElement(
                element_id=row["element_id"],
                role=row["role"],
                label=row["label"],
                value=row["value"],
                value_sha256=row["value_sha256"],
                value_length=row["value_length"],
                normalized_bounds=row["normalized_bounds"],
                interactive=bool(node.interactive),
                enabled=bool(node.enabled),
                clipped=bool(row["clipped"]),
                states=tuple(sorted((str(k), str(v)) for k, v in node.states)),
                parent_element_id=source_to_element.get(node.parent_ref),
                pixel_crop_sha256=row["pixel_crop_sha256"],
                source_ref=node.source_ref,
                parent_ref=node.parent_ref,
                label_source=node.label_source,
                raw_type=node.raw_type,
                sensitive=bool(node.sensitive),
            )
        )

    grammar_terms = [
        {
            "role": e.role,
            "bounds": list(e.normalized_bounds),
            "interactive": e.interactive,
            "parent": e.parent_element_id,
        }
        for e in compiled
    ]
    control_terms = [
        {
            "role": e.role,
            "label": (e.label or "").casefold() if e.role in STABLE_CONTROL_ROLES else None,
            "bounds": list(e.normalized_bounds),
            "interactive": e.interactive,
            "parent": e.parent_element_id,
        }
        for e in compiled
    ]
    content_terms = [
        {
            "element_id": e.element_id,
            "label": e.label,
            "value_sha256": e.value_sha256,
            "states": list(e.states),
            "enabled": e.enabled,
        }
        for e in compiled
    ]
    grammar_hash = _hash_json(grammar_terms)
    control_hash = _hash_json(control_terms)
    content_hash = _hash_json(content_terms)
    pixel_capture = build_capture(
        png_bytes,
        capture_method="surface_teacher",
        source_label=surface_id,
    )
    pixel_manifest = pixel_capture.to_manifest()
    pixel_sha = pixel_capture.image_sha256
    observation_hash = _hash_json(
        {
            "pixel_sha256": pixel_sha,
            "source_kind": source_kind_value,
            "source_payload_sha256": source_payload_sha256,
            "content_hash": content_hash,
        }
    )
    screen_key = "screen_" + _hash_json(
        {"surface_id": surface_id, "grammar_hash": grammar_hash, "control_hash": control_hash}
    )[:24]
    explanation = _explain(surface_id, compiled)

    lesson_payload = {
        "schema": LESSON_SCHEMA,
        "surface_id": surface_id,
        "source_kind": source_kind_value,
        "source_payload_sha256": source_payload_sha256,
        "pixel_sha256": pixel_sha,
        "grammar_hash": grammar_hash,
        "control_hash": control_hash,
        "content_hash": content_hash,
        "observation_hash": observation_hash,
        "screen_key": screen_key,
        "elements": [e.teacher_dict() for e in compiled],
        "policy": asdict(policy),
        "app_version": app_version,
        "locale": locale,
    }
    lesson_id = "lesson_" + _hash_json(lesson_payload)
    lesson = SurfaceLesson(
        schema=LESSON_SCHEMA,
        lesson_id=lesson_id,
        surface_id=surface_id,
        source_kind=source_kind_value,
        source_payload_sha256=source_payload_sha256,
        pixel_sha256=pixel_sha,
        pixel_manifest=pixel_manifest,
        width=width,
        height=height,
        grammar_hash=grammar_hash,
        control_hash=control_hash,
        content_hash=content_hash,
        observation_hash=observation_hash,
        screen_key=screen_key,
        explanation=explanation,
        elements=tuple(compiled),
        app_version=app_version,
        locale=locale,
    )
    return LessonArtifact(
        lesson=lesson,
        png_bytes=bytes(png_bytes),
        crop_pngs=crop_pngs if policy.write_element_crops else {},
    )


def compare_lessons(before: SurfaceLesson, after: SurfaceLesson) -> RevisionFinding:
    same_grammar = before.grammar_hash == after.grammar_hash
    same_controls = before.control_hash == after.control_hash
    same_content = before.content_hash == after.content_hash
    if same_grammar and same_controls and same_content:
        classification = RevisionClass.SAME_SCREEN
        detail = "grammar, stable controls, and exposed content match"
    elif same_grammar and same_controls:
        classification = RevisionClass.DYNAMIC_CONTENT
        detail = "layout and stable controls match; labels, values, or state changed"
    elif same_grammar:
        classification = RevisionClass.CONTROL_DRIFT
        detail = "layout grammar matches but stable control semantics changed"
    else:
        classification = RevisionClass.STRUCTURAL_DRIFT
        detail = "roles, geometry, interactivity, or containment changed"
    return RevisionFinding(classification, same_grammar, same_controls, same_content, detail)


def training_records(lesson: SurfaceLesson) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    runtime = lesson.runtime_projection()
    return (
        {
            "schema": "surface_teacher_training_v0",
            "task": "explain_surface",
            "image": "surface.png",
            "image_sha256": lesson.pixel_sha256,
            "target": lesson.explanation,
            "lesson_id": lesson.lesson_id,
        },
        {
            "schema": "surface_teacher_training_v0",
            "task": "ground_visible_elements",
            "image": "surface.png",
            "image_sha256": lesson.pixel_sha256,
            "target": runtime["elements"],
            "lesson_id": lesson.lesson_id,
        },
    )


def stage_lesson(artifact: LessonArtifact, out_dir: str | Path) -> Path:
    """Write a deterministic, inspectable lesson bundle.

    The original surface PNG is copied verbatim. Element crops are derived teaching
    artifacts and are not represented as original evidence.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    surface_path = out / "surface.png"
    teacher_path = out / "teacher_lesson.json"
    runtime_path = out / "runtime_projection.json"
    training_path = out / "training.jsonl"
    manifest_path = out / "lesson_manifest.json"

    surface_path.write_bytes(artifact.png_bytes)
    teacher_path.write_bytes(_json_bytes(artifact.lesson.teacher_dict()))
    runtime_path.write_bytes(_json_bytes(artifact.lesson.runtime_projection()))
    training_path.write_bytes(b"".join(_json_bytes(record) for record in training_records(artifact.lesson)))

    files: Dict[str, str] = {
        "surface.png": sha256_hex(surface_path.read_bytes()),
        "teacher_lesson.json": sha256_hex(teacher_path.read_bytes()),
        "runtime_projection.json": sha256_hex(runtime_path.read_bytes()),
        "training.jsonl": sha256_hex(training_path.read_bytes()),
    }
    if artifact.crop_pngs:
        crop_dir = out / "elements"
        crop_dir.mkdir(exist_ok=True)
        for element_id, crop_png in sorted(artifact.crop_pngs.items()):
            rel = f"elements/{element_id}.png"
            (out / rel).write_bytes(crop_png)
            files[rel] = sha256_hex(crop_png)

    manifest = {
        "schema": "surface_teacher_bundle_v0",
        "lesson_id": artifact.lesson.lesson_id,
        "screen_key": artifact.lesson.screen_key,
        "evidence_tier": LESSON_TIER,
        "evidence_tier_limits": list(LESSON_TIER_LIMITS),
        "files": files,
    }
    manifest_path.write_bytes(_json_bytes(manifest))
    return out


class SurfaceAtlas:
    """JSON store containing teacher-blind runtime projections only."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: Dict[str, Any] = {"schema": ATLAS_SCHEMA, "entries": []}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("schema") != ATLAS_SCHEMA:
                raise LessonRefused(f"unsupported atlas schema: {loaded.get('schema')!r}")
            self._data = loaded

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(self._data["entries"])

    def add(self, lesson: SurfaceLesson) -> None:
        projection = lesson.runtime_projection()
        entries = self._data["entries"]
        for entry in entries:
            if entry["screen_key"] == lesson.screen_key:
                entry["observation_count"] += 1
                entry["latest_lesson_id"] = lesson.lesson_id
                entry["latest_content_hash"] = lesson.content_hash
                entry["runtime_projection"] = projection
                self._persist()
                return
        entries.append(
            {
                "screen_key": lesson.screen_key,
                "surface_id": lesson.surface_id,
                "grammar_hash": lesson.grammar_hash,
                "control_hash": lesson.control_hash,
                "latest_content_hash": lesson.content_hash,
                "latest_lesson_id": lesson.lesson_id,
                "observation_count": 1,
                "runtime_projection": projection,
            }
        )
        entries.sort(key=lambda e: (e["surface_id"], e["screen_key"]))
        self._persist()

    def match(self, *, surface_id: str, grammar_hash: str, control_hash: str) -> AtlasMatch:
        grammar_candidate: Optional[Dict[str, Any]] = None
        for entry in self._data["entries"]:
            if entry["surface_id"] != surface_id or entry["grammar_hash"] != grammar_hash:
                continue
            if entry["control_hash"] == control_hash:
                return AtlasMatch(
                    AtlasMatchKind.EXACT,
                    entry["screen_key"],
                    entry["latest_lesson_id"],
                    entry["observation_count"],
                )
            grammar_candidate = entry
        if grammar_candidate is not None:
            return AtlasMatch(
                AtlasMatchKind.GRAMMAR_ONLY,
                grammar_candidate["screen_key"],
                grammar_candidate["latest_lesson_id"],
                grammar_candidate["observation_count"],
            )
        return AtlasMatch(AtlasMatchKind.NONE, None, None, 0)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(_json_bytes(self._data))
