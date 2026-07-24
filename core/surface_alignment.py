"""Temporal alignment certificates for Surface Teacher.

The static A -> structure -> B equality gate is intentionally conservative, but
real interfaces contain carets, clocks, spinners, video, animated progress, and
sub-pixel raster noise.  This module replaces a single global similarity number
with an inspectable certificate:

* stable semantic structure must agree;
* interactive control geometry must stay within a declared tolerance;
* volatile pixels are localized into a mask rather than ignored globally;
* the stable part of the image must remain within a strict error budget;
* the representative frame is the medoid of the observed burst;
* every acceptance or refusal is content-addressed and reproducible.

It is read-only.  No class in this module exposes an input method.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

ALIGNMENT_SCHEMA = "surface_teacher_alignment_v1"
ALIGNMENT_BUNDLE_SCHEMA = "surface_teacher_alignment_bundle_v1"


class AlignmentRefused(ValueError):
    """The observation burst cannot be bound to one defensible UI state."""


Bounds = Tuple[float, float, float, float]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact or None


def _bounds(value: Sequence[float]) -> Bounds:
    if len(value) != 4:
        raise AlignmentRefused(f"bounds must have four values, got {value!r}")
    vals = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in vals):
        raise AlignmentRefused(f"bounds contain a non-finite value: {vals!r}")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise AlignmentRefused(f"bounds are empty or inverted: {vals!r}")
    return vals


@dataclass(frozen=True)
class AlignmentNode:
    """One source-neutral node sampled during the alignment burst.

    ``semantic_key`` must be stable within the burst.  It may be a teacher-only
    source reference because alignment is performed before runtime projection.
    """

    semantic_key: str
    role: str
    bounds: Bounds
    label: Optional[str] = None
    interactive: bool = False
    enabled: bool = True
    parent_key: Optional[str] = None
    states: Tuple[Tuple[str, str], ...] = ()
    dynamic: bool = False

    def __post_init__(self) -> None:
        if not self.semantic_key.strip():
            raise AlignmentRefused("alignment node requires a semantic_key")
        object.__setattr__(self, "bounds", _bounds(self.bounds))
        object.__setattr__(self, "label", _clean(self.label))
        object.__setattr__(
            self,
            "states",
            tuple(sorted((str(k), str(v)) for k, v in self.states)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AlignmentNode":
        key = value.get("semantic_key") or value.get("element_id") or value.get("source_ref")
        if not key:
            raise AlignmentRefused("node mapping has no semantic_key, element_id, or source_ref")
        raw_bounds = value.get("bounds") or value.get("pixel_bounds")
        if raw_bounds is None:
            raise AlignmentRefused(f"node {key!r} has no pixel bounds")
        states = value.get("states") or ()
        if isinstance(states, Mapping):
            states = tuple(states.items())
        return cls(
            semantic_key=str(key),
            role=str(value.get("role") or "unknown"),
            bounds=_bounds(raw_bounds),
            label=value.get("label"),
            interactive=bool(value.get("interactive", False)),
            enabled=bool(value.get("enabled", True)),
            parent_key=(str(value["parent_key"]) if value.get("parent_key") is not None else None),
            states=tuple(states),
            dynamic=bool(value.get("dynamic", False)),
        )


@dataclass(frozen=True)
class FrameObservation:
    png_bytes: bytes
    nodes: Tuple[AlignmentNode, ...]
    observed_monotonic_ms: float
    source_digest: Optional[str] = None
    event_idle: bool = True
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.png_bytes, (bytes, bytearray)):
            raise AlignmentRefused("png_bytes must be bytes")
        if not self.nodes:
            raise AlignmentRefused("frame observation requires at least one node")
        if not math.isfinite(float(self.observed_monotonic_ms)):
            raise AlignmentRefused("observed_monotonic_ms must be finite")
        keys = [node.semantic_key for node in self.nodes]
        if len(keys) != len(set(keys)):
            raise AlignmentRefused("frame observation contains duplicate semantic keys")


@dataclass(frozen=True)
class AlignmentPolicy:
    minimum_samples: int = 3
    minimum_duration_ms: float = 180.0
    pixel_change_threshold: int = 10
    mask_dilation_px: int = 2
    max_dynamic_fraction: float = 0.18
    max_static_mean_difference: float = 0.35
    max_interactive_shift_px: float = 2.0
    require_event_idle: bool = False
    auto_dynamic_noninteractive: bool = True
    volatile_state_keys: Tuple[str, ...] = (
        "focused",
        "accessibility_focused",
        "cursor_visible",
    )

    def require_valid(self) -> None:
        if self.minimum_samples < 2:
            raise AlignmentRefused("minimum_samples must be at least two")
        if self.minimum_duration_ms < 0:
            raise AlignmentRefused("minimum_duration_ms cannot be negative")
        if not 0 <= self.pixel_change_threshold <= 255:
            raise AlignmentRefused("pixel_change_threshold must be in [0,255]")
        if self.mask_dilation_px < 0:
            raise AlignmentRefused("mask_dilation_px cannot be negative")
        if not 0 <= self.max_dynamic_fraction <= 1:
            raise AlignmentRefused("max_dynamic_fraction must be in [0,1]")
        if self.max_static_mean_difference < 0:
            raise AlignmentRefused("max_static_mean_difference cannot be negative")
        if self.max_interactive_shift_px < 0:
            raise AlignmentRefused("max_interactive_shift_px cannot be negative")


@dataclass(frozen=True)
class AlignmentCertificate:
    schema: str
    certificate_id: str
    accepted: bool
    reason: str
    sample_count: int
    duration_ms: float
    width: int
    height: int
    representative_index: int
    representative_pixel_sha256: str
    sample_pixel_sha256: Tuple[str, ...]
    stable_structure_sha256: str
    inferred_dynamic_keys: Tuple[str, ...]
    dynamic_pixel_fraction: float
    static_mean_difference: float
    max_interactive_shift_px: float
    event_idle_all_samples: bool
    volatility_mask_sha256: str
    policy: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["sample_pixel_sha256"] = list(self.sample_pixel_sha256)
        value["inferred_dynamic_keys"] = list(self.inferred_dynamic_keys)
        value["policy"] = dict(self.policy)
        return value


@dataclass(frozen=True)
class AlignmentArtifact:
    certificate: AlignmentCertificate
    representative_png: bytes
    volatility_mask_png: bytes


def _decode_png(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise AlignmentRefused(f"observation is not a decodable PNG: {exc}") from exc


def _canonical_pixel_sha(image: Image.Image) -> str:
    header = _json_bytes({"mode": image.mode, "size": list(image.size)})
    return _sha256(header + image.tobytes())


def _pair_mean_difference(left: Image.Image, right: Image.Image) -> float:
    diff = ImageChops.difference(left, right)
    return float(sum(ImageStat.Stat(diff).mean) / 3.0)


def _medoid_index(images: Sequence[Image.Image]) -> int:
    if len(images) == 1:
        return 0
    scores = []
    for i, image in enumerate(images):
        total = 0.0
        for j, other in enumerate(images):
            if i != j:
                total += _pair_mean_difference(image, other)
        scores.append(total)
    return min(range(len(scores)), key=lambda idx: (scores[idx], idx))


def _max_channel_difference(left: Image.Image, right: Image.Image) -> Image.Image:
    channels = ImageChops.difference(left, right).split()
    return ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])


def _binary_change_mask(left: Image.Image, right: Image.Image, threshold: int) -> Image.Image:
    max_diff = _max_channel_difference(left, right)
    return max_diff.point(lambda p: 255 if p > threshold else 0, mode="L")


def _dilate(mask: Image.Image, pixels: int) -> Image.Image:
    if pixels <= 0:
        return mask
    size = pixels * 2 + 1
    return mask.filter(ImageFilter.MaxFilter(size=size))


def _mask_fraction(mask: Image.Image) -> float:
    histogram = mask.histogram()
    changed = sum(histogram[1:])
    total = mask.width * mask.height
    return changed / total if total else 0.0


def _static_mean_difference(diff: Image.Image, dynamic_mask: Image.Image) -> float:
    gray = diff.convert("L")
    inverse = ImageChops.invert(dynamic_mask)
    static = ImageChops.multiply(gray, inverse)
    histogram = static.histogram()
    static_pixels = sum(inverse.histogram()[1:])
    if static_pixels == 0:
        return 0.0
    weighted = sum(index * count for index, count in enumerate(histogram))
    return weighted / static_pixels


def _state_without_volatile(node: AlignmentNode, policy: AlignmentPolicy) -> Tuple[Tuple[str, str], ...]:
    volatile = set(policy.volatile_state_keys)
    return tuple((key, value) for key, value in node.states if key not in volatile)


def _node_map(observation: FrameObservation) -> dict[str, AlignmentNode]:
    return {node.semantic_key: node for node in observation.nodes}


def _infer_dynamic_keys(
    observations: Sequence[FrameObservation], policy: AlignmentPolicy
) -> set[str]:
    maps = [_node_map(observation) for observation in observations]
    all_keys = set().union(*(mapping.keys() for mapping in maps))
    dynamic: set[str] = set()
    for key in all_keys:
        present = [mapping[key] for mapping in maps if key in mapping]
        if any(node.dynamic for node in present):
            dynamic.add(key)
            continue
        if not policy.auto_dynamic_noninteractive:
            continue
        if not present or any(node.interactive for node in present):
            continue
        if len(present) != len(maps):
            dynamic.add(key)
            continue
        labels = {node.label for node in present}
        states = {_state_without_volatile(node, policy) for node in present}
        if len(labels) > 1 or len(states) > 1:
            dynamic.add(key)
    return dynamic


def _stable_structure_payload(
    observations: Sequence[FrameObservation], dynamic_keys: set[str], policy: AlignmentPolicy
) -> list[dict[str, Any]]:
    maps = [_node_map(observation) for observation in observations]
    stable_keys = set(maps[0]) - dynamic_keys
    for mapping in maps[1:]:
        if (set(mapping) - dynamic_keys) != stable_keys:
            missing = sorted(stable_keys - set(mapping))
            added = sorted((set(mapping) - dynamic_keys) - stable_keys)
            raise AlignmentRefused(
                f"stable node membership changed; missing={missing[:5]!r} added={added[:5]!r}"
            )
    payload: list[dict[str, Any]] = []
    for key in sorted(stable_keys):
        first = maps[0][key]
        descriptor = {
            "semantic_key": key,
            "role": first.role,
            "label": first.label,
            "interactive": first.interactive,
            "enabled": first.enabled,
            "parent_key": first.parent_key,
            "states": list(_state_without_volatile(first, policy)),
        }
        for mapping in maps[1:]:
            node = mapping[key]
            candidate = {
                "semantic_key": key,
                "role": node.role,
                "label": node.label,
                "interactive": node.interactive,
                "enabled": node.enabled,
                "parent_key": node.parent_key,
                "states": list(_state_without_volatile(node, policy)),
            }
            if candidate != descriptor:
                raise AlignmentRefused(f"stable node semantics changed for {key!r}")
        payload.append(descriptor)
    return payload


def _max_interactive_shift(
    observations: Sequence[FrameObservation], dynamic_keys: set[str]
) -> float:
    maps = [_node_map(observation) for observation in observations]
    keys = sorted(
        key
        for key, node in maps[0].items()
        if node.interactive and key not in dynamic_keys
    )
    maximum = 0.0
    for key in keys:
        if any(key not in mapping for mapping in maps):
            raise AlignmentRefused(f"interactive node disappeared during capture: {key!r}")
        reference = maps[0][key].bounds
        for mapping in maps[1:]:
            candidate = mapping[key].bounds
            maximum = max(maximum, *(abs(a - b) for a, b in zip(reference, candidate)))
    return maximum


def _draw_dynamic_nodes(
    mask: Image.Image,
    observations: Sequence[FrameObservation],
    dynamic_keys: set[str],
) -> None:
    draw = ImageDraw.Draw(mask)
    width, height = mask.size
    for observation in observations:
        for node in observation.nodes:
            if node.semantic_key not in dynamic_keys:
                continue
            x1, y1, x2, y2 = node.bounds
            box = (
                max(0, int(math.floor(x1))),
                max(0, int(math.floor(y1))),
                min(width - 1, int(math.ceil(x2))),
                min(height - 1, int(math.ceil(y2))),
            )
            if box[2] >= box[0] and box[3] >= box[1]:
                draw.rectangle(box, fill=255)


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def certify_alignment(
    observations: Iterable[FrameObservation],
    policy: AlignmentPolicy = AlignmentPolicy(),
) -> AlignmentArtifact:
    """Build an alignment certificate from a bounded observation burst.

    Acceptance is strict in the stable regions and explicit in the volatile
    regions.  Any failed invariant raises :class:`AlignmentRefused`; callers may
    record that refusal as evidence rather than silently lowering the gate.
    """

    policy.require_valid()
    samples = tuple(observations)
    if len(samples) < policy.minimum_samples:
        raise AlignmentRefused(
            f"need at least {policy.minimum_samples} samples, got {len(samples)}"
        )
    ordered = tuple(sorted(samples, key=lambda item: item.observed_monotonic_ms))
    duration = float(ordered[-1].observed_monotonic_ms - ordered[0].observed_monotonic_ms)
    if duration < policy.minimum_duration_ms:
        raise AlignmentRefused(
            f"observation burst lasted {duration:.1f} ms, below {policy.minimum_duration_ms:.1f} ms"
        )
    if policy.require_event_idle and not all(sample.event_idle for sample in ordered):
        raise AlignmentRefused("source event stream was not idle for every sample")

    images = tuple(_decode_png(bytes(sample.png_bytes)) for sample in ordered)
    dimensions = {image.size for image in images}
    if len(dimensions) != 1:
        raise AlignmentRefused(f"screenshot dimensions changed during capture: {sorted(dimensions)!r}")
    width, height = images[0].size

    dynamic_keys = _infer_dynamic_keys(ordered, policy)
    stable_payload = _stable_structure_payload(ordered, dynamic_keys, policy)
    stable_structure_sha = _sha256(_json_bytes(stable_payload))
    maximum_shift = _max_interactive_shift(ordered, dynamic_keys)
    if maximum_shift > policy.max_interactive_shift_px:
        raise AlignmentRefused(
            f"interactive geometry moved {maximum_shift:.2f}px, above "
            f"{policy.max_interactive_shift_px:.2f}px"
        )

    representative_index = _medoid_index(images)
    representative = images[representative_index]
    mask = Image.new("L", representative.size, 0)
    for image in images:
        mask = ImageChops.lighter(
            mask,
            _binary_change_mask(representative, image, policy.pixel_change_threshold),
        )
    _draw_dynamic_nodes(mask, ordered, dynamic_keys)
    mask = _dilate(mask, policy.mask_dilation_px)
    dynamic_fraction = _mask_fraction(mask)
    if dynamic_fraction > policy.max_dynamic_fraction:
        raise AlignmentRefused(
            f"volatile pixels cover {dynamic_fraction:.3%}, above "
            f"{policy.max_dynamic_fraction:.3%}"
        )

    static_differences = [
        _static_mean_difference(ImageChops.difference(representative, image), mask)
        for image in images
    ]
    static_mean = max(static_differences, default=0.0)
    if static_mean > policy.max_static_mean_difference:
        raise AlignmentRefused(
            f"stable-region mean difference {static_mean:.4f} exceeds "
            f"{policy.max_static_mean_difference:.4f}"
        )

    representative_png = _encode_png(representative)
    mask_png = _encode_png(mask)
    sample_hashes = tuple(_canonical_pixel_sha(image) for image in images)
    policy_dict = asdict(policy)
    payload = {
        "schema": ALIGNMENT_SCHEMA,
        "sample_count": len(ordered),
        "duration_ms": round(duration, 3),
        "dimensions": [width, height],
        "representative_index": representative_index,
        "representative_pixel_sha256": _sha256(representative_png),
        "sample_pixel_sha256": list(sample_hashes),
        "stable_structure_sha256": stable_structure_sha,
        "inferred_dynamic_keys": sorted(dynamic_keys),
        "dynamic_pixel_fraction": round(dynamic_fraction, 8),
        "static_mean_difference": round(static_mean, 8),
        "max_interactive_shift_px": round(maximum_shift, 4),
        "event_idle_all_samples": all(sample.event_idle for sample in ordered),
        "volatility_mask_sha256": _sha256(mask_png),
        "policy": policy_dict,
    }
    certificate_id = "align_" + _sha256(_json_bytes(payload))
    certificate = AlignmentCertificate(
        schema=ALIGNMENT_SCHEMA,
        certificate_id=certificate_id,
        accepted=True,
        reason="stable structure and geometry; volatility localized within policy",
        sample_count=len(ordered),
        duration_ms=round(duration, 3),
        width=width,
        height=height,
        representative_index=representative_index,
        representative_pixel_sha256=_sha256(representative_png),
        sample_pixel_sha256=sample_hashes,
        stable_structure_sha256=stable_structure_sha,
        inferred_dynamic_keys=tuple(sorted(dynamic_keys)),
        dynamic_pixel_fraction=round(dynamic_fraction, 8),
        static_mean_difference=round(static_mean, 8),
        max_interactive_shift_px=round(maximum_shift, 4),
        event_idle_all_samples=all(sample.event_idle for sample in ordered),
        volatility_mask_sha256=_sha256(mask_png),
        policy=policy_dict,
    )
    return AlignmentArtifact(certificate, representative_png, mask_png)


def stage_alignment(artifact: AlignmentArtifact, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "aligned_surface.png": artifact.representative_png,
        "volatility_mask.png": artifact.volatility_mask_png,
        "alignment_certificate.json": _json_bytes(artifact.certificate.to_dict()),
    }
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = {
        "schema": ALIGNMENT_BUNDLE_SCHEMA,
        "certificate_id": artifact.certificate.certificate_id,
        "files": {name: _sha256(data) for name, data in sorted(files.items())},
    }
    (out / "alignment_manifest.json").write_bytes(_json_bytes(manifest))
    return out
