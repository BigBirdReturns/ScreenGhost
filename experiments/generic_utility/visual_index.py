"""Teacher-blind visual state index for warm ScreenGhost operation.

The index is deliberately model-free.  It stores distributions of taught visual
variants, stable-region masks, and element crop prototypes.  Querying uses only a
fresh screenshot and returns either a calibrated screen-family match or an
explicit unknown result.  A small embedding or GUI grounder can be added later;
known screens should normally stop here.
"""
from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from experiments.generic_utility.schema import (
    StudentObservation,
    VisibleElement,
    clean_text,
    json_bytes,
    sha256_bytes,
    sha256_json,
)


INDEX_SCHEMA = "screenghost_visual_state_index_v1"


class VisualIndexError(ValueError):
    pass


def _decode_png(value: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(value))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise VisualIndexError(f"not a decodable PNG: {exc}") from exc


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _dhash(image: Image.Image, size: int = 8) -> int:
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return int(value)


def _hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def _feature(image: Image.Image, size: Tuple[int, int] = (48, 96)) -> bytes:
    gray = image.convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.uint8).tobytes()


def _feature_array(value: str, size: Tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(value.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    expected = size[0] * size[1]
    if arr.size != expected:
        raise VisualIndexError(f"feature byte length {arr.size} != expected {expected}")
    return arr.reshape((size[1], size[0]))


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _mask_array(mask_png: Optional[bytes], size: Tuple[int, int]) -> np.ndarray:
    if mask_png is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    mask = _decode_png(mask_png).convert("L").resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) >= 128


def _bounds(value: Sequence[Any]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise VisualIndexError("element bounds require four values")
    vals = tuple(float(v) for v in value)
    x1, y1, x2, y2 = vals
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise VisualIndexError(f"invalid normalized bounds: {vals!r}")
    return vals


def _crop(image: Image.Image, bounds: Sequence[Any]) -> Image.Image:
    x1, y1, x2, y2 = _bounds(bounds)
    width, height = image.size
    box = (
        max(0, int(math.floor(x1 * width))),
        max(0, int(math.floor(y1 * height))),
        min(width, int(math.ceil(x2 * width))),
        min(height, int(math.ceil(y2 * height))),
    )
    return image.crop(box)


def _crop_feature(image: Image.Image, bounds: Sequence[Any], size: Tuple[int, int] = (72, 36)) -> bytes:
    return _feature(_crop(image, bounds), size=size)


def _mae_score(left: np.ndarray, right: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if left.shape != right.shape:
        return 0.0
    diff = np.abs(left.astype(np.int16) - right.astype(np.int16))
    if mask is not None:
        stable = ~mask
        if not np.any(stable):
            return 0.0
        mean = float(diff[stable].mean())
    else:
        mean = float(diff.mean())
    return max(0.0, min(1.0, 1.0 - mean / 255.0))


@dataclass(frozen=True)
class VisualIndexPolicy:
    feature_size: Tuple[int, int] = (48, 96)
    crop_feature_size: Tuple[int, int] = (72, 36)
    minimum_confidence: float = 0.90
    minimum_margin: float = 0.025
    minimum_crop_confidence: float = 0.0
    exact_pixel_bypasses_margin: bool = True
    dhash_weight: float = 0.10
    stable_feature_weight: float = 0.35
    crop_weight: float = 0.55
    maximum_variants_per_family: int = 12

    def require_valid(self) -> None:
        if not 0 <= self.minimum_confidence <= 1:
            raise VisualIndexError("minimum_confidence must be in [0,1]")
        if not 0 <= self.minimum_margin <= 1:
            raise VisualIndexError("minimum_margin must be in [0,1]")
        if not 0 <= self.minimum_crop_confidence <= 1:
            raise VisualIndexError("minimum_crop_confidence must be in [0,1]")
        weights = self.dhash_weight + self.stable_feature_weight + self.crop_weight
        if abs(weights - 1.0) > 1e-9:
            raise VisualIndexError("visual index weights must sum to 1")
        if self.maximum_variants_per_family < 1:
            raise VisualIndexError("maximum_variants_per_family must be positive")


@dataclass(frozen=True)
class CropPrototype:
    element_id: str
    role: str
    label: Optional[str]
    normalized_bounds: Tuple[float, float, float, float]
    feature_b64: str


@dataclass(frozen=True)
class VisualVariant:
    variant_id: str
    family_id: str
    screen_key: str
    surface_id: str
    app_family: Optional[str]
    app_version: Optional[str]
    width: int
    height: int
    pixel_sha256: str
    dhash_hex: str
    feature_b64: str
    dynamic_mask_sha256: Optional[str]
    dynamic_mask_b64: Optional[str]
    projection: Mapping[str, Any]
    crops: Tuple[CropPrototype, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["projection"] = dict(self.projection)
        value["metadata"] = dict(self.metadata)
        value["crops"] = [asdict(c) for c in self.crops]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualVariant":
        return cls(
            variant_id=str(value["variant_id"]),
            family_id=str(value["family_id"]),
            screen_key=str(value["screen_key"]),
            surface_id=str(value["surface_id"]),
            app_family=(str(value["app_family"]) if value.get("app_family") is not None else None),
            app_version=(str(value["app_version"]) if value.get("app_version") is not None else None),
            width=int(value["width"]),
            height=int(value["height"]),
            pixel_sha256=str(value["pixel_sha256"]),
            dhash_hex=str(value["dhash_hex"]),
            feature_b64=str(value["feature_b64"]),
            dynamic_mask_sha256=(
                str(value["dynamic_mask_sha256"])
                if value.get("dynamic_mask_sha256") is not None
                else None
            ),
            dynamic_mask_b64=(
                str(value["dynamic_mask_b64"])
                if value.get("dynamic_mask_b64") is not None
                else None
            ),
            projection=dict(value["projection"]),
            crops=tuple(CropPrototype(**row) for row in value.get("crops", [])),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class VariantScore:
    variant_id: str
    family_id: str
    screen_key: str
    total: float
    dhash: float
    stable_feature: float
    crop: float


@dataclass(frozen=True)
class VisualMatch:
    known: bool
    confidence: float
    margin: float
    family_id: Optional[str]
    screen_key: Optional[str]
    surface_id: Optional[str]
    app_family: Optional[str]
    projection: Optional[Mapping[str, Any]]
    best_variant_id: Optional[str]
    scores: Tuple[VariantScore, ...]
    reason: str

    def to_student_observation(self) -> StudentObservation:
        if not self.known or self.projection is None:
            return StudentObservation(
                observation_id="observation_" + sha256_json(
                    {"known": False, "scores": [asdict(score) for score in self.scores]}
                ),
                screen_key=None,
                surface_id=None,
                app_family=None,
                confidence=self.confidence,
                unknown=True,
                elements=(),
                evidence_sources=("pixels", "visual_index"),
                explanation=self.reason,
                match_detail={
                    "margin": self.margin,
                    "best_variant_id": self.best_variant_id,
                    "scores": [asdict(score) for score in self.scores[:5]],
                },
            )
        elements = tuple(VisibleElement.from_mapping(row) for row in self.projection.get("elements", []))
        return StudentObservation(
            observation_id="observation_" + sha256_json(
                {
                    "screen_key": self.screen_key,
                    "variant": self.best_variant_id,
                    "confidence": round(self.confidence, 6),
                }
            ),
            screen_key=self.screen_key,
            surface_id=self.surface_id,
            app_family=self.app_family,
            confidence=self.confidence,
            unknown=False,
            elements=elements,
            evidence_sources=("pixels", "visual_index"),
            explanation=clean_text(self.projection.get("explanation")),
            match_detail={
                "margin": self.margin,
                "best_variant_id": self.best_variant_id,
                "scores": [asdict(score) for score in self.scores[:5]],
                "screen_name": self.projection.get("screen_name"),
                "app_version": self.projection.get("app_version"),
            },
        )


class VisualStateIndex:
    def __init__(self, path: Optional[str | Path] = None, *, policy: VisualIndexPolicy = VisualIndexPolicy()):
        policy.require_valid()
        self.path = Path(path) if path is not None else None
        self.policy = policy
        self._variants: list[VisualVariant] = []
        if self.path is not None and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != INDEX_SCHEMA:
                raise VisualIndexError(f"unsupported index schema: {payload.get('schema')!r}")
            self._variants = [VisualVariant.from_dict(row) for row in payload.get("variants", [])]

    @property
    def variants(self) -> Tuple[VisualVariant, ...]:
        return tuple(self._variants)

    @property
    def family_count(self) -> int:
        return len({variant.family_id for variant in self._variants})

    def add(
        self,
        png_bytes: bytes,
        projection: Mapping[str, Any],
        *,
        dynamic_mask_png: Optional[bytes] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        family_id: Optional[str] = None,
    ) -> VisualVariant:
        image = _decode_png(png_bytes)
        width, height = image.size
        required = ("screen_key", "surface_id", "elements")
        missing = [name for name in required if projection.get(name) is None]
        if missing:
            raise VisualIndexError(f"projection missing required fields: {missing!r}")
        family_id = family_id or str(projection["screen_key"])
        feature = _feature(image, self.policy.feature_size)
        mask_b64 = _b64(dynamic_mask_png) if dynamic_mask_png is not None else None
        crops = []
        for row in projection.get("elements", []):
            if not row.get("interactive") or row.get("sensitive"):
                continue
            bounds = _bounds(row["normalized_bounds"])
            crop_feature = _crop_feature(image, bounds, self.policy.crop_feature_size)
            crops.append(
                CropPrototype(
                    element_id=str(row["element_id"]),
                    role=str(row.get("role") or "unknown"),
                    label=clean_text(row.get("label")),
                    normalized_bounds=bounds,
                    feature_b64=_b64(crop_feature),
                )
            )
        payload = {
            "family_id": family_id,
            "screen_key": str(projection["screen_key"]),
            "surface_id": str(projection["surface_id"]),
            "pixel_sha256": sha256_bytes(png_bytes),
            "metadata": dict(metadata or {}),
            "projection_content_hash": projection.get("content_hash"),
        }
        variant_id = "visual_variant_" + sha256_json(payload)
        variant = VisualVariant(
            variant_id=variant_id,
            family_id=family_id,
            screen_key=str(projection["screen_key"]),
            surface_id=str(projection["surface_id"]),
            app_family=(str(projection["app_family"]) if projection.get("app_family") is not None else None),
            app_version=(str(projection["app_version"]) if projection.get("app_version") is not None else None),
            width=width,
            height=height,
            pixel_sha256=sha256_bytes(png_bytes),
            dhash_hex=f"{_dhash(image):016x}",
            feature_b64=_b64(feature),
            dynamic_mask_sha256=(sha256_bytes(dynamic_mask_png) if dynamic_mask_png is not None else None),
            dynamic_mask_b64=mask_b64,
            projection=dict(projection),
            crops=tuple(crops),
            metadata=dict(metadata or {}),
        )
        self._variants = [existing for existing in self._variants if existing.variant_id != variant_id]
        self._variants.append(variant)
        same_family = [row for row in self._variants if row.family_id == family_id]
        if len(same_family) > self.policy.maximum_variants_per_family:
            keep_ids = {row.variant_id for row in same_family[-self.policy.maximum_variants_per_family :]}
            self._variants = [row for row in self._variants if row.family_id != family_id or row.variant_id in keep_ids]
        self._variants.sort(key=lambda row: (row.surface_id, row.family_id, row.variant_id))
        self._persist()
        return variant

    def match(
        self,
        png_bytes: bytes,
        *,
        surface_hint: Optional[str] = None,
        app_family_hint: Optional[str] = None,
        screen_name_hint: Optional[str] = None,
        screen_key_hint: Optional[str] = None,
        target_label_hint: Optional[str] = None,
        target_role_hint: Optional[str] = None,
        state_key_hint: Optional[str] = None,
        state_value_hint: Optional[str] = None,
    ) -> VisualMatch:
        if not self._variants:
            return VisualMatch(False, 0.0, 0.0, None, None, None, None, None, None, (), "visual index is empty")
        image = _decode_png(png_bytes)
        query_feature = np.frombuffer(_feature(image, self.policy.feature_size), dtype=np.uint8).reshape(
            (self.policy.feature_size[1], self.policy.feature_size[0])
        )
        query_dhash = _dhash(image)
        scores: list[VariantScore] = []
        for variant in self._variants:
            if surface_hint is not None and variant.surface_id != surface_hint:
                continue
            if app_family_hint is not None and variant.app_family != app_family_hint:
                continue
            if screen_name_hint is not None and clean_text(variant.projection.get("screen_name")) != clean_text(screen_name_hint):
                continue
            if screen_key_hint is not None and variant.screen_key != screen_key_hint:
                continue
            if state_key_hint is not None:
                expected_state = str(state_value_hint or "").casefold()
                matching_state = False
                for element in variant.projection.get("elements", []):
                    if target_label_hint is not None and clean_text(element.get("label")) != clean_text(target_label_hint):
                        continue
                    if target_role_hint is not None and str(element.get("role") or "") != str(target_role_hint):
                        continue
                    actual_state = str((element.get("states") or {}).get(state_key_hint) or "").casefold()
                    if actual_state == expected_state:
                        matching_state = True
                        break
                if not matching_state:
                    continue
            if (variant.width, variant.height) != image.size:
                # Orientation/density variants should be taught explicitly.  Aspect
                # mismatch is an unknown, not a resize-and-hope path.
                continue
            stored_feature = _feature_array(variant.feature_b64, self.policy.feature_size)
            mask_png = base64.b64decode(variant.dynamic_mask_b64) if variant.dynamic_mask_b64 else None
            dynamic_mask = _mask_array(mask_png, self.policy.feature_size)
            stable_score = _mae_score(query_feature, stored_feature, dynamic_mask)
            dhash_score = 1.0 - _hamming(query_dhash, int(variant.dhash_hex, 16)) / 64.0
            crop_scores = []
            for prototype in variant.crops:
                query_crop = np.frombuffer(
                    _crop_feature(image, prototype.normalized_bounds, self.policy.crop_feature_size),
                    dtype=np.uint8,
                ).reshape((self.policy.crop_feature_size[1], self.policy.crop_feature_size[0]))
                stored_crop = _feature_array(prototype.feature_b64, self.policy.crop_feature_size)
                crop_scores.append(_mae_score(query_crop, stored_crop))
            crop_score = float(np.mean(crop_scores)) if crop_scores else stable_score
            total = (
                self.policy.dhash_weight * dhash_score
                + self.policy.stable_feature_weight * stable_score
                + self.policy.crop_weight * crop_score
            )
            scores.append(
                VariantScore(
                    variant.variant_id,
                    variant.family_id,
                    variant.screen_key,
                    round(total, 6),
                    round(dhash_score, 6),
                    round(stable_score, 6),
                    round(crop_score, 6),
                )
            )
        if not scores:
            return VisualMatch(
                False,
                0.0,
                0.0,
                None,
                None,
                None,
                None,
                None,
                None,
                (),
                "no index variant matches viewport or declared transition hints",
            )
        scores.sort(key=lambda row: (-row.total, row.variant_id))
        best = scores[0]
        second = next((row for row in scores[1:] if row.family_id != best.family_id), None)
        margin = best.total - (second.total if second is not None else 0.0)
        variant = next(row for row in self._variants if row.variant_id == best.variant_id)
        query_pixel_sha256 = sha256_bytes(png_bytes)
        exact_families = {
            row.family_id
            for row in self._variants
            if row.pixel_sha256 == query_pixel_sha256
            and row.width == image.size[0]
            and row.height == image.size[1]
            and (surface_hint is None or row.surface_id == surface_hint)
            and (app_family_hint is None or row.app_family == app_family_hint)
            and (screen_name_hint is None or clean_text(row.projection.get("screen_name")) == clean_text(screen_name_hint))
            and (screen_key_hint is None or row.screen_key == screen_key_hint)
        }
        exact_unambiguous = (
            self.policy.exact_pixel_bypasses_margin
            and variant.pixel_sha256 == query_pixel_sha256
            and exact_families == {best.family_id}
        )
        known = (
            best.total >= self.policy.minimum_confidence
            and best.crop >= self.policy.minimum_crop_confidence
            and (margin >= self.policy.minimum_margin or exact_unambiguous)
        )
        if not known:
            return VisualMatch(
                False,
                best.total,
                margin,
                None,
                None,
                None,
                None,
                None,
                best.variant_id,
                tuple(scores),
                f"best visual match below confidence, crop, or margin gate: confidence={best.total:.4f}, crop={best.crop:.4f}, margin={margin:.4f}",
            )
        return VisualMatch(
            True,
            best.total,
            margin,
            variant.family_id,
            variant.screen_key,
            variant.surface_id,
            variant.app_family,
            dict(variant.projection),
            variant.variant_id,
            tuple(scores),
            "known visual family matched",
        )

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": INDEX_SCHEMA,
            "policy": asdict(self.policy),
            "variants": [variant.to_dict() for variant in self._variants],
        }
        self.path.write_bytes(json_bytes(payload))
