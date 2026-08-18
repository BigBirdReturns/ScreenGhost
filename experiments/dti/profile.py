"""Versioned, coordinate-free DTI surface and wardrobe profile."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from experiments.dti.schema import Bounds, WardrobeItem, normalized_bounds, sha256_json
from experiments.dti.theme_kernel import WardrobeAtlas


PROFILE_SCHEMA = "screenghost_dti_profile_v1"


@dataclass(frozen=True)
class SurfaceAnchor:
    anchor_id: str
    semantic_label: str
    role: str
    station: Optional[str] = None
    visual_prototype_sha256: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.anchor_id or not self.semantic_label or not self.role:
            raise ValueError("anchor_id, semantic_label, and role are required")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class DTIProfile:
    profile_id: str
    app_version: str
    window_title_pattern: str
    client_size: Tuple[int, int]
    observation_regions: Mapping[str, Bounds]
    anchors: Tuple[SurfaceAnchor, ...]
    quick_teleport_labels: Tuple[str, ...]
    wardrobe: WardrobeAtlas
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DTIProfile":
        if value.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"unsupported DTI profile schema: {value.get('schema')!r}")
        client_size = tuple(int(v) for v in value["client_size"])
        if len(client_size) != 2 or min(client_size) <= 0:
            raise ValueError("client_size requires two positive integers")
        regions = {
            str(name): normalized_bounds(bounds)
            for name, bounds in dict(value.get("observation_regions") or {}).items()
        }
        anchors = tuple(
            SurfaceAnchor(
                anchor_id=str(row["anchor_id"]),
                semantic_label=str(row["semantic_label"]),
                role=str(row["role"]),
                station=(str(row["station"]) if row.get("station") is not None else None),
                visual_prototype_sha256=(
                    str(row["visual_prototype_sha256"])
                    if row.get("visual_prototype_sha256") is not None
                    else None
                ),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in value.get("anchors", [])
        )
        wardrobe = WardrobeAtlas.from_records(value.get("wardrobe", []))
        payload = {
            "schema": PROFILE_SCHEMA,
            "app_version": str(value.get("app_version") or "unknown"),
            "window_title_pattern": str(value.get("window_title_pattern") or "Roblox"),
            "client_size": list(client_size),
            "observation_regions": {name: list(bounds) for name, bounds in sorted(regions.items())},
            "anchors": [
                {
                    "anchor_id": row.anchor_id,
                    "semantic_label": row.semantic_label,
                    "role": row.role,
                    "station": row.station,
                    "visual_prototype_sha256": row.visual_prototype_sha256,
                    "metadata": dict(row.metadata),
                }
                for row in anchors
            ],
            "quick_teleport_labels": list(value.get("quick_teleport_labels") or ()),
            "wardrobe": [item.to_dict() for item in wardrobe.items],
            "metadata": dict(value.get("metadata") or {}),
        }
        return cls(
            profile_id="dtiprofile1_" + sha256_json(payload),
            app_version=payload["app_version"],
            window_title_pattern=payload["window_title_pattern"],
            client_size=(client_size[0], client_size[1]),
            observation_regions=regions,
            anchors=anchors,
            quick_teleport_labels=tuple(str(v) for v in payload["quick_teleport_labels"]),
            wardrobe=wardrobe,
            metadata=payload["metadata"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "DTIProfile":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
