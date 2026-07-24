"""Pixel-only generic-control detector used by the deterministic emulator.

This detector is intentionally narrow and explicitly classified as an emulated
student.  It recognizes switch-shaped controls from rendered pixels without
reading the PhoneWorld teacher plane.  Real Android experiments should replace it
with a measured GUI grounder while keeping the same output contract.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image

from experiments.generic_utility.schema import StudentObservation, VisibleElement, sha256_json


@dataclass(frozen=True)
class GenericDetectorReceipt:
    detector: str
    elements_found: int
    pixel_only: bool
    detail: str


class SyntheticPixelDetector:
    """Detect common synthetic switch tracks from pixels only."""

    name = "synthetic_pixel_switch_detector_v1"

    def detect(self, png_bytes: bytes) -> tuple[StudentObservation, GenericDetectorReceipt]:
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.asarray(image, dtype=np.int16)
        height, width, _ = arr.shape
        # Blue accent tracks and medium-neutral off tracks.  The thresholds are
        # intentionally broad enough to survive light/dark theme variants.
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        blue = (b > 150) & (b - r > 45) & (b - g > 15)
        neutral = (np.max(arr, axis=2) - np.min(arr, axis=2) <= 22) & (r >= 85) & (r <= 185)
        mask = blue | neutral
        components = self._components(mask)
        elements: list[VisibleElement] = []
        for x1, y1, x2, y2, count in components:
            box_w, box_h = x2 - x1, y2 - y1
            if box_h < max(12, int(height * 0.018)) or box_h > int(height * 0.09):
                continue
            aspect = box_w / max(1, box_h)
            if not 1.25 <= aspect <= 2.6:
                continue
            if count < box_w * box_h * 0.30:
                continue
            # Ignore status icons near the top edge.
            if y2 < int(height * 0.10):
                continue
            crop = arr[y1:y2, x1:x2]
            white = np.all(crop >= 225, axis=2)
            if not np.any(white):
                continue
            xs = np.where(white)[1]
            checked = float(xs.mean()) > box_w / 2.0
            bounds = (
                round(max(0, x1 - box_h * 4) / width, 4),
                round(max(0, y1 - box_h) / height, 4),
                round(min(width, x2 + box_h // 2) / width, 4),
                round(min(height, y2 + box_h) / height, 4),
            )
            elements.append(
                VisibleElement(
                    element_id=f"generic_switch_{len(elements)}",
                    role="switch",
                    label=None,
                    normalized_bounds=bounds,
                    interactive=True,
                    enabled=True,
                    states={"checked": str(checked).lower()},
                )
            )
        elements.sort(key=lambda el: (el.normalized_bounds[1], el.normalized_bounds[0]))
        observation = StudentObservation(
            observation_id="observation_" + sha256_json(
                {"detector": self.name, "elements": [el.to_dict() for el in elements]}
            ),
            screen_key=None,
            surface_id=None,
            app_family=None,
            confidence=0.82 if elements else 0.0,
            unknown=not bool(elements),
            elements=tuple(elements),
            evidence_sources=("pixels", "small_grounder"),
            explanation="generic pixel-only switch detector",
            match_detail={"detector": self.name},
        )
        return observation, GenericDetectorReceipt(
            detector=self.name,
            elements_found=len(elements),
            pixel_only=True,
            detail="emulator-only shape detector; replace with a measured GUI grounder for live Android",
        )

    @staticmethod
    def _components(mask: np.ndarray) -> list[Tuple[int, int, int, int, int]]:
        height, width = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        out: list[Tuple[int, int, int, int, int]] = []
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or seen[y, x]:
                    continue
                stack = [(x, y)]
                seen[y, x] = True
                xs: list[int] = []
                ys: list[int] = []
                while stack:
                    cx, cy = stack.pop()
                    xs.append(cx)
                    ys.append(cy)
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
                out.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(xs)))
        return out
