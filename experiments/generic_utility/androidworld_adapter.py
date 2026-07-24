"""Optional AndroidWorld bridge for the Generic Utility campaign.

AndroidWorld already exposes a synchronized state containing pixels, an
accessibility forest, and processed UI elements.  This module adapts that state to
Surface Teacher and the transactional motor contract without importing
AndroidWorld until an actual run requests it.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from core.surface_alignment import AlignmentNode
from core.surface_temporal_teacher import StructureSnapshot
from experiments.generic_utility.schema import VisibleElement, json_bytes, sha256_bytes, sha256_json


class AndroidWorldUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class AndroidWorldMotorResult:
    accepted: bool
    injected: bool
    target_key: Optional[str]
    target_label: Optional[str]
    reason: str


class AndroidWorldBackend:
    """Duck-typed adapter around ``android_world.env.interface.AsyncEnv``."""

    def __init__(self, env: Any, *, surface_id: Optional[str] = None) -> None:
        self.env = env
        self.surface_id = surface_id
        self.teacher_reads = 0
        self.pixel_reads = 0
        self.actions_injected = 0
        self.focus_change_count = 0
        self._last_activity: Optional[str] = None
        self._cached_state: Any = None

    @classmethod
    def launch(
        cls,
        *,
        adb_path: str,
        console_port: int = 5554,
        perform_emulator_setup: bool = False,
    ) -> "AndroidWorldBackend":
        try:
            from android_world.env import env_launcher
        except Exception as exc:
            raise AndroidWorldUnavailable(
                "AndroidWorld is not installed. Run the supplied bootstrap script or install google-research/android_world."
            ) from exc
        env = env_launcher.load_and_setup_env(
            console_port=int(console_port),
            emulator_setup=bool(perform_emulator_setup),
            adb_path=str(adb_path),
        )
        return cls(env)

    @property
    def tick_ms(self) -> float:
        return time.monotonic() * 1000.0

    def reset(self, *, go_home: bool = True) -> Any:
        self._cached_state = self.env.reset(go_home=go_home)
        self._update_focus()
        return self._cached_state

    def close(self) -> None:
        self.env.close()

    def capture_png(self) -> bytes:
        self.pixel_reads += 1
        state = self.env.get_state(wait_to_stabilize=False)
        self._cached_state = state
        self._update_focus()
        return self._pixels_to_png(state.pixels)

    def stable_state(self) -> Any:
        state = self.env.get_state(wait_to_stabilize=True)
        self._cached_state = state
        self._update_focus()
        return state

    def teacher_snapshot(self) -> tuple[bytes, Mapping[str, Any], Tuple[AlignmentNode, ...]]:
        self.teacher_reads += 1
        state = self.stable_state()
        png = self._pixels_to_png(state.pixels)
        projection, nodes = self._projection_from_state(state, png)
        return png, projection, nodes

    def temporal_source(self) -> "AndroidWorldTemporalSource":
        return AndroidWorldTemporalSource(self)

    def tap_normalized(self, x: float, y: float) -> AndroidWorldMotorResult:
        try:
            from android_world.env import json_action
        except Exception as exc:
            raise AndroidWorldUnavailable("AndroidWorld json_action is unavailable") from exc
        width, height = self.env.logical_screen_size
        action = json_action.JSONAction(
            action_type=json_action.CLICK,
            x=int(round(float(x) * width)),
            y=int(round(float(y) * height)),
        )
        self.env.execute_action(action)
        self.actions_injected += 1
        return AndroidWorldMotorResult(True, True, None, None, "AndroidWorld click injected")

    def type_text(self, text: str) -> AndroidWorldMotorResult:
        try:
            from android_world.env import json_action
        except Exception as exc:
            raise AndroidWorldUnavailable("AndroidWorld json_action is unavailable") from exc
        self.env.execute_action(
            json_action.JSONAction(
                action_type=json_action.INPUT_TEXT,
                text=str(text),
                clear_text=True,
            )
        )
        self.actions_injected += 1
        return AndroidWorldMotorResult(True, True, None, None, "AndroidWorld text injected")

    def back(self) -> AndroidWorldMotorResult:
        try:
            from android_world.env import json_action
        except Exception as exc:
            raise AndroidWorldUnavailable("AndroidWorld json_action is unavailable") from exc
        self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
        self.actions_injected += 1
        return AndroidWorldMotorResult(True, True, None, None, "AndroidWorld back injected")

    @staticmethod
    def advance(milliseconds: float) -> None:
        time.sleep(max(0.0, float(milliseconds)) / 1000.0)

    def _update_focus(self) -> None:
        try:
            activity = str(self.env.foreground_activity_name or "")
        except Exception:
            return
        if self._last_activity is not None and activity != self._last_activity:
            self.focus_change_count += 1
        self._last_activity = activity

    @staticmethod
    def _pixels_to_png(pixels: Any) -> bytes:
        arr = np.asarray(pixels)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"AndroidWorld pixels have unexpected shape: {arr.shape}")
        arr = np.asarray(np.clip(arr[:, :, :3], 0, 255), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=9)
        return buffer.getvalue()

    def _projection_from_state(
        self, state: Any, png: bytes
    ) -> tuple[Mapping[str, Any], Tuple[AlignmentNode, ...]]:
        height, width = np.asarray(state.pixels).shape[:2]
        activity = ""
        try:
            activity = str(self.env.foreground_activity_name or "")
        except Exception:
            pass
        surface_id = self.surface_id or activity or "androidworld.unknown"
        elements: list[dict[str, Any]] = []
        nodes: list[AlignmentNode] = []
        grammar_terms = []
        control_terms = []
        content_terms = []
        for index, ui in enumerate(getattr(state, "ui_elements", ())):
            bbox = getattr(ui, "bbox_pixels", None)
            if bbox is None:
                continue
            x1, x2 = float(bbox.x_min), float(bbox.x_max)
            y1, y2 = float(bbox.y_min), float(bbox.y_max)
            if x2 <= x1 or y2 <= y1:
                continue
            label = (
                getattr(ui, "text", None)
                or getattr(ui, "content_description", None)
                or getattr(ui, "hint_text", None)
            )
            class_name = str(getattr(ui, "class_name", None) or "")
            role = self._role(class_name, ui)
            interactive = bool(
                getattr(ui, "is_clickable", False)
                or getattr(ui, "is_editable", False)
                or getattr(ui, "is_checkable", False)
                or getattr(ui, "is_scrollable", False)
            )
            key = str(
                getattr(ui, "resource_name", None)
                or getattr(ui, "resource_id", None)
                or f"aw:{index}:{class_name}"
            )
            states = {
                "checked": str(bool(getattr(ui, "is_checked", False))).lower(),
                "selected": str(bool(getattr(ui, "is_selected", False))).lower(),
                "focused": str(bool(getattr(ui, "is_focused", False))).lower(),
                "editable": str(bool(getattr(ui, "is_editable", False))).lower(),
                "scrollable": str(bool(getattr(ui, "is_scrollable", False))).lower(),
            }
            norm = (
                round(max(0.0, x1) / width, 4),
                round(max(0.0, y1) / height, 4),
                round(min(float(width), x2) / width, 4),
                round(min(float(height), y2) / height, 4),
            )
            if not (norm[0] < norm[2] and norm[1] < norm[3]):
                continue
            element_id = "el_" + sha256_json(
                {"role": role, "bounds": list(norm), "interactive": interactive, "ordinal": index}
            )[:20]
            elements.append(
                VisibleElement(
                    element_id=element_id,
                    role=role,
                    label=str(label) if label else None,
                    normalized_bounds=norm,
                    interactive=interactive,
                    enabled=bool(getattr(ui, "is_enabled", True)),
                    states=states,
                ).to_dict()
            )
            nodes.append(
                AlignmentNode(
                    semantic_key=key,
                    role=role,
                    bounds=(x1, y1, x2, y2),
                    label=str(label) if label else None,
                    interactive=interactive,
                    enabled=bool(getattr(ui, "is_enabled", True)),
                    states=tuple(states.items()),
                    dynamic=False,
                )
            )
            grammar_terms.append({"role": role, "bounds": list(norm), "interactive": interactive})
            control_terms.append(
                {
                    "role": role,
                    "label": (str(label).casefold() if label and interactive else None),
                    "bounds": list(norm),
                    "interactive": interactive,
                }
            )
            content_terms.append({"role": role, "label": label, "states": states})
        grammar_hash = sha256_json(grammar_terms)
        control_hash = sha256_json(control_terms)
        content_hash = sha256_json(content_terms)
        screen_key = "screen_" + sha256_json(
            {"surface_id": surface_id, "grammar_hash": grammar_hash, "control_hash": control_hash}
        )[:24]
        projection = {
            "schema": "surface_teacher_runtime_v0",
            "lesson_id": "lesson_" + sha256_json(
                {"screen_key": screen_key, "content_hash": content_hash, "png": sha256_bytes(png)}
            ),
            "surface_id": surface_id,
            "app_family": surface_id.split("/", 1)[0],
            "app_version": None,
            "screen_name": activity,
            "screen_key": screen_key,
            "grammar_hash": grammar_hash,
            "control_hash": control_hash,
            "content_hash": content_hash,
            "explanation": f"AndroidWorld surface {surface_id}",
            "width": width,
            "height": height,
            "elements": elements,
            "provenance": {
                "compiled_from": "androidworld_accessibility",
                "runtime_visibility": "teacher_blind",
                "privileged_fields_removed": True,
            },
        }
        return projection, tuple(nodes)

    @staticmethod
    def _role(class_name: str, ui: Any) -> str:
        tail = class_name.rsplit(".", 1)[-1].lower()
        if bool(getattr(ui, "is_editable", False)) or "edittext" in tail:
            return "text_field"
        if bool(getattr(ui, "is_checkable", False)):
            if "radio" in tail:
                return "radio"
            if "checkbox" in tail:
                return "checkbox"
            return "switch"
        if bool(getattr(ui, "is_scrollable", False)) or tail in {"recyclerview", "listview"}:
            return "list"
        if "button" in tail or bool(getattr(ui, "is_clickable", False)):
            return "button"
        if "image" in tail:
            return "image"
        return "text"


class AndroidWorldTemporalSource:
    def __init__(self, backend: AndroidWorldBackend) -> None:
        self.backend = backend
        self._latest: Any = None

    def capture_png(self) -> bytes:
        return self.backend.capture_png()

    def inspect_structure(self) -> StructureSnapshot:
        png, projection, nodes = self.backend.teacher_snapshot()
        digest = sha256_json(
            {
                "surface_id": projection["surface_id"],
                "screen_key": projection["screen_key"],
                "content_hash": projection["content_hash"],
            }
        )
        return StructureSnapshot(
            source_digest=digest,
            alignment_nodes=nodes,
            compiler_nodes=projection,
            event_idle=True,
            source_payload={"surface_id": projection["surface_id"]},
        )
