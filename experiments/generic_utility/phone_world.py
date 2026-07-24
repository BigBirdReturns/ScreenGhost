"""Deterministic phone-shaped world for cold-to-warm experiments.

This is not presented as Android hardware.  It is a reproducible laboratory that
renders multiple app families into pixels, exposes a hidden teacher plane, accepts
ordinary touch/text/back actions, models asynchronous settlement, and can inject
theme, density, font, content, layout, look-alike, and unknown-screen variation.

The student-facing API is pixels plus ordinary actions.  Teacher nodes are only
available through :meth:`teacher_snapshot` and every call is counted.
"""
from __future__ import annotations

import io
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from core.surface_alignment import AlignmentNode
from experiments.generic_utility.schema import (
    Bounds,
    Operator,
    SemanticGoal,
    VisibleElement,
    clean_text,
    json_bytes,
    sha256_bytes,
    sha256_json,
)


class PhoneWorldError(RuntimeError):
    pass


@dataclass(frozen=True)
class DisplayVariant:
    theme: str = "light"
    font_scale: float = 1.0
    density: float = 1.0
    orientation: str = "portrait"
    app_version: str = "1.0"
    variant_id: str = "default"
    move_controls: bool = False
    rename_control: bool = False
    overlay_target: bool = False

    def __post_init__(self) -> None:
        if self.theme not in {"light", "dark"}:
            raise ValueError("theme must be light or dark")
        if not 0.75 <= float(self.font_scale) <= 1.5:
            raise ValueError("font_scale must be in [0.75,1.5]")
        if not 0.75 <= float(self.density) <= 1.5:
            raise ValueError("density must be in [0.75,1.5]")
        if self.orientation not in {"portrait", "landscape"}:
            raise ValueError("orientation must be portrait or landscape")


@dataclass(frozen=True)
class WorldNode:
    key: str
    role: str
    bounds_px: Tuple[int, int, int, int]
    label: Optional[str] = None
    interactive: bool = False
    enabled: bool = True
    states: Mapping[str, str] = field(default_factory=dict)
    parent_key: Optional[str] = None
    dynamic: bool = False
    sensitive: bool = False

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bounds_px
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid node bounds: {self.bounds_px!r}")
        object.__setattr__(self, "states", dict(sorted((str(k), str(v)) for k, v in self.states.items())))

    def normalized(self, width: int, height: int) -> Bounds:
        x1, y1, x2, y2 = self.bounds_px
        return (
            round(x1 / width, 4),
            round(y1 / height, 4),
            round(x2 / width, 4),
            round(y2 / height, 4),
        )

    def alignment_node(self) -> AlignmentNode:
        return AlignmentNode(
            semantic_key=self.key,
            role=self.role,
            bounds=tuple(float(v) for v in self.bounds_px),
            label=self.label,
            interactive=self.interactive,
            enabled=self.enabled,
            parent_key=self.parent_key,
            states=tuple(self.states.items()),
            dynamic=self.dynamic,
        )


@dataclass(frozen=True)
class WorldFrame:
    png_bytes: bytes
    width: int
    height: int
    app_family: str
    surface_id: str
    screen_name: str
    screen_key: str
    nodes: Tuple[WorldNode, ...]
    runtime_projection: Mapping[str, Any]
    teacher_payload_sha256: str
    tick_ms: float


@dataclass(frozen=True)
class WorldActionResult:
    accepted: bool
    injected: bool
    action_type: str
    target_key: Optional[str]
    target_label: Optional[str]
    transition_due_ms: Optional[float]
    reason: str


@dataclass
class _PendingTransition:
    due_ms: float
    callback_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldTask:
    task_id: str
    app_family: str
    start_screen: str
    goals: Tuple[SemanticGoal, ...]
    success_screen: str
    description: str


class PhoneWorld:
    """A deterministic, phone-shaped state machine with a hidden teacher plane."""

    BASE_PORTRAIT = (360, 720)
    BASE_LANDSCAPE = (720, 360)

    def __init__(
        self,
        *,
        seed: int = 7,
        variant: DisplayVariant = DisplayVariant(),
        start_app: str = "launcher",
        start_screen: str = "home",
    ) -> None:
        self.seed = int(seed)
        self.random = random.Random(self.seed)
        self.variant = variant
        self.app_family = start_app
        self.screen_name = start_screen
        self.tick_ms = 0.0
        self._pending: Optional[_PendingTransition] = None
        self._input_buffer = ""
        self._dark_mode = False
        self._profile_name = ""
        self._timer_running = False
        self._timer_started_ms = 0.0
        self._holdout_enabled = False
        self._toast_until_ms = 0.0
        self.teacher_reads = 0
        self.pixel_reads = 0
        self.actions_injected = 0
        self.action_log: list[dict[str, Any]] = []
        self.focus_change_count = 0
        self._last_nodes: Tuple[WorldNode, ...] = ()

    @property
    def size(self) -> Tuple[int, int]:
        base = self.BASE_PORTRAIT if self.variant.orientation == "portrait" else self.BASE_LANDSCAPE
        return tuple(max(1, int(round(v * self.variant.density))) for v in base)

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "variant": self.variant.__dict__,
            "app_family": self.app_family,
            "screen_name": self.screen_name,
            "tick_ms": self.tick_ms,
            "input_buffer": self._input_buffer,
            "dark_mode": self._dark_mode,
            "profile_name": self._profile_name,
            "timer_running": self._timer_running,
            "timer_started_ms": self._timer_started_ms,
            "holdout_enabled": self._holdout_enabled,
            "toast_until_ms": self._toast_until_ms,
        }

    def restore_state(self, value: Mapping[str, Any]) -> None:
        self.app_family = str(value["app_family"])
        self.screen_name = str(value["screen_name"])
        self.tick_ms = float(value.get("tick_ms", 0.0))
        self._input_buffer = str(value.get("input_buffer", ""))
        self._dark_mode = bool(value.get("dark_mode", False))
        self._profile_name = str(value.get("profile_name", ""))
        self._timer_running = bool(value.get("timer_running", False))
        self._timer_started_ms = float(value.get("timer_started_ms", 0.0))
        self._holdout_enabled = bool(value.get("holdout_enabled", False))
        self._toast_until_ms = float(value.get("toast_until_ms", 0.0))
        self._pending = None

    def reset(
        self,
        *,
        app_family: Optional[str] = None,
        screen_name: Optional[str] = None,
        preserve_preferences: bool = False,
    ) -> None:
        if not preserve_preferences:
            self._dark_mode = False
            self._profile_name = ""
            self._holdout_enabled = False
        self._timer_running = False
        self._timer_started_ms = 0.0
        self._input_buffer = ""
        self._toast_until_ms = 0.0
        self._pending = None
        self.tick_ms = 0.0
        self.app_family = app_family or "launcher"
        self.screen_name = screen_name or "home"
        self.action_log.clear()
        self.actions_injected = 0
        self.teacher_reads = 0
        self.pixel_reads = 0
        self.focus_change_count = 0

    def set_variant(self, variant: DisplayVariant) -> None:
        self.variant = variant

    def advance(self, milliseconds: float) -> None:
        if milliseconds < 0:
            raise ValueError("cannot advance time backwards")
        self.tick_ms += float(milliseconds)
        if self._pending is not None and self.tick_ms >= self._pending.due_ms:
            pending = self._pending
            self._pending = None
            callback = getattr(self, pending.callback_name, None)
            if callback is None:
                raise PhoneWorldError(f"unknown pending callback: {pending.callback_name}")
            callback(**pending.metadata)

    # ------------------------------------------------------------------
    # Task catalog
    # ------------------------------------------------------------------
    @staticmethod
    def task_catalog() -> dict[str, WorldTask]:
        return {
            "settings_dark_mode": WorldTask(
                task_id="settings_dark_mode",
                app_family="settings",
                start_screen="home",
                goals=(
                    SemanticGoal("open-display", Operator.OPEN, target_label="Display", target_role="button", expected_screen="display"),
                    SemanticGoal(
                        "toggle-dark-mode",
                        Operator.TOGGLE,
                        target_label="Dark mode",
                        target_role="switch",
                        expected_state_key="checked",
                        expected_state_value="true",
                    ),
                    SemanticGoal("save-display", Operator.ACTIVATE, target_label="Save", target_role="button", expected_screen="saved"),
                ),
                success_screen="saved",
                description="Open Display, enable Dark mode, and save.",
            ),
            "profile_display_name": WorldTask(
                task_id="profile_display_name",
                app_family="profile",
                start_screen="home",
                goals=(
                    SemanticGoal("open-edit-profile", Operator.OPEN, target_label="Edit profile", target_role="button", expected_screen="edit"),
                    SemanticGoal("fill-display-name", Operator.FILL, target_label="Display name", target_role="text_field", value="Screen Ghost", expected_state_key="filled", expected_state_value="true"),
                    SemanticGoal("save-profile", Operator.ACTIVATE, target_label="Save profile", target_role="button", expected_screen="saved"),
                ),
                success_screen="saved",
                description="Edit the display name and save the profile.",
            ),
            "timer_start_stop": WorldTask(
                task_id="timer_start_stop",
                app_family="timer",
                start_screen="home",
                goals=(
                    SemanticGoal("start-timer", Operator.START, target_label="Start", target_role="button", expected_screen="running"),
                    SemanticGoal("stop-timer", Operator.STOP, target_label="Stop", target_role="button", expected_screen="stopped"),
                ),
                success_screen="stopped",
                description="Start and stop a dynamic timer.",
            ),
            "holdout_connectivity": WorldTask(
                task_id="holdout_connectivity",
                app_family="connectivity",
                start_screen="home",
                goals=(
                    SemanticGoal(
                        "toggle-connectivity",
                        Operator.TOGGLE,
                        target_role="switch",
                        expected_state_key="checked",
                        expected_state_value="true",
                    ),
                ),
                success_screen="home",
                description="Use generic phone grammar to toggle the only switch in an untaught app.",
            ),
        }

    def start_task(self, task_id: str) -> WorldTask:
        catalog = self.task_catalog()
        if task_id not in catalog:
            raise PhoneWorldError(f"unknown task: {task_id}")
        task = catalog[task_id]
        self.reset(app_family=task.app_family, screen_name=task.start_screen)
        return task

    def task_success(self, task: WorldTask) -> bool:
        if self.app_family != task.app_family or self.screen_name != task.success_screen:
            return False
        if task.task_id == "settings_dark_mode":
            return self._dark_mode
        if task.task_id == "profile_display_name":
            return self._profile_name == "Screen Ghost"
        if task.task_id == "timer_start_stop":
            return not self._timer_running
        if task.task_id == "holdout_connectivity":
            return self._holdout_enabled
        return True

    # ------------------------------------------------------------------
    # Student/teacher observation boundary
    # ------------------------------------------------------------------
    def capture_png(self) -> bytes:
        self.pixel_reads += 1
        return self._render_frame().png_bytes

    def observe_frame(self) -> WorldFrame:
        self.pixel_reads += 1
        return self._render_frame()

    def teacher_snapshot(self) -> WorldFrame:
        self.teacher_reads += 1
        return self._render_frame()

    def _render_frame(self) -> WorldFrame:
        width, height = self.size
        image = Image.new("RGB", (width, height), self._palette()["background"])
        draw = ImageDraw.Draw(image)
        nodes = self._draw_surface(draw, width, height)
        self._last_nodes = tuple(nodes)
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=False, compress_level=9)
        png = buf.getvalue()
        projection = self._projection(tuple(nodes), width, height, png)
        payload = {
            "app_family": self.app_family,
            "screen_name": self.screen_name,
            "variant": self.variant.__dict__,
            "nodes": [
                {
                    "key": n.key,
                    "role": n.role,
                    "bounds": list(n.bounds_px),
                    "label": n.label,
                    "interactive": n.interactive,
                    "enabled": n.enabled,
                    "states": dict(n.states),
                    "parent": n.parent_key,
                    "dynamic": n.dynamic,
                    "sensitive": n.sensitive,
                }
                for n in nodes
            ],
        }
        return WorldFrame(
            png_bytes=png,
            width=width,
            height=height,
            app_family=self.app_family,
            surface_id=f"phoneworld.{self.app_family}",
            screen_name=self.screen_name,
            screen_key=str(projection["screen_key"]),
            nodes=tuple(nodes),
            runtime_projection=projection,
            teacher_payload_sha256=sha256_json(payload),
            tick_ms=self.tick_ms,
        )

    def _projection(
        self,
        nodes: Tuple[WorldNode, ...],
        width: int,
        height: int,
        png: bytes,
    ) -> dict[str, Any]:
        # Decode the rendered frame once.  Re-opening the same PNG for every
        # element made repeated campaigns progressively expensive and obscured
        # the actual perception costs the experiment is meant to measure.
        image = Image.open(io.BytesIO(png)).convert("RGB")
        image.load()
        elements = []
        grammar_terms = []
        control_terms = []
        content_terms = []
        for node in nodes:
            norm = node.normalized(width, height)
            element_id = "el_" + sha256_json(
                {
                    "role": node.role,
                    "bounds": list(norm),
                    "interactive": node.interactive,
                    "parent": node.parent_key,
                    "ordinal_key": node.key,
                }
            )[:20]
            crop_hash = self._crop_hash(image, node.bounds_px)
            elements.append(
                VisibleElement(
                    element_id=element_id,
                    role=node.role,
                    label=node.label,
                    normalized_bounds=norm,
                    interactive=node.interactive,
                    enabled=node.enabled,
                    states=node.states,
                    parent_element_id=node.parent_key,
                    sensitive=node.sensitive,
                    pixel_crop_sha256=crop_hash,
                ).to_dict()
            )
            grammar_terms.append(
                {
                    "role": node.role,
                    "bounds": list(norm),
                    "interactive": node.interactive,
                    "parent": node.parent_key,
                }
            )
            control_terms.append(
                {
                    "role": node.role,
                    "label": (node.label or "").casefold() if node.interactive else None,
                    "bounds": list(norm),
                    "interactive": node.interactive,
                    "parent": node.parent_key,
                }
            )
            content_terms.append(
                {
                    "role": node.role,
                    "label": node.label,
                    "states": dict(node.states),
                    "enabled": node.enabled,
                    "dynamic": node.dynamic,
                }
            )
        grammar_hash = sha256_json(grammar_terms)
        control_hash = sha256_json(control_terms)
        content_hash = sha256_json(content_terms)
        screen_key = "screen_" + sha256_json(
            {
                "surface_id": f"phoneworld.{self.app_family}",
                "screen_name": self.screen_name,
                "grammar_hash": grammar_hash,
                "control_hash": control_hash,
            }
        )[:24]
        return {
            "schema": "surface_teacher_runtime_v0",
            "lesson_id": "lesson_" + sha256_json(
                {
                    "screen_key": screen_key,
                    "content_hash": content_hash,
                    "png": sha256_bytes(png),
                    "variant": self.variant.__dict__,
                }
            ),
            "surface_id": f"phoneworld.{self.app_family}",
            "app_family": self.app_family,
            "app_version": self.variant.app_version,
            "screen_name": self.screen_name,
            "screen_key": screen_key,
            "grammar_hash": grammar_hash,
            "control_hash": control_hash,
            "content_hash": content_hash,
            "explanation": f"{self.app_family} {self.screen_name} rendered by PhoneWorld",
            "width": width,
            "height": height,
            "elements": elements,
            "provenance": {
                "compiled_from": "phoneworld_hidden_teacher",
                "teacher_lesson_id": "synthetic",
                "runtime_visibility": "teacher_blind",
                "privileged_fields_removed": True,
            },
        }

    @staticmethod
    def _crop_hash(image: Image.Image, bounds: Tuple[int, int, int, int]) -> str:
        crop = image.crop(bounds)
        return sha256_bytes(json_bytes({"mode": crop.mode, "size": crop.size}) + crop.tobytes())

    # ------------------------------------------------------------------
    # Motor surface
    # ------------------------------------------------------------------
    def tap_normalized(self, x: float, y: float) -> WorldActionResult:
        width, height = self.size
        return self.tap(int(round(x * width)), int(round(y * height)))

    def tap(self, x: int, y: int) -> WorldActionResult:
        if self._pending is not None:
            return WorldActionResult(False, False, "tap", None, None, None, "transition pending")
        frame = self._render_frame()
        candidates = [
            node
            for node in frame.nodes
            if node.interactive and node.enabled and self._contains(node.bounds_px, x, y)
        ]
        if self.variant.overlay_target and self.app_family == "settings" and self.screen_name == "display":
            candidates = [n for n in candidates if n.key != "settings.dark_mode"]
        if not candidates:
            self.action_log.append({"at_ms": self.tick_ms, "type": "tap", "x": x, "y": y, "accepted": False})
            return WorldActionResult(False, False, "tap", None, None, None, "no enabled target at point")
        candidates.sort(key=lambda n: (self._area(n.bounds_px), n.key))
        target = candidates[0]
        self.actions_injected += 1
        self.action_log.append(
            {
                "at_ms": self.tick_ms,
                "type": "tap",
                "x": x,
                "y": y,
                "accepted": True,
                "target_key": target.key,
                "target_label": target.label,
            }
        )
        return self._activate_target(target)

    def type_text(self, text: str) -> WorldActionResult:
        if self._pending is not None:
            return WorldActionResult(False, False, "type", None, None, None, "transition pending")
        if not (self.app_family == "profile" and self.screen_name == "edit"):
            return WorldActionResult(False, False, "type", None, None, None, "no editable field focused")
        self.actions_injected += 1
        self._input_buffer = str(text)
        self.action_log.append(
            {
                "at_ms": self.tick_ms,
                "type": "type",
                "accepted": True,
                "target_key": "profile.display_name",
                "text_length": len(text),
                "text_sha256": sha256_bytes(text.encode("utf-8")),
            }
        )
        return WorldActionResult(True, True, "type", "profile.display_name", "Display name", None, "text entered")

    def back(self) -> WorldActionResult:
        if self._pending is not None:
            return WorldActionResult(False, False, "back", None, None, None, "transition pending")
        self.actions_injected += 1
        previous = self.screen_name
        if self.screen_name in {"display", "saved"} and self.app_family == "settings":
            self.screen_name = "home"
        elif self.screen_name in {"edit", "saved"} and self.app_family == "profile":
            self.screen_name = "home"
        elif self.screen_name in {"running", "stopped"} and self.app_family == "timer":
            self.screen_name = "home"
            self._timer_running = False
        else:
            self.app_family = "launcher"
            self.screen_name = "home"
        self.action_log.append({"at_ms": self.tick_ms, "type": "back", "accepted": True, "from": previous})
        return WorldActionResult(True, True, "back", None, None, None, "navigated back")

    def _activate_target(self, node: WorldNode) -> WorldActionResult:
        delay = 220.0
        callbacks: dict[str, tuple[str, dict[str, Any], str]] = {
            "settings.display": ("_set_screen", {"screen": "display"}, "opening Display"),
            "settings.dark_mode": ("_toggle_dark_mode", {}, "toggling Dark mode"),
            "settings.save": ("_set_screen", {"screen": "saved"}, "saving Display settings"),
            "profile.edit": ("_set_screen", {"screen": "edit"}, "opening profile editor"),
            "profile.display_name": ("_focus_profile_field", {}, "focusing Display name"),
            "profile.save": ("_save_profile", {}, "saving profile"),
            "timer.start": ("_start_timer", {}, "starting timer"),
            "timer.stop": ("_stop_timer", {}, "stopping timer"),
            "connectivity.switch": ("_toggle_holdout", {}, "toggling connectivity"),
            "launcher.settings": ("_open_app", {"app": "settings", "screen": "home"}, "opening Settings"),
            "launcher.profile": ("_open_app", {"app": "profile", "screen": "home"}, "opening Profile"),
            "launcher.timer": ("_open_app", {"app": "timer", "screen": "home"}, "opening Timer"),
        }
        if node.key not in callbacks:
            return WorldActionResult(False, False, "tap", node.key, node.label, None, "target has no action")
        callback, metadata, reason = callbacks[node.key]
        if node.key == "profile.display_name":
            delay = 40.0
        due = self.tick_ms + delay
        self._pending = _PendingTransition(due, callback, metadata)
        return WorldActionResult(True, True, "tap", node.key, node.label, due, reason)

    def _set_screen(self, *, screen: str) -> None:
        self.screen_name = screen
        if screen == "saved":
            self._toast_until_ms = self.tick_ms + 800.0

    def _open_app(self, *, app: str, screen: str) -> None:
        self.app_family = app
        self.screen_name = screen

    def _toggle_dark_mode(self) -> None:
        self._dark_mode = not self._dark_mode

    def _focus_profile_field(self) -> None:
        # Focus is represented visually; no additional state required.
        return

    def _save_profile(self) -> None:
        self._profile_name = self._input_buffer
        self.screen_name = "saved"
        self._toast_until_ms = self.tick_ms + 800.0

    def _start_timer(self) -> None:
        self._timer_running = True
        self._timer_started_ms = self.tick_ms
        self.screen_name = "running"

    def _stop_timer(self) -> None:
        self._timer_running = False
        self.screen_name = "stopped"

    def _toggle_holdout(self) -> None:
        self._holdout_enabled = not self._holdout_enabled

    @staticmethod
    def _contains(bounds: Tuple[int, int, int, int], x: int, y: int) -> bool:
        x1, y1, x2, y2 = bounds
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _area(bounds: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bounds
        return (x2 - x1) * (y2 - y1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _palette(self) -> dict[str, Tuple[int, int, int]]:
        dark = self.variant.theme == "dark" or self._dark_mode
        if dark:
            return {
                "background": (25, 27, 31),
                "surface": (43, 46, 52),
                "text": (242, 243, 245),
                "muted": (178, 182, 190),
                "line": (91, 96, 106),
                "accent": (108, 161, 255),
                "success": (83, 198, 128),
                "switch_off": (104, 109, 118),
                "danger": (235, 105, 105),
            }
        return {
            "background": (247, 248, 250),
            "surface": (255, 255, 255),
            "text": (28, 31, 36),
            "muted": (99, 105, 116),
            "line": (205, 209, 216),
            "accent": (54, 119, 238),
            "success": (42, 157, 93),
            "switch_off": (151, 156, 166),
            "danger": (203, 66, 66),
        }

    def _font(self, nominal: int, *, bold: bool = False) -> ImageFont.ImageFont:
        size = max(8, int(round(nominal * self.variant.font_scale * self.variant.density)))
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _sx(self, value: float) -> int:
        width, _ = self.size
        # All fixtures are authored in one canonical 360x720 phone coordinate
        # system.  Landscape changes the output viewport, not the authoring
        # coordinate basis; otherwise portrait y values would spill beyond 1.0.
        return int(round(value / self.BASE_PORTRAIT[0] * width))

    def _sy(self, value: float) -> int:
        _, height = self.size
        return int(round(value / self.BASE_PORTRAIT[1] * height))

    def _box(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
        return self._sx(x1), self._sy(y1), self._sx(x2), self._sy(y2)

    def _draw_surface(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> list[WorldNode]:
        if self.app_family == "launcher":
            return self._draw_launcher(draw, width, height)
        if self.app_family == "settings":
            return self._draw_settings(draw, width, height)
        if self.app_family == "profile":
            return self._draw_profile(draw, width, height)
        if self.app_family == "timer":
            return self._draw_timer(draw, width, height)
        if self.app_family == "connectivity":
            return self._draw_connectivity(draw, width, height)
        if self.app_family == "lookalike":
            return self._draw_lookalike(draw, width, height)
        if self.app_family == "unknown":
            return self._draw_unknown(draw, width, height)
        raise PhoneWorldError(f"unknown app family: {self.app_family}")

    def _draw_header(self, draw: ImageDraw.ImageDraw, title: str, *, dynamic_clock: bool = True) -> list[WorldNode]:
        p = self._palette()
        box = self._box(0, 0, 360, 70)
        draw.rectangle(box, fill=p["surface"])
        draw.line((box[0], box[3] - 1, box[2], box[3] - 1), fill=p["line"], width=max(1, self._sx(1)))
        draw.text((self._sx(18), self._sy(22)), title, fill=p["text"], font=self._font(19, bold=True))
        nodes = [WorldNode("header.title", "heading", self._box(12, 10, 250, 62), title)]
        if dynamic_clock:
            seconds = int(self.tick_ms // 1000) % 60
            clock = f"09:{seconds:02d}"
            draw.text((self._sx(300), self._sy(24)), clock, fill=p["muted"], font=self._font(12))
            nodes.append(WorldNode("system.clock", "text", self._box(292, 12, 352, 58), clock, dynamic=True))
        return nodes

    def _draw_launcher(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> list[WorldNode]:
        p = self._palette()
        nodes = self._draw_header(draw, "PhoneWorld")
        apps = [
            ("launcher.settings", "Settings", 32, 130, p["accent"]),
            ("launcher.profile", "Profile", 140, 130, p["success"]),
            ("launcher.timer", "Timer", 248, 130, p["danger"]),
        ]
        for key, label, x, y, color in apps:
            icon = self._box(x, y, x + 76, y + 76)
            draw.rounded_rectangle(icon, radius=self._sx(15), fill=color)
            draw.text((self._sx(x + 4), self._sy(y + 88)), label, fill=p["text"], font=self._font(12))
            nodes.append(WorldNode(key, "button", self._box(x - 8, y - 8, x + 84, y + 112), label, True))
        return nodes

    def _draw_settings(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> list[WorldNode]:
        p = self._palette()
        nodes = self._draw_header(draw, "Settings")
        if self.screen_name == "home":
            row = self._box(18, 118, 342, 194)
            draw.rounded_rectangle(row, radius=self._sx(12), fill=p["surface"], outline=p["line"], width=max(1, self._sx(1)))
            draw.text((self._sx(34), self._sy(140)), "Display", fill=p["text"], font=self._font(17, bold=True))
            draw.text((self._sx(314), self._sy(140)), ">", fill=p["muted"], font=self._font(18))
            nodes.append(WorldNode("settings.display", "button", row, "Display", True))
            return nodes
        if self.screen_name == "display":
            y_shift = 56 if self.variant.move_controls else 0
            label = "Night theme" if self.variant.rename_control else "Dark mode"
            row = self._box(18, 120 + y_shift, 342, 210 + y_shift)
            draw.rounded_rectangle(row, radius=self._sx(12), fill=p["surface"], outline=p["line"], width=max(1, self._sx(1)))
            draw.text((self._sx(34), self._sy(146 + y_shift)), label, fill=p["text"], font=self._font(16, bold=True))
            draw.text((self._sx(34), self._sy(174 + y_shift)), "Use a darker color palette", fill=p["muted"], font=self._font(11))
            switch = self._box(274, 142 + y_shift, 334, 182 + y_shift)
            self._draw_switch(draw, switch, self._dark_mode)
            nodes.append(
                WorldNode(
                    "settings.dark_mode",
                    "switch",
                    row,
                    label,
                    True,
                    states={"checked": str(self._dark_mode).lower()},
                )
            )
            save = self._box(210, 610, 334, 670)
            self._draw_button(draw, save, "Save")
            nodes.append(WorldNode("settings.save", "button", save, "Save", True))
            if self.variant.overlay_target:
                overlay = self._box(250, 128 + y_shift, 350, 200 + y_shift)
                draw.rectangle(overlay, fill=(225, 190, 70))
                draw.text((overlay[0] + self._sx(8), overlay[1] + self._sy(8)), "Loading", fill=(20, 20, 20), font=self._font(11))
                nodes.append(WorldNode("settings.overlay", "dialog", overlay, "Loading", False, dynamic=True))
            return nodes
        if self.screen_name == "saved":
            draw.text((self._sx(52), self._sy(228)), "Display settings saved", fill=p["success"], font=self._font(20, bold=True))
            nodes.append(WorldNode("settings.saved_message", "text", self._box(40, 200, 330, 275), "Display settings saved"))
            return nodes
        raise PhoneWorldError(f"unknown settings screen: {self.screen_name}")

    def _draw_profile(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> list[WorldNode]:
        p = self._palette()
        nodes = self._draw_header(draw, "Profile")
        if self.screen_name == "home":
            draw.ellipse(self._box(130, 115, 230, 215), fill=p["accent"])
            label = self._profile_name or "Guest User"
            draw.text((self._sx(110), self._sy(238)), label, fill=p["text"], font=self._font(18, bold=True))
            edit = self._box(80, 315, 280, 375)
            self._draw_button(draw, edit, "Edit profile")
            nodes.extend(
                [
                    WorldNode("profile.name", "text", self._box(70, 225, 300, 275), label),
                    WorldNode("profile.edit", "button", edit, "Edit profile", True),
                ]
            )
            return nodes
        if self.screen_name == "edit":
            draw.text((self._sx(24), self._sy(120)), "Display name", fill=p["text"], font=self._font(14, bold=True))
            field_box = self._box(22, 150, 338, 210)
            draw.rounded_rectangle(field_box, radius=self._sx(8), fill=p["surface"], outline=p["accent"], width=max(1, self._sx(2)))
            shown = self._input_buffer or "Enter a name"
            draw.text((self._sx(36), self._sy(170)), shown, fill=p["text"] if self._input_buffer else p["muted"], font=self._font(15))
            save = self._box(150, 610, 338, 670)
            self._draw_button(draw, save, "Save profile")
            nodes.extend(
                [
                    WorldNode(
                        "profile.display_name",
                        "text_field",
                        field_box,
                        "Display name",
                        True,
                        states={"focused": "true", "filled": str(bool(self._input_buffer)).lower()},
                        sensitive=False,
                    ),
                    WorldNode("profile.save", "button", save, "Save profile", True),
                ]
            )
            return nodes
        if self.screen_name == "saved":
            draw.text((self._sx(55), self._sy(230)), "Profile saved", fill=p["success"], font=self._font(22, bold=True))
            draw.text((self._sx(85), self._sy(280)), self._profile_name, fill=p["text"], font=self._font(17))
            nodes.extend(
                [
                    WorldNode("profile.saved_message", "text", self._box(45, 205, 315, 265), "Profile saved"),
                    WorldNode("profile.saved_name", "text", self._box(60, 265, 300, 315), self._profile_name),
                ]
            )
            return nodes
        raise PhoneWorldError(f"unknown profile screen: {self.screen_name}")

    def _draw_timer(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> list[WorldNode]:
        p = self._palette()
        nodes = self._draw_header(draw, "Timer")
        elapsed = max(0.0, self.tick_ms - self._timer_started_ms) if self._timer_running else 0.0
        if self.screen_name == "home":
            draw.text((self._sx(110), self._sy(210)), "00:00.0", fill=p["text"], font=self._font(34, bold=True))
            start = self._box(90, 410, 270, 480)
            self._draw_button(draw, start, "Start", fill=p["success"])
            nodes.extend(
                [
                    WorldNode("timer.value", "text", self._box(80, 180, 290, 260), "00:00.0", dynamic=True),
                    WorldNode("timer.start", "button", start, "Start", True),
                ]
            )
            return nodes
        if self.screen_name == "running":
            seconds = elapsed / 1000.0
            text = f"00:{seconds:04.1f}"
            draw.text((self._sx(95), self._sy(210)), text, fill=p["text"], font=self._font(34, bold=True))
            progress = min(1.0, (seconds % 10.0) / 10.0)
            ring = self._box(95, 300, 265, 470)
            draw.ellipse(ring, outline=p["line"], width=max(1, self._sx(10)))
            angle = int(progress * 360)
            draw.arc(ring, start=-90, end=-90 + angle, fill=p["accent"], width=max(1, self._sx(10)))
            stop = self._box(90, 540, 270, 610)
            self._draw_button(draw, stop, "Stop", fill=p["danger"])
            nodes.extend(
                [
                    WorldNode("timer.value", "text", self._box(75, 180, 295, 260), text, dynamic=True),
                    WorldNode("timer.progress", "progress", ring, "Elapsed progress", dynamic=True),
                    WorldNode("timer.stop", "button", stop, "Stop", True),
                ]
            )
            return nodes
        if self.screen_name == "stopped":
            draw.text((self._sx(85), self._sy(230)), "Timer stopped", fill=p["success"], font=self._font(24, bold=True))
            nodes.append(WorldNode("timer.stopped_message", "text", self._box(60, 200, 310, 275), "Timer stopped"))
            return nodes
        raise PhoneWorldError(f"unknown timer screen: {self.screen_name}")

    def _draw_connectivity(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> list[WorldNode]:
        p = self._palette()
        nodes = self._draw_header(draw, "Connections")
        row = self._box(20, 150, 340, 248)
        draw.rounded_rectangle(row, radius=self._sx(18), fill=(230, 239, 255) if self.variant.theme == "light" else (51, 61, 78))
        draw.ellipse(self._box(38, 175, 82, 219), fill=p["accent"])
        # This label intentionally differs from taught settings labels.
        draw.text((self._sx(98), self._sy(172)), "Wireless link", fill=p["text"], font=self._font(16, bold=True))
        draw.text((self._sx(98), self._sy(201)), "Allow nearby data", fill=p["muted"], font=self._font(11))
        switch = self._box(270, 177, 330, 217)
        self._draw_switch(draw, switch, self._holdout_enabled)
        nodes.append(
            WorldNode(
                "connectivity.switch",
                "switch",
                row,
                "Wireless link",
                True,
                states={"checked": str(self._holdout_enabled).lower()},
            )
        )
        return nodes

    def _draw_lookalike(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> list[WorldNode]:
        p = self._palette()
        nodes = self._draw_header(draw, "Settings")
        row = self._box(18, 120, 342, 210)
        draw.rounded_rectangle(row, radius=self._sx(12), fill=p["surface"], outline=p["line"], width=max(1, self._sx(1)))
        draw.text((self._sx(34), self._sy(146)), "Demo mode", fill=p["text"], font=self._font(16, bold=True))
        draw.text((self._sx(34), self._sy(174)), "Does not change system theme", fill=p["muted"], font=self._font(11))
        self._draw_switch(draw, self._box(274, 142, 334, 182), False)
        nodes.append(WorldNode("lookalike.demo", "switch", row, "Demo mode", True, states={"checked": "false"}))
        return nodes

    def _draw_unknown(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> list[WorldNode]:
        p = self._palette()
        nodes = self._draw_header(draw, "Canvas")
        for i in range(9):
            x = 20 + (i % 3) * 112
            y = 120 + (i // 3) * 130
            color = (
                70 + (i * 31) % 160,
                80 + (i * 53) % 150,
                90 + (i * 71) % 140,
            )
            draw.polygon(
                [
                    (self._sx(x + 10), self._sy(y)),
                    (self._sx(x + 86), self._sy(y + 28)),
                    (self._sx(x + 60), self._sy(y + 100)),
                    (self._sx(x), self._sy(y + 72)),
                ],
                fill=color,
            )
        draw.text((self._sx(72), self._sy(555)), "Custom rendered surface", fill=p["text"], font=self._font(17, bold=True))
        nodes.append(WorldNode("unknown.canvas", "canvas", self._box(10, 90, 350, 540), "Custom surface"))
        return nodes

    def _draw_switch(self, draw: ImageDraw.ImageDraw, bounds: Tuple[int, int, int, int], checked: bool) -> None:
        p = self._palette()
        x1, y1, x2, y2 = bounds
        draw.rounded_rectangle(bounds, radius=max(1, (y2 - y1) // 2), fill=p["accent"] if checked else p["switch_off"])
        radius = max(4, int((y2 - y1) * 0.36))
        cy = (y1 + y2) // 2
        cx = x2 - radius - max(2, self._sx(4)) if checked else x1 + radius + max(2, self._sx(4))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255))

    def _draw_button(
        self,
        draw: ImageDraw.ImageDraw,
        bounds: Tuple[int, int, int, int],
        label: str,
        *,
        fill: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        p = self._palette()
        fill = fill or p["accent"]
        draw.rounded_rectangle(bounds, radius=self._sx(10), fill=fill)
        font = self._font(15, bold=True)
        bbox = draw.textbbox((0, 0), label, font=font)
        tx = (bounds[0] + bounds[2] - (bbox[2] - bbox[0])) // 2
        ty = (bounds[1] + bounds[3] - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)


class PhoneWorldTemporalSource:
    """Read-only temporal-teacher adapter for :class:`PhoneWorld`."""

    def __init__(self, world: PhoneWorld):
        self.world = world

    def capture_png(self) -> bytes:
        return self.world.capture_png()

    def inspect_structure(self):
        from core.surface_temporal_teacher import StructureSnapshot

        frame = self.world.teacher_snapshot()
        return StructureSnapshot(
            source_digest=frame.teacher_payload_sha256,
            alignment_nodes=tuple(node.alignment_node() for node in frame.nodes),
            compiler_nodes=frame.runtime_projection,
            event_idle=not self.world.pending,
            source_payload={"app_family": frame.app_family, "screen": frame.screen_name},
        )
