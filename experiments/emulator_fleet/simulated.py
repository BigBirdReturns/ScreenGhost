"""Deterministic multi-instance laboratory built from PhoneWorld."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from PIL import ImageDraw

from experiments.emulator_fleet.schema import (
    EmulatorVendor,
    InstanceProfile,
    InstanceRef,
    InstanceStatus,
    MacroAction,
    MacroActionKind,
)
from experiments.generic_utility.phone_world import DisplayVariant, PhoneWorld, WorldActionResult, WorldNode


class AccountPhoneWorld(PhoneWorld):
    """PhoneWorld variant with account-specific dynamic content in the header."""

    def __init__(self, *, account_label: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.account_label = account_label

    def _draw_header(self, draw: ImageDraw.ImageDraw, title: str, *, dynamic_clock: bool = True):
        nodes = super()._draw_header(draw, title, dynamic_clock=dynamic_clock)
        if self.account_label:
            p = self._palette()
            label = self.account_label[:14]
            draw.text((self._sx(184), self._sy(47)), label, fill=p["muted"], font=self._font(9))
            nodes.append(
                WorldNode(
                    "account.dynamic_label",
                    "text",
                    self._box(176, 38, 286, 65),
                    label,
                    dynamic=True,
                )
            )
        return nodes


@dataclass(frozen=True)
class SimulatedFleetSpec:
    instance_id: str
    name: str
    variant: DisplayVariant
    account_label: str = ""
    expected_semantic_success: bool = True
    expected_baseline_success: bool = True
    tags: tuple[str, ...] = ()


class SimulatedInstanceAdapter:
    """One isolated PhoneWorld instance with macro and semantic motor surfaces."""

    def __init__(self, spec: SimulatedFleetSpec, *, seed: int = 7) -> None:
        self.spec = spec
        self.instance_id = spec.instance_id
        self.world = AccountPhoneWorld(
            seed=seed,
            variant=spec.variant,
            account_label=spec.account_label,
        )
        width, height = self.world.size
        self.profile = InstanceProfile(
            profile_id=f"profile:{spec.instance_id}",
            width=width,
            height=height,
            dpi=max(120, int(round(320 * spec.variant.density))),
            theme=spec.variant.theme,
            font_scale=spec.variant.font_scale,
            tags=spec.tags,
        )
        self.ref = InstanceRef(
            vendor=EmulatorVendor.SIMULATED,
            instance_id=spec.instance_id,
            name=spec.name,
            index=None,
            status=InstanceStatus.RUNNING,
            metadata={"tags": list(spec.tags), "profile": self.profile.to_dict()},
        )

    @property
    def pending(self) -> bool:
        return self.world.pending

    @property
    def teacher_reads(self) -> int:
        return self.world.teacher_reads

    @property
    def pixel_reads(self) -> int:
        return self.world.pixel_reads

    @property
    def actions_injected(self) -> int:
        return self.world.actions_injected

    def now_ms(self) -> float:
        return float(self.world.tick_ms)

    def start_task(self, task_id: str):
        return self.world.start_task(task_id)

    def reset_task(self, task_id: str):
        return self.world.start_task(task_id)

    def task_success(self, task) -> bool:
        return self.world.task_success(task)

    def capture_png(self) -> bytes:
        return self.world.capture_png()

    def capture_teacher(self) -> Mapping[str, Any]:
        return dict(self.world.teacher_snapshot().runtime_projection)

    def advance(self, milliseconds: float) -> None:
        self.world.advance(milliseconds)

    def tap_normalized(self, x: float, y: float) -> WorldActionResult:
        return self.world.tap_normalized(x, y)

    def tap_source_pixels(self, x: int, y: int) -> WorldActionResult:
        # Coordinate macros recorded at the leader's resolution are intentionally
        # replayed as source pixels. Resolution or layout drift therefore remains
        # visible instead of being silently normalized by the experiment harness.
        return self.world.tap(x, y)

    def type_text(self, text: str) -> WorldActionResult:
        return self.world.type_text(text)

    def back(self) -> WorldActionResult:
        return self.world.back()

    def home(self) -> WorldActionResult:
        # PhoneWorld has no host-level HOME motor. Reset to launcher without
        # pretending this is a verified app transition.
        return WorldActionResult(False, False, "home", None, None, None, "home unsupported in PhoneWorld")

    def swipe_normalized(self, path, duration_ms: float = 300.0) -> WorldActionResult:
        # Current campaign tasks do not require scrolling. Preserve the gesture
        # contract and fail honestly if a future macro introduces it.
        return WorldActionResult(False, False, "swipe", None, None, None, "scroll surface not implemented in PhoneWorld")

    def long_press_normalized(self, x: float, y: float, duration_ms: float) -> WorldActionResult:
        del duration_ms
        return self.world.tap_normalized(x, y)

    def execute_macro_action(
        self,
        action: MacroAction,
        *,
        text_values: Mapping[str, str],
    ) -> WorldActionResult:
        if action.kind is MacroActionKind.TAP and action.point is not None:
            return self.world.tap_normalized(*action.point)
        if action.kind is MacroActionKind.LONG_PRESS and action.point is not None:
            return self.long_press_normalized(*action.point, action.duration_ms)
        if action.kind is MacroActionKind.SWIPE:
            return self.swipe_normalized(action.path, action.duration_ms)
        if action.kind is MacroActionKind.TEXT:
            return self.world.type_text(text_values[action.text_ref or ""])
        if action.kind is MacroActionKind.KEY:
            key = str(action.key or "").casefold()
            if key in {"back", "escape", "keycode_back", "4"}:
                return self.world.back()
            if key in {"home", "keycode_home", "3"}:
                return self.home()
            return WorldActionResult(False, False, "key", None, None, None, f"unsupported key {action.key!r}")
        if action.kind is MacroActionKind.WAIT:
            self.world.advance(action.duration_ms)
            return WorldActionResult(True, False, "wait", None, None, None, "wait consumed")
        return WorldActionResult(False, False, action.kind.value, None, None, None, "unsupported macro action")

    def snapshot_state(self) -> dict[str, Any]:
        return self.world.snapshot_state()

    def restore_state(self, value: Mapping[str, Any]) -> None:
        self.world.restore_state(value)

    def current_screen(self) -> str:
        return self.world.screen_name


class SimulatedFleet:
    def __init__(self, specs: list[SimulatedFleetSpec], *, seed: int = 7) -> None:
        ids = [row.instance_id for row in specs]
        if len(ids) != len(set(ids)):
            raise ValueError("simulated fleet instance IDs must be unique")
        self.instances = {
            spec.instance_id: SimulatedInstanceAdapter(spec, seed=seed + index)
            for index, spec in enumerate(specs)
        }

    def get(self, instance_id: str) -> SimulatedInstanceAdapter:
        return self.instances[instance_id]

    def list_instances(self) -> tuple[SimulatedInstanceAdapter, ...]:
        return tuple(self.instances[key] for key in sorted(self.instances))
