"""ScreenGhost Interface Procedure v0 — bounded, auditable interface operation.

Human interaction is the fallback protocol. When a vendor's API is captured,
cloud-gated, or revoked, the product still has to show a state to a human and
let that human press a control. This module operates that human-facing surface
the way a careful human would -- and unlike "GUI automation", every run is
capture with custody, procedure, and proof:

    approved procedure (anchor, target, bounds, verification)
      -> wait for the known anchor
      -> BEFORE screenshot (pixel evidence)
      -> locate the target; confirm its label; confirm it sits inside the
         APPROVED bounds -- else drift, no click
      -> one click, inside approved bounds only
      -> wait for the visible state transition; AFTER screenshot
      -> verified -> COMPLETED trace; else -> DRIFT trace

DRIFT IS A FIRST-CLASS OUTCOME, NOT AN ERROR. If the tile moves, the label
changes, a modal hides the anchor, or the verification text never appears, the
runner does not improvise, retry, or hunt for the control. It stops at the
deviation, captures the drift state, and the trace says exactly which check
failed. A new surface needs a new approved procedure, not a cleverer robot.
That non-improvisation property is what separates an operator from malware.

Verification honesty: the state check reads the rendered text at an approved
locator -- ScreenGhost's existing view-tree rung (exact on-screen text). The
EVIDENCE is the before/after pixels; the CHECK is text. No OCR, no image model,
no pixel interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Tuple, runtime_checkable

TRACE_TIER = "interface_procedure_trace"
TRACE_TIER_LIMITS: Tuple[str, ...] = (
    "rendered pixels + one bounded action + visible verification only",
    "not vendor backend state",
    "not API truth",
    "not platform truth",
    "not legal-grade provenance by itself",
)

Box = Tuple[float, float, float, float]  # x, y, width, height


class ProcedureRefused(ValueError):
    """The runner refused to start: the procedure is not fully bounded."""


class Outcome(str, Enum):
    COMPLETED = "completed"
    DRIFT = "drift"


class DriftReason(str, Enum):
    """Exactly which check failed. Each stops the run at that point."""

    ANCHOR_MISSING = "anchor_missing"                # the known surface anchor is not there
    TARGET_MISSING = "target_missing"                # the approved control does not exist
    TARGET_LABEL_MISMATCH = "target_label_mismatch"  # the control exists but is not what was approved
    TARGET_OUTSIDE_BOUNDS = "target_outside_bounds"  # the control moved; NO click is performed
    VERIFICATION_FAILED = "verification_failed"      # clicked, but the visible state never confirmed


@dataclass(frozen=True)
class ApprovedBounds:
    """The only screen region the procedure is allowed to act inside."""

    x: float
    y: float
    width: float
    height: float

    def contains(self, box: Box) -> bool:
        bx, by, bw, bh = box
        return (
            bx >= self.x
            and by >= self.y
            and bx + bw <= self.x + self.width
            and by + bh <= self.y + self.height
        )


@dataclass(frozen=True)
class InterfaceProcedure:
    """One approved, fully-bounded interface procedure. Declarative; the runner
    executes exactly this and nothing else."""

    procedure_id: str
    surface_label: str            # e.g. "local-fixture-surface", later "google-home-emulator"
    anchor_selector: str          # the known dashboard anchor that must be present
    anchor_text: str              # its exact rendered text
    target_selector: str          # the approved control
    target_label: str             # text the control must carry (containment)
    approved_bounds: ApprovedBounds
    verify_selector: str          # where the visible state appears
    verify_expected_text: str     # exact rendered text that proves the transition
    verify_timeout_ms: int = 3000

    def require_bounded(self) -> None:
        """Every field that bounds the action must be present. An unbounded
        procedure is refused before the driver is touched."""
        missing = [
            name
            for name, val in (
                ("procedure_id", self.procedure_id),
                ("surface_label", self.surface_label),
                ("anchor_selector", self.anchor_selector),
                ("anchor_text", self.anchor_text),
                ("target_selector", self.target_selector),
                ("target_label", self.target_label),
                ("verify_selector", self.verify_selector),
                ("verify_expected_text", self.verify_expected_text),
            )
            if not (isinstance(val, str) and val.strip())
        ]
        if missing:
            raise ProcedureRefused(
                f"procedure is not fully bounded; missing/empty: {missing}. "
                f"An unbounded procedure is refused, not improvised around."
            )


@runtime_checkable
class SurfaceDriver(Protocol):
    """The minimal surface a driver must expose. Read + one pointed click.

    Implementations: PlaywrightSurfaceDriver (browser), later an Android
    emulator driver, a Windows-app driver -- the runner never changes.
    """

    def screenshot(self) -> bytes: ...

    def text(self, selector: str) -> Optional[str]: ...

    def find_box(self, selector: str) -> Optional[Box]: ...

    def click(self, x: float, y: float) -> None: ...

    def wait_text(self, selector: str, expected: str, timeout_ms: int) -> bool: ...


@dataclass(frozen=True)
class StepRecord:
    step: str
    ok: bool
    detail: str


@dataclass
class ProcedureTrace:
    """The auditable record of one run: what was approved, what happened, and
    the pixel evidence at each stage. This is what genesis seals."""

    procedure: InterfaceProcedure
    outcome: Outcome
    drift_reason: Optional[DriftReason] = None
    steps: List[StepRecord] = field(default_factory=list)
    before_png: Optional[bytes] = None   # rendered surface before the action
    after_png: Optional[bytes] = None    # rendered surface after the action
    drift_png: Optional[bytes] = None    # rendered surface at the moment of drift
    clicked: bool = False
    click_point: Optional[Tuple[float, float]] = None


class ProcedureRunner:
    """Execute one approved procedure against one surface. No improvisation:
    the first failed check ends the run with a drift trace."""

    def run(self, driver: SurfaceDriver, procedure: InterfaceProcedure) -> ProcedureTrace:
        procedure.require_bounded()
        trace = ProcedureTrace(procedure=procedure, outcome=Outcome.DRIFT)

        def step(name: str, ok: bool, detail: str) -> bool:
            trace.steps.append(StepRecord(step=name, ok=ok, detail=detail))
            return ok

        def drift(reason: DriftReason) -> ProcedureTrace:
            # Capture the surface exactly as it looked when the check failed,
            # then STOP. No retry, no fallback locator, no further action.
            trace.drift_reason = reason
            trace.drift_png = driver.screenshot()
            trace.steps.append(StepRecord(step="capture_drift", ok=True, detail=reason.value))
            return trace

        # 1) the known anchor must be rendered, exactly
        anchor = driver.text(procedure.anchor_selector)
        if not step(
            "wait_for_anchor",
            anchor is not None and anchor.strip() == procedure.anchor_text,
            f"expected {procedure.anchor_text!r} at {procedure.anchor_selector!r}, saw {anchor!r}",
        ):
            return drift(DriftReason.ANCHOR_MISSING)

        # 2) before evidence
        trace.before_png = driver.screenshot()
        step("capture_before", True, "before screenshot captured")

        # 3) the approved control must exist ...
        box = driver.find_box(procedure.target_selector)
        if not step(
            "locate_target",
            box is not None,
            f"target {procedure.target_selector!r} box={box}",
        ):
            return drift(DriftReason.TARGET_MISSING)

        # 4) ... carry the approved label ...
        label = driver.text(procedure.target_selector)
        if not step(
            "check_label",
            label is not None and procedure.target_label in label,
            f"expected label containing {procedure.target_label!r}, saw {label!r}",
        ):
            return drift(DriftReason.TARGET_LABEL_MISMATCH)

        # 5) ... and sit entirely inside the approved bounds. A moved control is
        # drift: the click is NOT performed anywhere else.
        if not step(
            "check_bounds",
            procedure.approved_bounds.contains(box),
            f"target box {box} vs approved {procedure.approved_bounds}",
        ):
            return drift(DriftReason.TARGET_OUTSIDE_BOUNDS)

        # 6) one click, at the center of the approved control
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        driver.click(cx, cy)
        trace.clicked = True
        trace.click_point = (cx, cy)
        step("click", True, f"clicked ({cx:.1f}, {cy:.1f}) inside approved bounds")

        # 7) the visible state must confirm within the timeout
        verified = driver.wait_text(
            procedure.verify_selector, procedure.verify_expected_text, procedure.verify_timeout_ms
        )
        # after evidence is captured either way -- a failed verification is
        # exactly the state that must go on the record
        trace.after_png = driver.screenshot()
        step("capture_after", True, "after screenshot captured")
        if not step(
            "verify",
            verified,
            f"expected {procedure.verify_expected_text!r} at {procedure.verify_selector!r}",
        ):
            trace.drift_reason = DriftReason.VERIFICATION_FAILED
            trace.steps.append(
                StepRecord(step="capture_drift", ok=True, detail=DriftReason.VERIFICATION_FAILED.value)
            )
            return trace

        trace.outcome = Outcome.COMPLETED
        return trace


class PlaywrightSurfaceDriver:
    """Browser surface via Playwright/Chromium. Imported lazily; the runner and
    its tests never require a browser. Read-only against the page except for the
    single pointed click the runner requests."""

    def __init__(
        self,
        url: str,
        *,
        executable_path: Optional[str] = None,
        viewport: Tuple[int, int] = (800, 600),
    ) -> None:
        from playwright.sync_api import sync_playwright  # lazy

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True, executable_path=executable_path)
        self._page = self._browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        self._page.goto(url)

    def screenshot(self) -> bytes:
        return self._page.screenshot()

    def text(self, selector: str) -> Optional[str]:
        loc = self._page.locator(selector)
        if loc.count() == 0:
            return None
        return loc.first.text_content()

    def find_box(self, selector: str) -> Optional[Box]:
        loc = self._page.locator(selector)
        if loc.count() == 0:
            return None
        b = loc.first.bounding_box()
        if b is None:
            return None
        return (b["x"], b["y"], b["width"], b["height"])

    def click(self, x: float, y: float) -> None:
        self._page.mouse.click(x, y)

    def wait_text(self, selector: str, expected: str, timeout_ms: int) -> bool:
        import time

        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            t = self.text(selector)
            if t is not None and t.strip() == expected:
                return True
            if time.monotonic() >= deadline:
                return False
            self._page.wait_for_timeout(50)

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()
