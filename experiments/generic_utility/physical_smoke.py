"""Read-only physical/AVD smoke using ScreenGhost's existing ADB driver."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from core.surface_alignment import AlignmentPolicy, stage_alignment
from core.surface_temporal_teacher import StructureSnapshot, teach_temporal_surface
from experiments.generic_utility.schema import json_bytes, sha256_bytes


class AdbTemporalSource:
    def __init__(self, driver: Any, device: Optional[str]) -> None:
        self.driver = driver
        self.device = device
        self.last_size: Optional[tuple[int, int]] = None
        self.pixel_reads = 0
        self.teacher_reads = 0

    def capture_png(self) -> bytes:
        image = self.driver.screencap(self.device)
        self.last_size = tuple(image.size)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=6)
        self.pixel_reads += 1
        return buffer.getvalue()

    def inspect_structure(self) -> StructureSnapshot:
        if self.last_size is None:
            raise RuntimeError("capture_png must run before inspect_structure")
        from core.teacher_android import parse_uiautomator_xml
        from core.surface_alignment import AlignmentNode
        from experiments.generic_utility.schema import sha256_json

        xml = self.driver.dump_ui_xml(self.device)
        teacher_nodes = parse_uiautomator_xml(xml, viewport=self.last_size)
        alignment_nodes = tuple(
            AlignmentNode(
                semantic_key=node.source_ref,
                role=node.role,
                bounds=(node.bounds.x1, node.bounds.y1, node.bounds.x2, node.bounds.y2),
                label=node.label,
                interactive=node.interactive,
                enabled=node.enabled,
                parent_key=node.parent_ref,
                states=node.states,
                dynamic=False,
            )
            for node in teacher_nodes
        )
        self.teacher_reads += 1
        return StructureSnapshot(
            source_digest=sha256_json({"xml": xml}),
            alignment_nodes=alignment_nodes,
            compiler_nodes=teacher_nodes,
            event_idle=True,
            source_payload={"xml_sha256": sha256_bytes(xml.encode("utf-8"))},
        )


def run_physical_read_smoke(
    out_dir: str | Path,
    *,
    device: Optional[str] = None,
    surface_id: str = "physical.current",
    burst_count: int = 3,
) -> Path:
    try:
        from drivers import AndroidAdbDriver
        from core.surface_teacher import LessonPolicy, SourceKind, compile_lesson, stage_lesson
    except Exception as exc:
        raise RuntimeError(
            "Physical smoke requires ScreenGhost main plus PR #13 Surface Teacher modules"
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    driver = AndroidAdbDriver()
    if not driver.available():
        raise RuntimeError("adb is unavailable")
    devices = driver.list_devices()
    chosen = device or (devices[0] if len(devices) == 1 else None)
    if not chosen:
        raise RuntimeError(f"choose --device from connected devices: {devices}")
    source = AdbTemporalSource(driver, chosen)

    def compiler(png: bytes, nodes: Any, burst_digest: str, certificate: dict[str, Any]):
        return compile_lesson(
            png,
            surface_id=surface_id,
            source_kind=SourceKind.ANDROID_UIAUTOMATOR,
            source_payload_sha256=burst_digest,
            nodes=nodes,
            policy=LessonPolicy(),
        )

    started = time.monotonic()
    result = teach_temporal_surface(
        source,
        compiler,
        burst_count=burst_count,
        interval_ms=120,
        alignment_policy=AlignmentPolicy(
            minimum_samples=max(3, burst_count * 2),
            minimum_duration_ms=180,
            max_dynamic_fraction=0.20,
            max_static_mean_difference=0.75,
            max_interactive_shift_px=3.0,
        ),
    )
    stage_alignment(result.alignment, out / "alignment")
    stage_lesson(result.compiled_lesson, out / "lesson")
    receipt = {
        "schema": "screenghost_physical_read_smoke_v1",
        "device": chosen,
        "surface_id": surface_id,
        "duration_ms": (time.monotonic() - started) * 1000.0,
        "alignment_certificate_id": result.alignment.certificate.certificate_id,
        "lesson_id": result.compiled_lesson.lesson.lesson_id,
        "pixel_reads": source.pixel_reads,
        "teacher_reads": source.teacher_reads,
        "input_actions": 0,
        "listener_started": False,
        "host_foreground_requested": False,
        "status": "pass",
    }
    (out / "receipt.json").write_bytes(json_bytes(receipt))
    return out
