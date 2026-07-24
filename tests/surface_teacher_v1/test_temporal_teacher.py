from __future__ import annotations

import hashlib

from core.surface_alignment import AlignmentNode, AlignmentPolicy
from core.surface_temporal_teacher import (
    StructureSnapshot,
    teach_temporal_surface,
)
from tests.surface_teacher_v1.support import png_frame


class FakeSource:
    def __init__(self):
        self.capture_index = 0
        self.inspect_index = 0
        self.calls = []

    def capture_png(self):
        self.calls.append("capture_png")
        value = self.capture_index // 2
        self.capture_index += 1
        return png_frame(spinner=value)

    def inspect_structure(self):
        self.calls.append("inspect_structure")
        index = self.inspect_index
        self.inspect_index += 1
        nodes = (
            AlignmentNode("title", "heading", (12, 12, 140, 48), label="Settings"),
            AlignmentNode("dark", "switch", (20, 80, 180, 126), label="Dark mode", interactive=True),
            AlignmentNode("clock", "text", (150, 20, 180, 40), label=str(index), dynamic=True),
        )
        payload = f"structure-{index}"
        return StructureSnapshot(
            hashlib.sha256(payload.encode()).hexdigest(),
            nodes,
            compiler_nodes={"snapshot": index},
            source_payload=payload,
        )


def test_temporal_orchestrator_is_read_only_and_calls_compiler_once():
    source = FakeSource()
    ticks = iter((0.0, 10.0, 100.0, 110.0, 220.0, 230.0))
    calls = []

    def compiler(png, nodes, digest, certificate):
        calls.append((png, nodes, digest, certificate))
        return {"lesson": "compiled", "snapshot": nodes["snapshot"]}

    result = teach_temporal_surface(
        source,
        compiler,
        burst_count=3,
        interval_ms=0,
        monotonic_ms=lambda: next(ticks),
        sleep=lambda _seconds: None,
        alignment_policy=AlignmentPolicy(minimum_samples=6, minimum_duration_ms=180),
    )
    assert result.compiled_lesson["lesson"] == "compiled"
    assert len(calls) == 1
    assert source.calls == [
        "capture_png", "inspect_structure", "capture_png",
        "capture_png", "inspect_structure", "capture_png",
        "capture_png", "inspect_structure", "capture_png",
    ]
    assert all("tap" not in call and "click" not in call for call in source.calls)
    assert calls[0][3]["certificate_id"] == result.alignment.certificate.certificate_id
