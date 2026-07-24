"""Read-only AndroidWorld smoke for the final emulator transport check."""
from __future__ import annotations

import time
from pathlib import Path

from core.surface_alignment import AlignmentPolicy, stage_alignment
from core.surface_curriculum import build_curriculum, stage_curriculum
from core.surface_temporal_teacher import teach_temporal_surface
from experiments.generic_utility.androidworld_adapter import AndroidWorldBackend
from experiments.generic_utility.schema import json_bytes


def run_androidworld_read_smoke(
    out_dir: str | Path,
    *,
    adb_path: str,
    console_port: int = 5554,
    perform_emulator_setup: bool = False,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    backend = AndroidWorldBackend.launch(
        adb_path=adb_path,
        console_port=console_port,
        perform_emulator_setup=perform_emulator_setup,
    )
    started = time.monotonic()
    try:
        backend.reset(go_home=True)
        result = teach_temporal_surface(
            backend.temporal_source(),
            lambda png, projection, burst_digest, certificate: {
                "projection": dict(projection),
                "burst_digest": burst_digest,
                "certificate": dict(certificate),
            },
            burst_count=3,
            interval_ms=120,
            alignment_policy=AlignmentPolicy(
                minimum_samples=6,
                minimum_duration_ms=180,
                max_dynamic_fraction=0.25,
                max_static_mean_difference=0.75,
                max_interactive_shift_px=3.0,
            ),
        )
        projection = result.compiled_lesson["projection"]
        stage_alignment(result.alignment, out / "alignment")
        curriculum = build_curriculum(result.alignment.representative_png, projection)
        stage_curriculum(curriculum, out / "curriculum", source_png=result.alignment.representative_png)
        receipt = {
            "schema": "screenghost_androidworld_read_smoke_v1",
            "surface_id": projection["surface_id"],
            "screen_key": projection["screen_key"],
            "alignment_certificate_id": result.alignment.certificate.certificate_id,
            "curriculum_id": curriculum.curriculum_id,
            "duration_ms": (time.monotonic() - started) * 1000.0,
            "teacher_reads": backend.teacher_reads,
            "pixel_reads": backend.pixel_reads,
            "input_actions": backend.actions_injected,
            "status": "pass",
        }
        (out / "receipt.json").write_bytes(json_bytes(receipt))
        return out
    finally:
        backend.close()
