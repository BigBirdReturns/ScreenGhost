from __future__ import annotations

import json

import pytest

from core.surface_alignment import (
    AlignmentNode,
    AlignmentPolicy,
    AlignmentRefused,
    FrameObservation,
    certify_alignment,
    stage_alignment,
)
from tests.surface_teacher_v1.support import png_frame


def nodes(*, button_x: int = 20, button_label: str = "Dark mode"):
    return (
        AlignmentNode("title", "heading", (12, 12, 140, 48), label="Settings"),
        AlignmentNode(
            "dark-mode",
            "switch",
            (button_x, 80, button_x + 160, 126),
            label=button_label,
            interactive=True,
            states=(("checked", "false"),),
        ),
        AlignmentNode("clock", "text", (150, 20, 180, 40), label="clock", dynamic=True),
    )


def observations(*, moving_button: bool = False, changed_label: bool = False, full_change: bool = False):
    out = []
    for index, timestamp in enumerate((0.0, 100.0, 220.0)):
        button_x = 20 + (5 if moving_button and index == 2 else 0)
        label = "Night mode" if changed_label and index == 2 else "Dark mode"
        background = (20, 20, 20) if full_change and index == 2 else (245, 245, 245)
        out.append(
            FrameObservation(
                png_frame(spinner=index, background=background, button_x=button_x),
                nodes(button_x=button_x, button_label=label),
                timestamp,
                event_idle=True,
            )
        )
    return out


def test_temporal_certificate_accepts_localized_volatility():
    artifact = certify_alignment(observations())
    cert = artifact.certificate
    assert cert.accepted
    assert cert.sample_count == 3
    assert cert.inferred_dynamic_keys == ("clock",)
    assert 0 < cert.dynamic_pixel_fraction < 0.18
    assert cert.static_mean_difference == 0
    assert artifact.representative_png.startswith(b"\x89PNG")
    assert artifact.volatility_mask_png.startswith(b"\x89PNG")


def test_certificate_is_deterministic():
    first = certify_alignment(observations())
    second = certify_alignment(observations())
    assert first.certificate.certificate_id == second.certificate.certificate_id
    assert first.representative_png == second.representative_png
    assert first.volatility_mask_png == second.volatility_mask_png


def test_interactive_geometry_shift_refuses_even_when_pixels_are_maskable():
    with pytest.raises(AlignmentRefused, match="interactive geometry moved"):
        certify_alignment(observations(moving_button=True))


def test_interactive_semantic_change_refuses():
    with pytest.raises(AlignmentRefused, match="stable node semantics changed"):
        certify_alignment(observations(changed_label=True))


def test_large_visual_change_refuses():
    with pytest.raises(AlignmentRefused, match="volatile pixels cover"):
        certify_alignment(observations(full_change=True))


def test_event_idle_can_be_required():
    samples = observations()
    samples[1] = FrameObservation(samples[1].png_bytes, samples[1].nodes, 100.0, event_idle=False)
    with pytest.raises(AlignmentRefused, match="event stream"):
        certify_alignment(samples, AlignmentPolicy(require_event_idle=True))


def test_alignment_bundle_has_hash_manifest(tmp_path):
    artifact = certify_alignment(observations())
    stage_alignment(artifact, tmp_path)
    manifest = json.loads((tmp_path / "alignment_manifest.json").read_text())
    assert manifest["certificate_id"] == artifact.certificate.certificate_id
    assert set(manifest["files"]) == {
        "aligned_surface.png",
        "alignment_certificate.json",
        "volatility_mask.png",
    }
