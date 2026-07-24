from __future__ import annotations

from core.surface_perception import (
    InferenceKind,
    PerceptionPolicy,
    PerceptionRequest,
    PerceptionTier,
    RunScopedInferenceQueue,
    route_perception,
)


def test_known_screen_stays_off_gpu():
    request = PerceptionRequest.create(
        "toggle dark mode",
        "screen-a",
        atlas_confidence=0.98,
        novelty=0.05,
        changed_fraction=0.02,
    )
    decision = route_perception(request)
    assert decision.selected_tier is PerceptionTier.ATLAS
    assert decision.attempted_tiers == (PerceptionTier.ATLAS,)


def test_uncertain_target_escalates_to_small_grounder_before_large_vlm():
    request = PerceptionRequest.create(
        "open settings",
        "unknown",
        atlas_confidence=0.2,
        prototype_confidence=0.4,
        small_grounder_confidence=0.83,
        large_vlm_confidence=0.95,
        novelty=0.8,
        changed_fraction=0.5,
    )
    decision = route_perception(request)
    assert decision.selected_tier is PerceptionTier.SMALL_GROUNDER
    assert PerceptionTier.LARGE_VLM not in decision.attempted_tiers


def test_teacher_blind_failure_does_not_consult_teacher():
    request = PerceptionRequest.create("unknown", "unknown", teacher_blind=True)
    decision = route_perception(request, PerceptionPolicy(allow_teacher_review=True))
    assert decision.accepted is False
    assert PerceptionTier.TEACHER_REVIEW not in decision.attempted_tiers


def test_run_scoped_queue_batches_by_kind_and_deduplicates():
    calls = []

    def runner(job):
        calls.append((job["kind"], job["identity"]))
        return {"ok": True}

    kinds = [
        InferenceKind("ground", lambda: {"ready": True}, runner=runner),
        InferenceKind("describe", lambda: {"ready": True}, runner=runner),
    ]
    queue = RunScopedInferenceQueue(kinds)
    assert queue.enqueue("ground", "a", lane="warm") is not None
    assert queue.enqueue("describe", "b", lane="interactive") is not None
    assert queue.enqueue("ground", "c", lane="interactive") is not None
    assert queue.enqueue("ground", "a", lane="warm") is None
    queue.enqueue("describe", "d", lane="interactive")
    result = queue.drain()
    assert result["ran"] == 4
    assert calls == [
        ("describe", "b"),
        ("describe", "d"),
        ("ground", "c"),
        ("ground", "a"),
    ]
