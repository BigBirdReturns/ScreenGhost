from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.generic_utility.phone_grammar import PhoneGrammar
from experiments.generic_utility.phone_world import PhoneWorld
from experiments.generic_utility.schema import EvidenceSource
from experiments.generic_utility.transaction import (
    ObservationContext,
    PendingActionError,
    SettlementPolicy,
    TransactionError,
    TransactionalController,
)
from experiments.generic_utility.visual_index import VisualIndexPolicy, VisualStateIndex


def setup_controller():
    world = PhoneWorld(start_app="settings", start_screen="home")
    index = VisualStateIndex(policy=VisualIndexPolicy(minimum_margin=0.05))
    for screen in ("home", "display", "saved"):
        world.reset(app_family="settings", screen_name=screen)
        frame = world.teacher_snapshot()
        index.add(frame.png_bytes, frame.runtime_projection)
    world.start_task("settings_dark_mode")
    tracker = {"screen": "home"}

    def observe(context: ObservationContext | None = None):
        expected = context.expected_screen if context and context.expected_screen else tracker["screen"]
        key = context.previous_screen_key if context and not context.expected_screen else None
        match = index.match(
            world.capture_png(),
            app_family_hint="settings",
            screen_name_hint=expected,
            screen_key_hint=key,
        )
        if match.known:
            tracker["screen"] = match.projection["screen_name"]
        return match.to_student_observation()

    controller = TransactionalController(
        world,
        observe,
        policy=SettlementPolicy(timeout_ms=2000, poll_interval_ms=80, stable_samples=3, minimum_stable_ms=160),
        now_ms=lambda: world.tick_ms,
    )
    first = observe()
    goal = PhoneWorld.task_catalog()["settings_dark_mode"].goals[0]
    action = PhoneGrammar().resolve(goal, first, app_specific_memory=True).action
    assert action is not None
    return world, controller, action


def test_pending_settle_verify_commits():
    world, controller, action = setup_controller()
    receipt = controller.execute(action, idempotency_key="first")
    assert receipt.committed and receipt.status == "verified"
    assert world.screen_name == "display" and world.actions_injected == 1


def test_second_action_rejected_while_pending():
    _world, controller, action = setup_controller()
    controller.begin(action, idempotency_key="first")
    with pytest.raises(PendingActionError):
        controller.begin(action, idempotency_key="second")
    assert controller.counters.pending_overlap_rejections == 1


def test_idempotency_replay_does_not_reinject():
    world, controller, action = setup_controller()
    first = controller.execute(action, idempotency_key="same")
    second = controller.execute(action, idempotency_key="same")
    assert first.receipt_id == second.receipt_id
    assert world.actions_injected == 1


def test_teacher_dependent_action_refused_before_motor():
    world, controller, action = setup_controller()
    poisoned = replace(action, evidence_sources=(EvidenceSource.TEACHER.value,))
    with pytest.raises(TransactionError):
        controller.execute(poisoned, idempotency_key="poisoned")
    assert world.actions_injected == 0


def test_overlay_causes_postcondition_abort_without_retry():
    world, controller, action = setup_controller()
    # The first screen's target remains visible; move the action point off target
    # to prove a motor refusal does not become a retry loop.
    bad = replace(action, normalized_point=(0.99, 0.99))
    receipt = controller.execute(bad, idempotency_key="bad")
    assert not receipt.committed and not receipt.injected
    assert world.actions_injected == 0


def test_observation_context_carries_expected_transition():
    world, controller, action = setup_controller()
    seen = []
    original = controller.observe

    def wrapped(context=None):
        seen.append(context)
        return original(context)

    controller.observe = wrapped
    controller._observe_accepts_context = True
    assert controller.execute(action, idempotency_key="ctx").committed
    assert any(context and context.expected_screen == "display" for context in seen)
