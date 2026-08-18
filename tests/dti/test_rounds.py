from collections import deque

import pytest

from experiments.dti.rounds import (
    BoundedGameController,
    DTIRoundTracker,
    PendingActionError,
    PhaseClassifier,
    RoundSignals,
    SettlementPolicy,
)
from experiments.dti.schema import (
    ControlMode,
    DTIObservation,
    GameAction,
    GameActionKind,
    PlayVenue,
    RoundPhase,
    RunContext,
)


def obs(phase: RoundPhase, *, label: str | None = None, unknown: bool = False) -> DTIObservation:
    return DTIObservation.build(
        phase=phase,
        confidence=0.99 if not unknown else 0.1,
        visible_labels=(() if label is None else (label,)),
        unknown=unknown,
    )


def test_phase_classifier_uses_freeplay_and_runway_signals() -> None:
    classifier = PhaseClassifier()
    freeplay = classifier.classify(
        RoundSignals(texts=("Freeplay Mode",), freeplay_marker=True)
    )
    runway = classifier.classify(
        RoundSignals(labels=("Walk the Runway",), freeplay_marker=True)
    )
    assert freeplay.phase is RoundPhase.FREEPLAY
    assert runway.phase is RoundPhase.RUNWAY_READY


def test_tracker_halts_after_unknown_budget() -> None:
    tracker = DTIRoundTracker(unknown_limit=2)
    tracker.update(obs(RoundPhase.FREEPLAY))
    first = tracker.update(obs(RoundPhase.UNKNOWN, unknown=True))
    second = tracker.update(obs(RoundPhase.UNKNOWN, unknown=True))
    assert not first.accepted
    assert second.current is RoundPhase.HALTED


def test_tracker_enters_recovery_on_illegal_transition() -> None:
    tracker = DTIRoundTracker(stable_samples=1)
    tracker.update(obs(RoundPhase.DRESSING))
    transition = tracker.update(obs(RoundPhase.FREEPLAY))
    assert not transition.accepted
    assert transition.current is RoundPhase.RECOVERY


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, duration_ms: int) -> None:
        self.value += duration_ms


class FakeMotor:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[tuple] = []

    def press(self, keys):
        self.calls.append(("press", tuple(keys)))

    def hold(self, keys, duration_ms):
        self.calls.append(("hold", tuple(keys), duration_ms))

    def click_normalized(self, x, y):
        self.calls.append(("click", x, y))

    def move_mouse(self, dx, dy):
        self.calls.append(("move", dx, dy))

    def scroll(self, delta):
        self.calls.append(("scroll", delta))

    def wait(self, duration_ms):
        self.calls.append(("wait", duration_ms))
        self.clock.advance(duration_ms)


class ObservationQueue:
    def __init__(self, values):
        self.values = deque(values)
        self.last = values[-1]

    def __call__(self):
        if self.values:
            self.last = self.values.popleft()
        return self.last


def safe_context() -> RunContext:
    return RunContext(venue=PlayVenue.FREEPLAY, mode=ControlMode.AUTONOMOUS)


def move_action() -> GameAction:
    return GameAction(
        action_id="move-dressing-room",
        kind=GameActionKind.HOLD,
        semantic_label="move_to_dressing_room",
        keys=("w",),
        duration_ms=200,
        expected_phase=RoundPhase.FREEPLAY,
    )


def test_controller_commits_once_after_visible_settlement() -> None:
    clock = FakeClock()
    motor = FakeMotor(clock)
    observations = ObservationQueue(
        [
            obs(RoundPhase.FREEPLAY, label="start"),
            obs(RoundPhase.FREEPLAY, label="arrived"),
            obs(RoundPhase.FREEPLAY, label="arrived"),
            obs(RoundPhase.FREEPLAY, label="arrived"),
        ]
    )
    controller = BoundedGameController(
        motor,
        observations,
        settlement=SettlementPolicy(
            timeout_ms=1000,
            poll_interval_ms=100,
            stable_samples=2,
            minimum_stable_ms=100,
        ),
        now_ms=clock.now,
    )
    receipt = controller.execute(safe_context(), move_action(), idempotency_key="round-1-step-1")
    assert receipt.committed
    assert receipt.status == "verified"
    assert motor.calls[0][0] == "hold"
    assert controller.actions_injected == 1

    duplicate = controller.execute(safe_context(), move_action(), idempotency_key="round-1-step-1")
    assert duplicate.receipt_id == receipt.receipt_id
    assert controller.actions_injected == 1
    assert controller.duplicate_requests == 1


def test_controller_rejects_pending_overlap() -> None:
    clock = FakeClock()
    motor = FakeMotor(clock)
    observations = ObservationQueue([obs(RoundPhase.FREEPLAY), obs(RoundPhase.FREEPLAY)])
    controller = BoundedGameController(motor, observations, now_ms=clock.now)
    pending = controller.begin(safe_context(), move_action(), idempotency_key="one")
    assert not hasattr(pending, "receipt_id")
    with pytest.raises(PendingActionError):
        controller.begin(safe_context(), move_action(), idempotency_key="two")


def test_controller_records_policy_block_without_injection() -> None:
    clock = FakeClock()
    motor = FakeMotor(clock)
    observations = ObservationQueue([obs(RoundPhase.FREEPLAY)])
    controller = BoundedGameController(motor, observations, now_ms=clock.now)
    blocked = RunContext(
        venue=PlayVenue.PUBLIC_SERVER,
        mode=ControlMode.AUTONOMOUS,
    )
    receipt = controller.execute(blocked, move_action(), idempotency_key="blocked")
    assert receipt.status == "policy_blocked"
    assert not receipt.injected
    assert not motor.calls
