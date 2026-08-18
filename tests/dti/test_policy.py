from experiments.dti.policy import DTISafetyPolicy
from experiments.dti.schema import (
    ControlMode,
    GameAction,
    GameActionKind,
    PlayVenue,
    RunContext,
)


def safe_action(label: str = "move_to_dressing_room") -> GameAction:
    return GameAction(
        action_id="move-1",
        kind=GameActionKind.HOLD,
        semantic_label=label,
        keys=("w",),
        duration_ms=250,
    )


def test_freeplay_autonomous_practice_is_allowed_with_human_present() -> None:
    context = RunContext(venue=PlayVenue.FREEPLAY, mode=ControlMode.AUTONOMOUS)
    decision = DTISafetyPolicy().evaluate(context, safe_action())
    assert decision.allowed
    assert decision.warnings


def test_emulator_and_modified_client_are_blocked() -> None:
    context = RunContext(
        venue=PlayVenue.FREEPLAY,
        mode=ControlMode.FAMILY_COPILOT,
        emulator=True,
        modified_client=True,
    )
    decision = DTISafetyPolicy().evaluate(context, safe_action())
    assert not decision.allowed
    assert any("emulator" in reason for reason in decision.reasons)
    assert any("modified" in reason for reason in decision.reasons)


def test_public_autonomous_is_blocked() -> None:
    context = RunContext(venue=PlayVenue.PUBLIC_SERVER, mode=ControlMode.AUTONOMOUS)
    assert not DTISafetyPolicy().evaluate(context, safe_action()).allowed


def test_public_family_copilot_is_allowed_but_warned() -> None:
    context = RunContext(venue=PlayVenue.PUBLIC_SERVER, mode=ControlMode.FAMILY_COPILOT)
    decision = DTISafetyPolicy().evaluate(context, safe_action())
    assert decision.allowed
    assert any("public-server" in warning for warning in decision.warnings)


def test_voting_purpose_is_structurally_blocked() -> None:
    context = RunContext(venue=PlayVenue.PRIVATE_SERVER, mode=ControlMode.ASSISTED)
    decision = DTISafetyPolicy().evaluate(context, safe_action("vote_for_player"))
    assert not decision.allowed
    assert any("purpose" in reason for reason in decision.reasons)


def test_unallowlisted_key_is_blocked() -> None:
    context = RunContext(venue=PlayVenue.FREEPLAY, mode=ControlMode.FAMILY_COPILOT)
    action = GameAction(
        action_id="bad-key",
        kind=GameActionKind.PRESS,
        semantic_label="system_shortcut",
        keys=("f4",),
    )
    assert not DTISafetyPolicy().evaluate(context, action).allowed
