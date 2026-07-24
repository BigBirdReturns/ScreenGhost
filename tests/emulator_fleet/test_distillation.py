from experiments.emulator_fleet.campaign import record_coordinate_demonstration
from experiments.emulator_fleet.distill import distill_macro
from experiments.emulator_fleet.simulated import SimulatedFleetSpec, SimulatedInstanceAdapter
from experiments.generic_utility.phone_world import DisplayVariant


def test_coordinate_demonstration_compiles_to_semantic_targets():
    leader = SimulatedInstanceAdapter(
        SimulatedFleetSpec("leader", "Leader", DisplayVariant()), seed=8
    )
    macro, _snapshots = record_coordinate_demonstration(leader)
    leader.reset_task("settings_dark_mode")
    procedure, receipt = distill_macro(macro, leader, app_family="settings")
    assert receipt.steps_compiled == 3
    assert [step.target.label for step in procedure.steps] == [
        "Display",
        "Dark mode",
        "Save",
    ]
    assert procedure.steps[1].expected_state_key == "checked"
    assert procedure.steps[1].expected_state_value == "true"
    assert receipt.teacher_reads > 0
